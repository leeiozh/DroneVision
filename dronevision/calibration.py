"""Checks of the camera model against the video itself.

* ``estimate_heading`` — heading (and pitch/roll) correction from OSM waterlines visible in a frame.
  FlightRecord heading can be wrong by >10° after a takeoff from a boat and is only corrected by
  the flight controller during fast flight (DJI_0371: -12° for the whole hover).
* ``check_sync`` — log/video time offset and focal length from global image motion, plus heading
  jumps: moments when ``osd.yaw`` changes while the picture does not rotate.
"""
from __future__ import annotations

from pathlib import Path
import struct
import subprocess

import numpy as np

from .geometry import camera_basis, project_to_image
from .pose import enu_offsets, frame_poses, load_context
from .cvio import imwrite, open_video
from .paths import tool
from .projection import LOCAL_COASTLINE, _dependencies

WORK_WIDTH = 960          # heading fit resolution
SEARCH_PX = 40            # search window along the shore normal, work pixels
SHORE_STEP_M = 6.0


# --- heading from coastline -------------------------------------------------------------------

def _land_rings(path: Path, bbox) -> list[tuple[np.ndarray, bool]]:
    """Polygon rings (lon, lat) with the outer-ring flag; ESRI outer rings are clockwise."""
    data = path.read_bytes()
    west, south, east, north = bbox
    rings, offset = [], 100
    while offset + 8 <= len(data):
        _, words = struct.unpack_from(">2i", data, offset)
        offset += 8
        end = offset + words * 2
        if end > len(data):
            break
        if struct.unpack_from("<i", data, offset)[0] == 5:
            xmin, ymin, xmax, ymax = struct.unpack_from("<4d", data, offset + 4)
            if xmax >= west and xmin <= east and ymax >= south and ymin <= north:
                parts_count, points_count = struct.unpack_from("<2i", data, offset + 36)
                parts = list(struct.unpack_from(f"<{parts_count}i", data, offset + 44)) + [points_count]
                points = np.frombuffer(data, "<f8", 2 * points_count, offset + 44 + 4 * parts_count).reshape(-1, 2)
                for start, stop in zip(parts[:-1], parts[1:]):
                    ring = points[start:stop]
                    area = 0.5 * np.sum(ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1])
                    rings.append((ring, area < 0))
        offset = end
    return rings


def _shore_points(rings_enu, step_m=SHORE_STEP_M):
    points, normals = [], []
    for ring, outer in rings_enu:
        length = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(ring, axis=0), axis=1))])
        if length[-1] <= step_m:
            continue
        s = np.arange(0, length[-1], step_m)
        dense = np.column_stack([np.interp(s, length, ring[:, 0]), np.interp(s, length, ring[:, 1])])
        tangent = np.gradient(dense, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
        right = np.column_stack([tangent[:, 1], -tangent[:, 0]])
        points.append(dense)
        normals.append(right if outer else -right)          # towards land
    return np.concatenate(points), np.concatenate(normals)


def _clip_front(polygon, direction, near):
    out = []
    for p, q in zip(polygon, np.roll(polygon, -1, axis=0)):
        dp, dq = p @ direction - near, q @ direction - near
        if dp >= 0:
            out.append(p)
        if (dp >= 0) != (dq >= 0):
            out.append(p + dp / (dp - dq) * (q - p))
    return np.asarray(out) if len(out) >= 3 else None


class _HeadingProblem:
    def __init__(self, cv2, frame, pose, rings_enu, max_range_m):
        self.cv2 = cv2
        self.height, self.width = frame.shape[:2]
        self.scale = WORK_WIDTH / self.width
        small = cv2.resize(frame, (WORK_WIDTH, int(round(self.height * self.scale))), interpolation=cv2.INTER_AREA)
        self.small = small
        lab = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32), (0, 0), 1.5)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean = cv2.blur(gray, (7, 7))
        texture = cv2.GaussianBlur(np.log1p(np.sqrt(np.maximum(cv2.blur(gray * gray, (7, 7)) - mean * mean, 0))), (0, 0), 2)
        b, g, _ = cv2.split(cv2.GaussianBlur(small.astype(np.float32), (0, 0), 1.5))
        self.features = np.dstack([lab, texture * 20, g - b]).reshape(-1, 5)
        self.pose = pose
        self.rings = [(ring - pose["offset"], outer) for ring, outer in rings_enu]
        points, normals = _shore_points(self.rings)
        distance = np.linalg.norm(points, axis=1)
        keep = (distance > 100) & (distance < max_range_m)
        self.points, self.normals, self.distance = points[keep], normals[keep], distance[keep]
        self.max_range_m = max_range_m

    def basis(self, x):
        return camera_basis(self.pose["yaw"] + x[0], self.pose["pitch"] + x[1], self.pose["roll"] + x[2])

    def land_mask(self, basis):
        h, w = int(round(self.height * self.scale)), WORK_WIDTH
        mask = np.zeros((h, w), np.uint8)
        forward = basis[0][:2] / max(np.linalg.norm(basis[0][:2]), 1e-9)
        for ring, outer in self.rings:
            clipped = _clip_front(ring, forward, 30.0)
            if clipped is None:
                continue
            clipped = _clip_front(clipped, -forward, -1.5 * self.max_range_m)
            if clipped is None:
                continue
            uv, depth = project_to_image(clipped, self.pose["altitude"], self.width, self.height, self.pose["focal_px"], basis)
            if np.any(depth <= 0):
                continue
            q = np.clip(np.round(uv * self.scale * 4), -1e6, 1e6).astype(np.int32)
            self.cv2.fillPoly(mask, [q.reshape(-1, 1, 2)], 1 if outer else 0, self.cv2.LINE_8, shift=2)
        return mask

    def sea_region(self, basis, max_distance):
        h, w = int(round(self.height * self.scale)), WORK_WIDTH
        ys, xs = np.mgrid[0:h, 0:w]
        forward, right, down = basis
        x = (xs / self.scale - self.width / 2) / self.pose["focal_px"]
        y = (ys / self.scale - self.height / 2) / self.pose["focal_px"]
        rz = forward[2] + x * right[2] + y * down[2]
        rh = np.hypot(forward[0] + x * right[0] + y * down[0], forward[1] + x * right[1] + y * down[1])
        return np.where(rz < 0, self.pose["altitude"] * rh / np.maximum(-rz, 1e-9), np.inf) < max_distance

    def land_probability(self, x):
        """Linear discriminant trained on OSM labels far from the projected shore (robust to label noise)."""
        cv2 = self.cv2
        basis = self.basis(x)
        labels = self.land_mask(basis)
        valid = self.sea_region(basis, 1.5 * self.max_range_m)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        land = (cv2.erode(labels, kernel).astype(bool) & valid).ravel()
        water = (cv2.erode(1 - labels, kernel).astype(bool) & valid).ravel()
        if land.sum() < 200 or water.sum() < 200:
            return None
        a, b = self.features[land], self.features[water]
        covariance = (np.cov(a.T) * len(a) + np.cov(b.T) * len(b)) / (len(a) + len(b)) + np.eye(5) * 1e-3
        weights = np.linalg.solve(covariance, a.mean(0) - b.mean(0))
        score = self.features @ weights - 0.5 * (a.mean(0) + b.mean(0)) @ weights + np.log(len(a) / len(b))
        probability = (1 / (1 + np.exp(-np.clip(score, -30, 30)))).reshape(labels.shape).astype(np.float32)
        return cv2.GaussianBlur(probability, (0, 0), 1.0)

    def facing(self, x):
        """Shore samples whose water side faces the camera: only there the waterline is not hidden by relief."""
        basis = self.basis(x)
        towards_camera = -self.points / self.distance[:, None]
        uv, depth = project_to_image(self.points, self.pose["altitude"], self.width, self.height, self.pose["focal_px"], basis)
        inside = (depth > 0) & (uv[:, 0] > 20) & (uv[:, 0] < self.width - 20) & (uv[:, 1] > 20) & (uv[:, 1] < self.height - 20)
        return inside & (-(self.normals * towards_camera).sum(1) > 0.15)

    def residuals(self, x, probability, selected):
        """Signed offset (work px) along the projected land normal to the water->land crossing."""
        basis = self.basis(x)
        points, normals = self.points[selected], self.normals[selected]
        p0, _ = project_to_image(points, self.pose["altitude"], self.width, self.height, self.pose["focal_px"], basis)
        p1, _ = project_to_image(points + 3.0 * normals, self.pose["altitude"], self.width, self.height, self.pose["focal_px"], basis)
        p0, p1 = p0 * self.scale, p1 * self.scale
        direction = p1 - p0
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
        offsets = np.arange(-SEARCH_PX, SEARCH_PX + 1, dtype=np.float32)
        samples = p0[:, None, :] + offsets[None, :, None] * direction[:, None, :]
        values = self.cv2.remap(probability, samples[..., 0].astype(np.float32), samples[..., 1].astype(np.float32),
                                self.cv2.INTER_LINEAR, borderMode=self.cv2.BORDER_CONSTANT, borderValue=-1)
        water = (values >= 0) & (values < 0.5)
        land = values >= 0.5
        # crossing j -> j+1 with >= 5 water samples before and >= 4 land samples after
        run_water = np.lib.stride_tricks.sliding_window_view(water, 5, axis=1).all(axis=2)[:, :-4]
        run_land = np.lib.stride_tricks.sliding_window_view(land, 4, axis=1).all(axis=2)[:, 5:]
        crossing = run_water & run_land                               # index k => crossing between k+4 and k+5
        centre = offsets[4:4 + crossing.shape[1]]
        distance = np.where(crossing, np.abs(centre)[None, :], np.inf)
        best = np.argmin(distance, axis=1)
        rows = np.arange(len(best))
        a, b = values[rows, best + 4], values[rows, best + 5]
        result = centre[best] + (0.5 - a) / np.maximum(b - a, 1e-6)
        return np.where(np.isfinite(distance[rows, best]), result, np.nan)

    def normal_shift(self, x, selected, axis, delta_deg=0.5):
        """Work pixels per degree that each shore sample moves across the shore when parameter ``axis`` changes."""
        points, normals = self.points[selected], self.normals[selected]
        step = np.zeros(3); step[axis] = delta_deg
        args = (self.pose["altitude"], self.width, self.height, self.pose["focal_px"])
        p0, _ = project_to_image(points, *args, self.basis(x))
        p1, _ = project_to_image(points, *args, self.basis(x + step))
        pn, _ = project_to_image(points + 3.0 * normals, *args, self.basis(x))
        direction = pn - p0
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
        return np.abs(np.sum((p1 - p0) * direction, axis=1)) * self.scale / delta_deg

    @staticmethod
    def cost(residuals):
        r = np.where(np.isfinite(residuals), residuals, SEARCH_PX)
        return float(np.mean(36.0 * np.log1p((r / 6.0) ** 2)))


def _nelder_mead(function, x0, step, iterations=250, tolerance=1e-5):
    simplex = [np.asarray(x0, float)] + [np.asarray(x0, float) + np.eye(len(x0))[i] * step[i] for i in range(len(x0))]
    values = [function(x) for x in simplex]
    for _ in range(iterations):
        order = np.argsort(values)
        simplex, values = [simplex[i] for i in order], [values[i] for i in order]
        if abs(values[-1] - values[0]) < tolerance:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        reflected = centroid + (centroid - simplex[-1]); fr = function(reflected)
        if fr < values[0]:
            expanded = centroid + 2 * (centroid - simplex[-1]); fe = function(expanded)
            simplex[-1], values[-1] = (expanded, fe) if fe < fr else (reflected, fr)
        elif fr < values[-2]:
            simplex[-1], values[-1] = reflected, fr
        else:
            contracted = centroid + 0.5 * (simplex[-1] - centroid); fc = function(contracted)
            if fc < values[-1]:
                simplex[-1], values[-1] = contracted, fc
            else:
                simplex = [simplex[0]] + [simplex[0] + 0.5 * (s - simplex[0]) for s in simplex[1:]]
                values = [values[0]] + [function(s) for s in simplex[1:]]
    best = int(np.argmin(values))
    return simplex[best], values[best]


def estimate_heading(video_path: str, time_s: float, takeoff_height_m: float = 0.0, horizontal_fov_deg: float = 0.0,
                     max_range_m: float = 3500.0, search_deg: float = 30.0, output_dir: str | None = None) -> dict:
    """Fit heading/pitch/roll corrections for one frame by aligning OSM waterlines with the image."""
    cv2, geod = _dependencies()
    context = load_context(video_path)
    time_s = float(np.clip(time_s, 0, context.meta["duration"]))
    pose = {k: v[0] for k, v in frame_poses(context, [time_s], geod, 0.0, takeoff_height_m, horizontal_fov_deg).items()}
    from .coastdata import land_file
    lat, lon = float(pose["lat"]), float(pose["lon"])
    dlat = 1.6 * max_range_m / 111_000
    dlon = dlat / np.cos(np.deg2rad(lat))
    land_path = land_file((lon - dlon, lat - dlat, lon + dlon, lat + dlat))
    if land_path is None:
        raise ValueError("Для этого района нет береговой линии: скачайте её командой «python -m dronevision coastline»")
    rings = _land_rings(land_path, (lon - dlon, lat - dlat, lon + dlon, lat + dlat))
    if not rings:
        raise ValueError("В окрестности кадра нет береговой линии")
    lat0, lon0 = context.takeoff
    rings_enu = [(enu_offsets(geod, lat0, lon0, ring[:, 1], ring[:, 0]), outer) for ring, outer in rings]

    cap = open_video(context.video)
    try:
        frame_index = int(round(time_s * context.meta["fps"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise RuntimeError(f"Не удалось прочитать кадр {frame_index}")
    problem = _HeadingProblem(cv2, frame, pose, rings_enu, max_range_m)

    def evaluate(x, probability, selected):
        return problem.cost(problem.residuals(x, probability, selected))

    # Coarse heading scan: the log can be >10° off, beyond the local search window. The land classifier is
    # trained once with the logged heading; retraining it per candidate lets wrong headings "explain" haze/sky.
    probability = problem.land_probability(np.zeros(3))
    if probability is None:
        raise ValueError("В кадре не видно берега, обращённого к камере")
    scan = []
    for yaw in np.arange(-search_deg, search_deg + 0.1, 1.0):
        x = np.array([yaw, 0.0, 0.0])
        selected = problem.facing(x)
        if selected.sum() >= 30:
            scan.append((evaluate(x, probability, selected), yaw))
    if not scan:
        raise ValueError("В кадре не видно берега, обращённого к камере")
    x = np.array([min(scan)[1], 0.0, 0.0])
    for step in (np.array([1.5, 0.8, 0.8]), np.array([0.6, 0.4, 0.4]), np.array([0.3, 0.2, 0.2])):
        retrained, selected = problem.land_probability(x), problem.facing(x)
        probability = probability if retrained is None else retrained
        x, _ = _nelder_mead(lambda v: evaluate(v, probability, selected), x, step)
    residuals = problem.residuals(x, probability, selected)
    matched = int(np.isfinite(residuals).sum())
    median_px = float(np.nanmedian(np.abs(residuals)) / problem.scale) if matched else float("nan")
    # Heading is observable only where the shore normal has a horizontal image component: a distant shore
    # along the horizon moves along itself when the heading changes.
    yaw_sensitive = int(np.sum(np.isfinite(residuals) & (problem.normal_shift(x, selected, 0) >= 4.0)))
    quality = ("good" if matched >= 60 and median_px <= 30 and yaw_sensitive >= 30
               else "weak" if matched >= 20 else "failed")
    result = {"time_s": round(time_s, 3), "frame_index": frame_index,
              "yaw_offset_deg": round(float(x[0]), 2), "pitch_offset_deg": round(float(x[1]), 2),
              "roll_offset_deg": round(float(x[2]), 2), "log_yaw_deg": float(pose["log_yaw"]),
              "shore_points": int(selected.sum()), "matched_points": matched,
              "median_residual_px": round(median_px, 1), "yaw_sensitive_points": yaw_sensitive, "quality": quality,
              "coarse_scan": [[float(y), round(c, 2)] for c, y in sorted(scan, key=lambda item: item[1])]}
    if output_dir:
        overlay = problem.small.copy()
        tint = np.zeros_like(overlay); tint[..., 2] = (probability > 0.5) * 255
        overlay = cv2.addWeighted(overlay, 0.8, tint, 0.2, 0)
        for params, color in ((np.zeros(3), (0, 0, 255)), (x, (0, 255, 0))):
            uv, _ = project_to_image(problem.points[selected], pose["altitude"], problem.width, problem.height,
                                     pose["focal_px"], problem.basis(params))
            for u, v in uv * problem.scale:
                cv2.circle(overlay, (int(u), int(v)), 1, color, -1)
        path = Path(output_dir).expanduser() / f"{context.video.stem}_{time_s:08.3f}s_heading.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        imwrite(path, overlay)
        result["overlay"] = str(path)
    return result


# --- sync, focal length and heading jumps from image motion ------------------------------------

def check_sync(video_path: str, start_s: float = 0.0, end_s: float | None = None, work_width: int = 480,
               max_offset_s: float = 3.0) -> dict:
    """Compare global image shifts with log yaw/pitch rates (pure-rotation approximation)."""
    cv2, geod = _dependencies()
    context = load_context(video_path)
    meta = context.meta
    end_s = meta["duration"] if end_s is None else min(end_s, meta["duration"])
    work_height = int(round(meta["height"] * work_width / meta["width"] / 2) * 2)
    command = [tool("ffmpeg"), "-v", "error", "-ss", f"{start_s:.3f}", "-i", str(context.video), "-t", f"{end_s - start_s:.3f}",
               "-vf", f"scale={work_width}:{work_height}:flags=area,format=gray", "-f", "rawvideo", "-"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    window = cv2.createHanningWindow((work_width, work_height), cv2.CV_32F)
    previous, shifts = None, []
    size = work_width * work_height
    while True:
        buffer = process.stdout.read(size)
        if len(buffer) < size:
            break
        image = np.frombuffer(buffer, np.uint8).reshape(work_height, work_width).astype(np.float32)
        if previous is not None:
            (dx, dy), response = cv2.phaseCorrelate(previous, image, window)
            shifts.append((dx, dy, response))
        previous = image
    process.wait()
    if len(shifts) < 30:
        raise ValueError("Слишком короткий фрагмент для проверки синхронизации")
    shifts = np.asarray(shifts)
    frame_time = start_s + np.arange(1, len(shifts) + 1) / meta["fps"]
    log = context.log
    yaw = np.rad2deg(np.unwrap(np.deg2rad(log["yaw"])))

    def deltas(values, extra):
        now = np.interp(context.start_epoch + frame_time + extra, log["time"], values)
        before = np.interp(context.start_epoch + frame_time - 1 / meta["fps"] + extra, log["time"], values)
        return now - before

    poses = frame_poses(context, frame_time, geod)
    steady = (np.abs(np.gradient(poses["focal_mm"])) < 1e-6) & (shifts[:, 2] > 0.2)
    rows = []
    for extra in np.arange(-max_offset_s, max_offset_s + 1e-9, 0.02):
        d_yaw, d_pitch = deltas(yaw, extra), deltas(log["pit_2"], extra)
        c_yaw = np.corrcoef(shifts[steady, 0], d_yaw[steady])[0, 1] if np.std(d_yaw[steady]) > 1e-3 else np.nan
        c_pitch = np.corrcoef(shifts[steady, 1], d_pitch[steady])[0, 1] if np.std(d_pitch[steady]) > 1e-3 else np.nan
        rows.append((extra, c_yaw, c_pitch))
    rows = np.asarray(rows)
    # Image moves left when the camera turns right and up when it tilts down.
    score = np.nan_to_num(-rows[:, 1]) + np.nan_to_num(rows[:, 2])
    best = int(np.argmax(score))
    extra = float(rows[best, 0])
    d_yaw, d_pitch = deltas(yaw, extra), deltas(log["pit_2"], extra)
    image_scale = meta["width"] / work_width
    cos_pitch = np.cos(np.deg2rad(poses["pitch"]))
    # Horizontal image shift ~ f * dyaw * cos(pitch) only for a moderately tilted camera without tilt motion;
    # low-texture water makes phase correlation under-estimate shifts, so use strong responses only.
    usable = steady & (shifts[:, 2] > 0.4) & (np.abs(poses["pitch"]) < 45) & (np.abs(d_pitch) < 0.05)
    window_frames = max(int(round(meta["fps"])), 1)
    ratios = []
    for i in range(0, len(shifts) - window_frames, window_frames):
        segment = np.arange(i, i + window_frames)[usable[i:i + window_frames]]
        predictor = np.deg2rad(d_yaw[segment]) * cos_pitch[segment]
        if len(segment) > 0.8 * window_frames and abs(np.sum(predictor)) > np.deg2rad(2):
            ratios.append(-np.sum(shifts[segment, 0]) / np.sum(predictor) * image_scale * 24.0 / np.median(poses["focal_mm"][segment]))
    focal_24 = float(np.median(ratios)) if len(ratios) >= 3 else None
    # Heading jumps: logged yaw change without matching picture rotation over 2 s windows.
    image_yaw = np.rad2deg(-shifts[:, 0] / (poses["focal_px"] / image_scale * cos_pitch))
    image_yaw[~(steady & (np.abs(poses["pitch"]) < 45) & (np.abs(d_pitch) < 0.05))] = np.nan
    window_frames *= 2
    jumps = []
    for i in range(0, len(shifts) - window_frames, window_frames // 2):
        segment = slice(i, i + window_frames)
        if np.mean(np.isfinite(image_yaw[segment])) < 0.8:
            continue
        logged, seen = float(np.sum(d_yaw[segment])), float(np.nansum(image_yaw[segment]))
        if abs(logged - seen) > max(8.0, 0.2 * abs(logged)):
            jumps.append({"time_s": round(float(frame_time[i]), 1), "log_yaw_change_deg": round(logged, 1),
                          "image_yaw_change_deg": round(seen, 1)})
    yaw_corr = float(-rows[best, 1]) if np.isfinite(rows[best, 1]) else 0.0
    pitch_corr = float(rows[best, 2]) if np.isfinite(rows[best, 2]) else 0.0
    reliable = max(yaw_corr, pitch_corr) >= 0.6    # needs turns or gimbal tilts in the fragment
    return {"extra_log_shift_s": round(extra, 2) if reliable else None,
            "yaw_correlation": round(yaw_corr, 3), "pitch_correlation": round(pitch_corr, 3),
            "reliable": reliable,
            "note": None if reliable else "Во фрагменте нет поворотов или наклонов камеры — синхронизацию проверить нельзя",
            "current_sync": {"utc_offset_hours": context.utc_offset_hours, "clock_offset_seconds": context.clock_offset_seconds},
            "suggested_clock_offset_seconds": round(context.clock_offset_seconds - extra, 2) if reliable else None,
            "focal_px_at_24mm": None if focal_24 is None else round(focal_24),
            "model_focal_px_at_24mm": round(float(poses["focal_px"][0] * 24.0 / poses["focal_mm"][0])),
            "heading_jumps": jumps, "frames": len(shifts)}


def estimate_headings(video_path: str, times_s, takeoff_height_m: float = 0.0, horizontal_fov_deg: float = 0.0,
                      output_dir: str | None = None) -> dict:
    """Heading correction from several frames: median over frames with a usable shore match."""
    frames = []
    for time_s in times_s:
        try:
            frames.append(estimate_heading(video_path, time_s, takeoff_height_m, horizontal_fov_deg, output_dir=output_dir))
        except ValueError as exc:
            frames.append({"time_s": time_s, "quality": "failed", "error": str(exc)})
    usable = [f for f in frames if f["quality"] == "good"] or [f for f in frames if f["quality"] == "weak"]
    for frame in frames:
        frame.pop("coarse_scan", None)
    if not usable:
        return {"yaw_offset_deg": None, "frames": frames}
    offsets = np.array([f["yaw_offset_deg"] for f in usable])
    return {"yaw_offset_deg": round(float(np.median(offsets)), 2), "based_on": usable[0]["quality"],
            "spread_deg": round(float(offsets.max() - offsets.min()), 2) if len(offsets) > 1 else None,
            "pitch_offset_deg": round(float(np.median([f["pitch_offset_deg"] for f in usable])), 2),
            "roll_offset_deg": round(float(np.median([f["roll_offset_deg"] for f in usable])), 2),
            "frames": frames}


# --- logged attitude vs. image rotation --------------------------------------------------------

def check_attitude(video_path: str, start_s: float = 0.0, end_s: float | None = None, step_s: float = 0.5,
                   shifts_s=(-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3), takeoff_height_m: float = 1.0) -> dict:
    """How well FlightRecord attitude explains the image motion between frames ``step_s`` apart.

    All statistics except ``rms_pair_error_without_shift_deg`` are given at the best extra log shift.

    For every pair the attitude correction of the second frame that makes static details coincide is fitted
    (``refine.match_edge``).  Reports the best extra log time shift, the ratio of image-measured to logged
    yaw turns, and the drift of yaw/pitch/roll accumulated over the fragment (what the log misses).
    """
    import dataclasses
    from .mosaic import _read_frame
    from .refine import _pose, _residual, frame_features, match_edge

    cv2, geod = _dependencies()
    context = load_context(video_path)
    meta = context.meta
    end_s = meta["duration"] - 0.1 if end_s is None else min(end_s, meta["duration"] - 0.1)
    indices = np.unique(np.round(np.arange(start_s, end_s + 1e-6, step_s) * meta["fps"]).astype(int))
    width, height = meta["width"], meta["height"]
    poses = frame_poses(context, indices / meta["fps"], geod, 0.0, takeoff_height_m)
    cap = open_video(context.video)
    features, position = [], 10 ** 9
    try:
        for index in indices:
            frame, position = _read_frame(cv2, cap, int(index), position)
            features.append(frame_features(cv2, frame))
    finally:
        cap.release()
    pairs = []
    for k in range(len(indices) - 1):
        matched = match_edge(cv2, features[k], features[k + 1], _pose(poses, k), _pose(poses, k + 1), width, height)
        if matched is not None:
            pairs.append((k, matched[0], matched[1]))
    if len(pairs) < 3:
        raise ValueError("Слишком мало пар кадров с общими неподвижными деталями")

    def fit(pa, pb, p1, p2):
        x = np.zeros(3)
        for _ in range(6):
            def residual(v):
                return _residual(p1, p2, pa, dict(pb, yaw=pb["yaw"] + v[0], pitch=pb["pitch"] + v[1],
                                                  roll=pb["roll"] + v[2]), width, height).ravel()
            r0 = residual(x)
            jac = np.column_stack([(residual(x + 0.01 * e) - r0) / 0.01 for e in np.eye(3)])
            w = 1 / np.maximum(1, np.abs(r0) / 2)
            dx = np.linalg.lstsq(jac * w[:, None], -r0 * w, rcond=None)[0]
            x += dx
            if np.abs(dx).max() < 1e-4:
                break
        return x

    fitted = {}

    def evaluate(shift):
        shift = round(float(shift), 3)
        if shift not in fitted:
            log = dict(context.log, time=context.log["time"] + shift)
            shifted = frame_poses(dataclasses.replace(context, log=log), indices / meta["fps"], geod, 0.0,
                                  takeoff_height_m)
            fitted[shift] = np.array([fit(_pose(shifted, k), _pose(shifted, k + 1), p1, p2) for k, p1, p2 in pairs])
        rows = fitted[shift]
        return float(np.sqrt(np.mean(rows[:, 0] ** 2)) + np.sqrt(np.mean(rows[:, 1] ** 2)))

    coarse = min(shifts_s, key=evaluate)
    best = min(np.round(np.arange(coarse - 0.05, coarse + 0.0501, 0.01), 3), key=evaluate)
    scan = [(shift, *np.sqrt(np.mean(rows[:, :2] ** 2, axis=0))) for shift, rows in sorted(fitted.items())]
    rows = fitted[round(float(best), 3)]
    ks = np.array([k for k, _, _ in pairs])
    logged_turn = (np.diff(np.unwrap(np.deg2rad(poses["log_yaw"])))[ks]) * 180 / np.pi
    turning = np.abs(logged_turn) > 1.0
    ratio = (float(np.sum(logged_turn[turning] + rows[turning, 0]) / np.sum(logged_turn[turning]))
             if turning.sum() >= 3 else None)
    drift = np.cumsum(rows, axis=0)
    times = (indices[ks + 1] / meta["fps"]).round(2)
    zero = fitted.get(0.0)
    return {
        "pairs": len(pairs), "step_s": step_s,
        "best_extra_log_shift_s": float(best),
        "current_sync": {"utc_offset_hours": context.utc_offset_hours,
                         "clock_offset_seconds": context.clock_offset_seconds},
        "rms_pair_error_without_shift_deg": None if zero is None else
        dict(zip(("yaw", "pitch", "roll"), np.sqrt(np.mean(zero ** 2, axis=0)).round(3).tolist())),
        "shift_scan": [{"shift_s": s, "rms_yaw_deg": round(float(y), 3), "rms_pitch_deg": round(float(p), 3)}
                       for s, y, p in scan],
        "rms_pair_error_deg": dict(zip(("yaw", "pitch", "roll"), np.sqrt(np.mean(rows ** 2, axis=0)).round(3).tolist())),
        "image_to_log_yaw_turn_ratio": None if ratio is None else round(ratio, 4),
        "accumulated_drift_deg": {"yaw": round(float(drift[-1, 0]), 2), "pitch": round(float(drift[-1, 1]), 2),
                                  "roll": round(float(drift[-1, 2]), 2)},
        "max_abs_drift_deg": dict(zip(("yaw", "pitch", "roll"), np.abs(drift).max(axis=0).round(2).tolist())),
        "drift_series": [[float(t), *np.round(d, 3).tolist()] for t, d in zip(times, drift)],
        "note": ("Ориентация из лога годится как среднее, но не для совмещения кадров между собой: "
                 "включайте выравнивание по изображению") if np.abs(drift).max() > 0.2 else "",
    }
