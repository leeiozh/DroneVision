"""Is there a coast in the frame, where should it be drawn, and what camera attitude makes it fit.

``coast_view``   — land outlines around the drone as viewing directions (azimuth, depression), the camera
                   of the frame and whether the outlines fall inside it.  The browser projects the
                   directions itself, so the overlay follows the video and can be dragged.
``align``        — automatic alignment of one frame: pitch and roll from the sea horizon (``horizon.py``),
                   heading from the waterlines of the land in view.

Directions are computed for sea-level waterlines with the curvature of the Earth and standard refraction:
a shore 20 km away from 200 m height is seen 0.7° below the horizontal, the sea horizon 0.43° below it.
Relief is not known: only the waterline facing the camera is expected to match (see docs/projection.md).
"""
from __future__ import annotations

import numpy as np

from . import coastdata, horizon
from .cvio import open_video
from .geometry import camera_basis
from .pose import frame_poses, load_context, pose_offsets, save_pose_offsets

MAX_VIEW_M = 90_000.0          # nothing at sea level is seen farther from a drone below 500 m


def _dependencies():
    import cv2
    from pyproj import Geod
    return cv2, Geod(ellps="WGS84")


def _pose(context, time_s: float, geod, yaw_offset_deg: float, takeoff_height_m: float, horizontal_fov_deg: float) -> dict:
    time_s = float(np.clip(time_s, 0, context.meta["duration"]))
    return {k: v[0] for k, v in frame_poses(context, [time_s], geod, yaw_offset_deg, takeoff_height_m,
                                            horizontal_fov_deg).items()}


def view_limit_m(altitude_m: float) -> float:
    """Sea horizon distance plus what low land beyond it can add (cliffs and hills of ~60 m)."""
    return min(coastdata.horizon_distance_m(altitude_m) + coastdata.horizon_distance_m(60.0), MAX_VIEW_M)


def land_directions(lat: float, lon: float, altitude_m: float, geod, create: bool = True):
    """Land rings around a point as arrays [azimuth°, depression°, distance m]; None without data."""
    limit = view_limit_m(altitude_m)
    box = coastdata.box_around(lat, lon, limit)
    path = coastdata.land_file(box, create)
    if path is None:
        return None, None
    rings = []
    for ring in coastdata.read_rings(path, box):
        azimuth, _, distance = geod.inv(np.full(len(ring), lon), np.full(len(ring), lat), ring[:, 0], ring[:, 1])
        distance = np.maximum(distance, 1.0)
        if distance.min() > limit:
            continue
        depression = np.rad2deg(np.arctan((altitude_m + distance ** 2 / (2 * coastdata.EFFECTIVE_RADIUS_M)) / distance))
        rings.append(np.column_stack([azimuth % 360, depression, distance]))
    return rings, path


def to_pixels(directions: np.ndarray, yaw: float, pitch: float, roll: float, focal_px: float, width: int, height: int):
    """Pixels (u, v) and depth sign of viewing directions [azimuth°, depression°]."""
    azimuth, depression = np.deg2rad(directions[:, 0]), np.deg2rad(directions[:, 1])
    vectors = np.column_stack([np.sin(azimuth) * np.cos(depression), np.cos(azimuth) * np.cos(depression),
                               -np.sin(depression)])
    forward, right, down = camera_basis(yaw, pitch, roll)
    depth = vectors @ forward
    with np.errstate(divide="ignore", invalid="ignore"):
        u = width / 2 + focal_px * (vectors @ right) / depth
        v = height / 2 + focal_px * (vectors @ down) / depth
    return u, v, depth


def _thin(ring: np.ndarray, step_deg: float) -> np.ndarray:
    """Drop vertices closer than ``step_deg`` on the sky to the last kept one (about half a pixel)."""
    keep, last = [0], ring[0]
    for i in range(1, len(ring)):
        delta = (ring[i, 0] - last[0] + 180) % 360 - 180
        if abs(delta) * np.cos(np.deg2rad(ring[i, 1])) >= step_deg or abs(ring[i, 1] - last[1]) >= step_deg:
            keep.append(i)
            last = ring[i]
    if keep[-1] != len(ring) - 1:
        keep.append(len(ring) - 1)
    return ring[keep]


def coast_view(video_path: str, time_s: float, yaw_offset_deg: float = 0.0, takeoff_height_m: float = 0.0,
               horizontal_fov_deg: float = 0.0) -> dict:
    """Answer of ``/api/coast-view`` for one frame."""
    from .viewer import safe_media_path

    _, geod = _dependencies()
    context = load_context(safe_media_path(video_path))
    pose = _pose(context, time_s, geod, yaw_offset_deg, takeoff_height_m, horizontal_fov_deg)
    width, height = context.meta["width"], context.meta["height"]
    altitude = float(pose["altitude"])
    offsets = pose_offsets(context.video)
    camera = {"yaw_deg": float(pose["yaw"]), "log_yaw_deg": float(pose["log_yaw"]), "pitch_deg": float(pose["pitch"]),
              "log_pitch_deg": float(pose["log_pitch"]), "roll_deg": float(pose["roll"]),
              "log_roll_deg": float(pose["log_roll"]), "focal_px": float(pose["focal_px"]), "width": width,
              "height": height, "altitude_m": altitude, "latitude": float(pose["lat"]), "longitude": float(pose["lon"]),
              "pitch_offset_deg": offsets["pitch_offset_deg"], "roll_offset_deg": offsets["roll_offset_deg"]}
    result = {"time_s": round(float(pose["time_s"]), 3), "camera": camera, "source": coastdata.SOURCE,
              "horizon_dip_deg": round(coastdata.horizon_dip_deg(altitude), 4),
              "horizon_distance_km": round(coastdata.horizon_distance_m(altitude) / 1000, 1),
              "data": False, "in_view": False, "maybe_in_view": False, "lines": []}
    rings, path = land_directions(camera["latitude"], camera["longitude"], altitude, geod)
    if rings is None:
        result["note"] = ("Для этого района нет береговой линии. Скачайте её один раз командой "
                          "«python -m dronevision coastline» (нужен интернет, ~1 ГБ).")
        return result
    result.update(data=True, file=str(path))
    if not rings:
        result["note"] = f"В радиусе видимости ({view_limit_m(altitude) / 1000:.0f} км) суши нет."
        return result
    step = 0.4 * np.rad2deg(1 / camera["focal_px"])
    nearest, inside, nearby, span = np.inf, 0, 0, []
    for ring in rings:
        u, v, depth = to_pixels(ring, camera["yaw_deg"], camera["pitch_deg"], camera["roll_deg"], camera["focal_px"],
                                width, height)
        seen = (depth > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height) & (ring[:, 2] <= view_limit_m(altitude))
        # The logged heading can be 10–20° off (docs/projection.md): land just outside may be in the frame.
        loose = (depth > 0) & (u >= -0.35 * width) & (u < 1.35 * width) & (v >= -0.1 * height) & (v < 1.1 * height)
        inside += int(seen.sum())
        nearby += int(loose.sum())
        if seen.any():
            nearest = min(nearest, float(ring[seen, 2].min()))
            span += ring[seen, 0].tolist()
        if loose.any():
            result["lines"].append(np.round(_thin(ring, step), 4).tolist())
    result.update(in_view=inside >= 5, maybe_in_view=nearby >= 5, points_in_view=inside)
    if inside >= 5:
        centre = np.rad2deg(np.angle(np.exp(1j * np.deg2rad(span)).mean())) % 360
        result.update(nearest_km=round(nearest / 1000, 2), azimuth_deg=round(float(centre), 1))
    elif nearby >= 5:
        result["note"] = "По логу берег рядом с краем кадра: курс лога может быть неточным, проверьте наложением."
    else:
        result["lines"] = []
        result["note"] = "Берег есть в районе, но не в поле зрения камеры."
    return result


def _read_frame(cv2, context, time_s: float):
    cap = open_video(context.video)
    try:
        index = int(round(float(np.clip(time_s, 0, context.meta["duration"])) * context.meta["fps"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise RuntimeError(f"Не удалось прочитать кадр {index}")
    return frame


LAND_RISE_M = 40.0             # assumed height of land above its waterline: what makes it stand out


def _heading_by_land_mask(cv2, frame, rings, log_yaw: float, pitch: float, roll: float, focal: float,
                          altitude: float, search_deg: float, pitch_search_deg: float) -> dict | None:
    """Heading (and, without a horizon, pitch) shift that puts the mapped land where the picture shows it.

    The part of the frame below the sea horizon is resampled to viewing angles (azimuth across, depression
    down).  The land polygons are drawn in the same angles for a wider range of azimuths: their sea-level
    footprint, smeared upwards by ``LAND_RISE_M`` because land rises above its waterline.  Sliding the strip
    along the drawing is a turn of the camera; for every shift the drawing is correlated with the blueness
    (B − R) of the strip.  Land of any kind — rock, tundra, forest, sand, snow — is less blue than the sea
    beside it, while brightness can go either way, and the sky part of the frame is dominated by haze
    (tried on DJI_0531: brightness and the skyline gave false peaks, blueness below the horizon did not).
    Slow changes along the horizon (sun side, slicks) are removed from both by a running mean.
    """
    height, width = frame.shape[:2]
    dip = coastdata.horizon_dip_deg(altitude)
    step = np.rad2deg(2.0 / focal)                                   # two pixels of the frame
    half = np.rad2deg(np.arctan(width / 2 / focal)) - 0.3
    nearest = min(float(ring[:, 2].min()) for ring in rings)
    lower_edge = -pitch + np.rad2deg(np.arctan(height / 2 / focal)) - dip - 0.3
    deepest = float(min(np.rad2deg(np.arctan(altitude / nearest)) - dip + 0.6, lower_edge, 25.0))
    if deepest < 6 * step:
        return None
    if deepest / step > 360:                                         # a shore right below: coarser strip
        step = deepest / 360
    rows = np.arange(1.2 * step, deepest, step)
    columns = np.arange(-half, half, step)
    theta, delta = np.meshgrid(columns, rows)
    u, v, _ = to_pixels(np.column_stack([log_yaw + theta.ravel(), dip + delta.ravel()]), log_yaw, pitch, roll,
                        focal, width, height)
    blurred = cv2.GaussianBlur(frame, (0, 0), max(step / np.rad2deg(1 / focal) / 2, 0.8)).astype(np.float32)
    strip = cv2.remap(blurred, u.reshape(theta.shape).astype(np.float32), v.reshape(theta.shape).astype(np.float32),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    blueness = strip[..., 0] - strip[..., 2]
    pad_rows = int(round(pitch_search_deg / step)) if pitch_search_deg >= 0.5 else 0
    azimuths = np.arange(-half - search_deg, half + search_deg, step)
    mask = np.zeros((len(rows) + 2 * pad_rows, len(azimuths)), np.uint8)
    for ring in rings:
        relative = (ring[:, 0] - log_yaw + 180) % 360 - 180
        ahead = np.abs(relative) < 150
        if ahead.sum() < 3:
            continue
        points = np.column_stack([(relative[ahead] - azimuths[0]) / step,
                                  (ring[ahead, 1] - dip - rows[0]) / step + pad_rows])
        rise = int(np.clip(round(np.rad2deg(np.arctan(LAND_RISE_M / max(float(ring[ahead, 2].min()), 50.0))) / step), 1, 60))
        for lift in range(rise + 1):
            cv2.fillPoly(mask, [np.int32(np.round(points - [0, lift]))], 1)
    if not 0.002 < float(mask.mean()) < 0.95:
        return None
    best = None
    for window_deg in (5.0, 15.0):
        size = int(round(window_deg / step)) | 1
        drawing = mask.astype(np.float32)
        drawing -= cv2.blur(drawing, (size, 1), borderType=cv2.BORDER_REFLECT)
        template = blueness - cv2.blur(blueness, (size, 1), borderType=cv2.BORDER_REFLECT)
        if float(template.std()) < 1e-3 or float(drawing.std()) < 1e-4:
            continue
        # Land is less blue: the sought shift has the most negative correlation.
        score = -cv2.matchTemplate(np.ascontiguousarray(drawing), np.ascontiguousarray(template.astype(np.float32)),
                                   cv2.TM_CCOEFF_NORMED)
        middle = float(np.median(score))
        # Mostly-empty drawings give a score that is zero almost everywhere: the spread must not collapse.
        spread = max(1.4826 * float(np.median(np.abs(score - middle))), 0.5 * float(score.std()), 0.01)
        _, value, _, (j, i) = cv2.minMaxLoc(score)
        away = score.copy()
        reach = int(round(1.5 / step))
        away[:, max(j - reach, 0):j + reach + 1] = -2
        rival = float(away.max())
        candidate = {"peak": (value - middle) / spread, "rival": (rival - middle) / spread, "correlation": float(value),
                     "window_deg": window_deg, "i": i, "j": j, "score": score}
        if best is None or candidate["peak"] - candidate["rival"] > best["peak"] - best["rival"]:
            best = candidate
    if best is None:
        return None
    score, i, j = best.pop("score"), best.pop("i"), best.pop("j")
    fine = 0.0
    if 0 < j < score.shape[1] - 1:
        a, b, c = score[i, j - 1], score[i, j], score[i, j + 1]
        fine = float(0.5 * (a - c) / (a - 2 * b + c)) if a - 2 * b + c else 0.0
    # Column j of the drawing under column 0 of the strip: the strip's azimuth log_yaw + columns[0] is really
    # log_yaw + azimuths[j], so the camera heading is larger by azimuths[j] − columns[0].  Row i likewise:
    # the land is (i − pad) steps lower than assumed, i.e. the camera looks higher by that angle.
    best.update(yaw_shift_deg=float(azimuths[0] + (j + fine) * step - columns[0]),
                pitch_shift_deg=float((i - pad_rows) * step),
                land_share=round(float(mask[i:i + len(rows), j:j + len(columns)].mean()), 4),
                strip_deg=[round(float(2 * half), 1), round(float(deepest), 2)])
    return best


def align(video_path: str, time_s: float, yaw_offset_deg: float = 0.0, takeoff_height_m: float = 0.0,
          horizontal_fov_deg: float = 0.0, search_deg: float = 30.0) -> dict:
    """Heading, pitch and roll offsets of one frame from the horizon and the land in view.

    Returns the offsets to *use* (not increments): ``yaw_offset_deg`` for the «Поправка курса» field and
    ``pitch_offset_deg`` / ``roll_offset_deg`` for the per-video sidecar (``pose.save_pose_offsets``).
    """
    from .viewer import safe_media_path

    cv2, geod = _dependencies()
    context = load_context(safe_media_path(video_path))
    pose = _pose(context, time_s, geod, 0.0, takeoff_height_m, horizontal_fov_deg)
    width, height = context.meta["width"], context.meta["height"]
    focal, altitude = float(pose["focal_px"]), float(pose["altitude"])
    frame = _read_frame(cv2, context, time_s)
    result = {"time_s": round(float(pose["time_s"]), 3), "log_yaw_deg": float(pose["log_yaw"]),
              "log_pitch_deg": float(pose["log_pitch"]), "log_roll_deg": float(pose["log_roll"]),
              "yaw_offset_deg": float(yaw_offset_deg), "pitch_offset_deg": float(pose["pitch"] - pose["log_pitch"]),
              "roll_offset_deg": float(pose["roll"] - pose["log_roll"]), "horizon": None, "coast": None, "notes": []}

    # 1. Pitch and roll: the sea horizon, searched around the logged pitch (the stored offset is not trusted here).
    line = horizon.measure(cv2, frame, focal, altitude, float(pose["log_pitch"]))
    pitch, roll = float(pose["pitch"]), float(pose["roll"])
    if line is not None:
        pitch, roll = line["pitch_deg"], line["roll_deg"]
        result["horizon"] = line
        result.update(pitch_offset_deg=round(pitch - float(pose["log_pitch"]), 3),
                      roll_offset_deg=round(roll - float(pose["log_roll"]), 3))
    else:
        result["notes"].append("линия горизонта не найдена — наклон и крен оставлены как были")

    # 2. Heading: waterlines against image edges.
    rings, _ = land_directions(float(pose["lat"]), float(pose["lon"]), altitude, geod)
    if not rings:
        result["notes"].append("береговой линии для района нет" if rings is None else "суши в радиусе видимости нет")
        result["quality"] = "horizon" if line is not None else "failed"
        return _rounded(result)
    match = _heading_by_land_mask(cv2, frame, rings, float(pose["log_yaw"]), pitch, roll, focal, altitude, search_deg,
                                  0.08 if line is not None else 2.0)
    if match is None:
        result["notes"].append("берег дальше морского горизонта или не попадает в кадр — курс не подобран")
        result["quality"] = "horizon" if line is not None else "failed"
        return _rounded(result)
    # ``peak``: how many robust standard deviations the best heading stands above all the others;
    # ``rival``: the same for the best heading farther than 1.5° away.
    seen = match["land_share"] >= 0.005
    quality = ("good" if seen and match["peak"] >= 5.5 and match["peak"] >= 1.3 * match["rival"] and match["correlation"] >= 0.15
               else "weak" if seen and match["peak"] >= 4.5 and match["peak"] >= 1.1 * match["rival"] and match["correlation"] >= 0.1
               else "failed")
    result["coast"] = {key: (round(value, 3) if isinstance(value, float) else value) for key, value in match.items()}
    if quality != "failed":
        result["yaw_offset_deg"] = match["yaw_shift_deg"]
        if line is None:
            result["pitch_offset_deg"] = float(pitch + match["pitch_shift_deg"] - pose["log_pitch"])
    else:
        result["notes"].append("берег в кадре не распознан уверенно — курс не изменён, выровняйте наложение вручную")
    result["quality"] = quality
    return _rounded(result)


def _rounded(result: dict) -> dict:
    for key in ("yaw_offset_deg", "pitch_offset_deg", "roll_offset_deg"):
        result[key] = round(float(result[key]), 3)
    return result


def save_alignment(video_path: str, pitch_offset_deg: float, roll_offset_deg: float, **extra) -> dict:
    from .viewer import safe_media_path
    return save_pose_offsets(safe_media_path(video_path), pitch_offset_deg, roll_offset_deg, **extra)
