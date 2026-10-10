from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from .adjust import apply_adjustments
from .config import ProjectConfig
from .cvio import imwrite, open_video
from .geometry import (camera_basis, finest_ground_resolution, ground_corners, ground_points, horizon_crop_row,
                       pixel_rays)
from .grid import overlay_metric_grid
from .naming import metadata as naming_metadata, output_stem
from .pose import frame_poses, load_context

MAX_GROUND_DISTANCE_M = 10_000.0
# Interactive rasters: finer default resolution (0.25 m/px) needs a larger canvas; ~48 MP stays manageable
# for PNG encoding and the browser.
MAX_CANVAS_PIXELS = 48_000_000
MAX_CANVAS_SIDE = 10_000


def _dependencies():
    try:
        import cv2
        from pyproj import Geod
    except ImportError as exc:
        raise RuntimeError("Для обработки установите зависимости: pip install -r requirements-app.txt") from exc
    return cv2, Geod(ellps="WGS84")


def _read_frame(cv2, cap, frame_index: int, position: int) -> tuple[np.ndarray, int]:
    """Decode sequentially for short skips: seeking in 4K H.264 restarts from a keyframe."""
    if not 0 <= frame_index - position <= 90:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        position = frame_index
    while position < frame_index:
        if not cap.grab():
            raise RuntimeError(f"Не удалось прочитать кадр {position}")
        position += 1
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Не удалось прочитать кадр {frame_index}")
    return frame, frame_index + 1


_THREADS = max(1, min(8, (os.cpu_count() or 2)))
_POOL = ThreadPoolExecutor(_THREADS)
LUMA = np.array([0.114, 0.587, 0.299], dtype=np.float32)       # BGR weights of Gray = 0.299R + 0.587G + 0.114B


def _footprints(poses: dict, width: int, height: int, margin_rows: int, max_distance_m: float):
    """Crop row, ENU corners and per-row blending weight of every frame.

    The weight is the inverse sea-surface area of one image pixel along the centre column,
    ``f² h / d³`` (d = slant range): finer ground detail wins wherever frames overlap, whatever the
    altitude, zoom or tilt of each frame.  Weights are normalised by the largest over all frames.
    (Previously ``(d_min / d)²`` was normalised per frame, so a frame starting far away got up to
    tens of times too much weight in its distant part.)
    """
    crop_rows, corners, row_weight = [], [], []
    for i in range(len(poses["time_s"])):
        basis = camera_basis(poses["yaw"][i], poses["pitch"][i], poses["roll"][i])
        focal, altitude = poses["focal_px"][i], poses["altitude"][i]
        crop = max(horizon_crop_row(width, height, focal, basis, altitude, np.inf, margin_rows),
                   horizon_crop_row(width, height, focal, basis, altitude, max_distance_m))
        crop_rows.append(crop)
        corners.append(poses["offset"][i] + ground_corners(width, height, focal, basis, altitude, crop))
        rows = np.arange(crop, height)
        centre = ground_points(pixel_rays(np.full(len(rows), width / 2), rows, width, height, focal, basis), altitude)
        slant = np.sqrt((centre ** 2).sum(axis=1) + altitude ** 2)
        row_weight.append(focal ** 2 * altitude / slant ** 3)
    peak = max(float(w.max()) for w in row_weight)
    return crop_rows, np.asarray(corners), [w / peak for w in row_weight]


def _canvas(corners: np.ndarray, resolution: float, max_pixels: int, max_side: int | None = None,
            region: dict | None = None):
    """North-up canvas on multiples of the resolution; ``region`` = {"east": [a, b], "north": [c, d]} clips it."""
    lower = corners.reshape(-1, 2).min(axis=0)
    upper = corners.reshape(-1, 2).max(axis=0)
    if region and region.get("exact"):
        # The caller fixes the canvas (e.g. both frames of a comparison must share it).
        lower = np.array([min(region["east"]), min(region["north"])], dtype=np.float64)
        upper = np.array([max(region["east"]), max(region["north"])], dtype=np.float64)
    elif region:
        lower = np.maximum(lower, [min(region["east"]), min(region["north"])])
        upper = np.minimum(upper, [max(region["east"]), max(region["north"])])
        if np.any(upper <= lower):
            raise ValueError("Выбранная область не пересекается с кадрами")
    span = upper - lower
    size = np.maximum(np.ceil(span / resolution).astype(int), 2)
    if max_side and (size.prod() > max_pixels or size.max() > max_side):
        resolution *= max(np.sqrt(size.prod() / max_pixels), size.max() / max_side)
        resolution = float(np.ceil(resolution * 1000) / 1000)   # exact in the rounded JSON value
    elif size.prod() > max_pixels:
        raise ValueError(f"Мозаика {size[0]}×{size[1]} слишком велика; увеличьте метров/пиксель")
    minimum = np.floor(lower / resolution) * resolution
    maximum = np.ceil(upper / resolution) * resolution
    size = np.maximum(np.round((maximum - minimum) / resolution).astype(int), 2)
    return minimum, maximum, float(resolution), int(size[0]), int(size[1])


def _edge_taper(length: int, ramp: int) -> np.ndarray:
    """1 inside, falling smoothly to ~0 over ``ramp`` samples at both ends (seam-free blending)."""
    position = np.minimum(np.arange(length) + 0.5, length - np.arange(length) - 0.5)
    t = np.clip(position / max(ramp, 1), 0, 1)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def _accumulate(cv2, cropped, corners, row_weight, minimum, maximum, resolution, accumulator, weights,
                valid_rows: int | None = None, interpolation: int | None = None, luma_sq=None, keep=None):
    """Warp one cropped frame into the north-up canvas with its blending weight.

    ``luma_sq`` accumulates w·Gray² (first pass of the robust average); ``keep(y0, y1, x0, x1, warped)``
    returns a mask of samples to use (second pass: outliers such as sun glints are dropped).
    """
    height, width = cropped.shape[:2]
    canvas_h, canvas_w = weights.shape
    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    destination = np.column_stack([(corners[:, 0] - minimum[0]) / resolution, (maximum[1] - corners[:, 1]) / resolution])
    to_canvas = cv2.getPerspectiveTransform(source, destination.astype(np.float32))
    x0, y0 = np.maximum(np.floor(destination.min(axis=0)).astype(int), 0)
    x1 = min(int(np.ceil(destination[:, 0].max())) + 1, canvas_w)
    y1 = min(int(np.ceil(destination[:, 1].max())) + 1, canvas_h)
    if x1 <= x0 or y1 <= y0:
        return to_canvas
    rows = np.asarray(row_weight, dtype=np.float32).copy()
    usable = height if valid_rows is None else max(min(int(valid_rows), height), 0)
    rows[usable:] = 0
    rows[:usable] *= _edge_taper(usable, max(8, usable // 40))
    weight = rows[:, None] * _edge_taper(width, max(8, width // 50))[None, :]
    flags = cv2.INTER_LINEAR if interpolation is None else interpolation
    luma = LUMA.reshape(1, 3)

    def strip(top: int, bottom: int) -> None:
        # Canvas rows top..bottom: every strip warps and accumulates independently (numpy/OpenCV release
        # the GIL on large arrays, so strips run in parallel threads).
        shift = np.array([[1, 0, -x0], [0, 1, -top], [0, 0, 1]], dtype=np.float64) @ to_canvas
        size = (x1 - x0, bottom - top)
        warped_weight = cv2.warpPerspective(weight, shift, size, flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if not warped_weight.any():
            return
        warped = cv2.warpPerspective(cropped, shift, size, flags=flags)
        if keep is not None:
            warped_weight *= keep(top, bottom, x0, x1, warped)
        region = accumulator[top:bottom, x0:x1]
        cv2.accumulateProduct(warped.astype(np.float32), cv2.merge([warped_weight] * 3), region)
        cv2.accumulate(warped_weight, weights[top:bottom, x0:x1])
        if luma_sq is not None:
            grey = cv2.transform(warped.astype(np.float32), luma)
            cv2.accumulateProduct(grey * grey, warped_weight, luma_sq[top:bottom, x0:x1])

    step = max(64, int(np.ceil((y1 - y0) / _THREADS)))
    list(_POOL.map(lambda top: strip(top, min(top + step, y1)), range(y0, y1, step)))
    return to_canvas


def _finish(accumulator, weights):
    return np.where(weights[..., None] > 0, accumulator / np.maximum(weights[..., None], 1e-6), 0).clip(0, 255).round().astype(np.uint8)


def run(config: ProjectConfig, progress=lambda value, message: None) -> Path:
    """Build a north-up mosaic while keeping only one decoded/projected frame in memory."""
    cv2, geod = _dependencies()
    config.validate()
    config_path = config.save()
    context = load_context(config.video_path, config.log_path, config.utc_offset_hours, config.clock_offset_seconds)
    meta = context.meta
    ranges = config.time_ranges()
    if max(end for _, end in ranges) > meta["duration"]:
        raise ValueError("Конец выбранного фрагмента находится за пределами видео")
    indices = np.unique(np.concatenate([np.arange(round(start * meta["fps"]), round(end * meta["fps"]),
                                                   config.frame_interval, dtype=int) for start, end in ranges]))
    if not len(indices):
        raise ValueError("В выбранные фрагменты не попало ни одного кадра")
    poses = frame_poses(context, indices / meta["fps"], geod, config.yaw_offset_deg, config.takeoff_height_m,
                        config.horizontal_fov_deg)
    times = poses["epoch"]
    width, height = meta["width"], meta["height"]
    bottom_row = int(round(height * (1 - np.clip(config.bottom_crop_percent, 0, 95) / 100)))

    progress(0.05, "Расчёт границ мозаики")
    margin = int(round(height * max(config.horizon_margin, 0.0)))
    crop_rows, global_corners, row_weight = _footprints(poses, width, height, margin, MAX_GROUND_DISTANCE_M)
    minimum, maximum, resolution, canvas_w, canvas_h = _canvas(global_corners, config.resolution_m_per_px, 300_000_000)
    accumulator = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weights = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    run_dir = Path(config.project_dir).resolve() / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    tracker = None
    if config.run_masking or config.run_tracking or config.run_velocities:
        from .ice import Tracker
        tracker = Tracker(config.max_track_distance_m)

    cap = open_video(config.video_path)
    position = 0
    try:
        for n, frame_index in enumerate(indices):
            frame, position = _read_frame(cv2, cap, int(frame_index), position)
            cropped = frame[crop_rows[n]:, :]
            valid_rows = bottom_row - crop_rows[n]
            if valid_rows <= 1:
                raise ValueError("Нижняя граница обрезки находится выше горизонта")
            to_canvas = _accumulate(cv2, cropped, global_corners[n], row_weight[n], minimum, maximum, resolution,
                                    accumulator, weights, valid_rows)
            if tracker is not None:
                from .ice import detect
                ice_mask, objects = detect(cv2, cropped[:valid_rows], config.ice_s_max, config.ice_v_min,
                                           config.min_ice_area_px)
                if n == 0:
                    imwrite(run_dir / "ice_mask_preview.png", ice_mask)
                world_objects = []
                if objects:
                    points = np.float32([[x, y] for x, y, _ in objects]).reshape(-1, 1, 2)
                    mapped = cv2.perspectiveTransform(points, to_canvas).reshape(-1, 2)
                    world_objects = [(float(p[0] * resolution + minimum[0]), float(maximum[1] - p[1] * resolution), area)
                                     for p, (_, _, area) in zip(mapped, objects)]
                tracker.update(n, float(times[n]), world_objects)
            progress(0.1 + 0.8 * (n + 1) / len(indices), f"Кадр {n + 1} из {len(indices)}")
    finally:
        cap.release()
    output = run_dir / config.output_name
    mosaic, grid_step = overlay_metric_grid(cv2, _finish(accumulator, weights), minimum, maximum, resolution)
    if not imwrite(output, mosaic):
        raise RuntimeError("Не удалось сохранить мозаику")
    artifacts = {"mosaic": str(output)}
    if tracker is not None:
        progress(0.94, "Сохранение объектов, треков и скоростей")
        tracks_path, velocity_path, velocities = tracker.save(run_dir)
        artifacts.update(mask_preview=str(run_dir / "ice_mask_preview.png"), tracks=str(tracks_path), velocities=str(velocity_path))
    lat0, lon0 = context.takeoff
    manifest = {"project_config": str(config_path), "output": str(output), "artifacts": artifacts, "video": meta,
                "frames_used": len(indices), "canvas": [canvas_w, canvas_h],
                "segments": ranges,
                "reference": {"lat": float(lat0), "lon": float(lon0), "meaning": "точка взлёта, ENU-метры"},
                "bounds_m": {"east": [float(minimum[0]), float(maximum[0])], "north": [float(minimum[1]), float(maximum[1])]},
                "grid_step_m": grid_step,
                "bottom_crop_percent": config.bottom_crop_percent,
                "orientation": "north_up",
                "camera": {"profile": context.profile, "yaw_offset_deg": config.yaw_offset_deg,
                           "horizontal_fov_deg": [float(poses["horizontal_fov_deg"].min()), float(poses["horizontal_fov_deg"].max())],
                           "log_path": context.log_path, "utc_offset_hours": context.utc_offset_hours,
                           "clock_offset_seconds": context.clock_offset_seconds,
                           "time_base": "SRT" if context.srt is not None else "creation_time"},
                "estimated_gsd_at_mean_altitude": float(np.mean(poses["altitude"] / poses["focal_px"]))}
    (run_dir / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    progress(1.0, "Мозаика готова")
    return output


def _angle_delta(values: np.ndarray) -> np.ndarray:
    """Shortest signed difference between consecutive angles in degrees."""
    return (np.diff(values) + 180.0) % 360.0 - 180.0


def _stable_temporal_sample(context, geod, start_s: float, end_s: float, yaw_offset_deg: float,
                            takeoff_height_m: float, horizontal_fov_deg: float) -> tuple[np.ndarray, dict]:
    """Choose a quiet 2–5 s sequence at about 6 Hz and reject abrupt pose changes."""
    start_s = max(0.0, float(start_s)); end_s = min(float(context.meta["duration"]), float(end_s))
    if end_s - start_s < 1.0:
        raise ValueError("Для осреднения выберите интервал не короче 1 секунды; лучше 2–5 секунд")
    sample_rate = min(6.0, float(context.meta["fps"]))
    step = 1.0 / sample_rate
    times = np.arange(start_s, end_s + step * 0.25, step, dtype=np.float64)
    times = times[times <= end_s + 1e-6]
    if times[-1] < end_s - step * 0.5:
        times = np.append(times, end_s)
    requested_candidates = len(times)
    poses = frame_poses(context, times, geod, yaw_offset_deg, takeoff_height_m, horizontal_fov_deg)

    dt = np.maximum(np.diff(times), 1e-6)
    yaw_rate = np.abs(_angle_delta(poses["yaw"])) / dt
    pitch_rate = np.abs(np.diff(poses["pitch"])) / dt
    roll_rate = np.abs(np.diff(poses["roll"])) / dt
    speed = np.linalg.norm(np.diff(poses["offset"], axis=0), axis=1) / dt
    fov_rate = np.abs(np.diff(poses["horizontal_fov_deg"])) / dt
    # The limits detect manoeuvres and telemetry jumps, but allow ordinary hover drift.
    edge_ok = ((yaw_rate <= 2.0) & (pitch_rate <= 1.5) & (roll_rate <= 1.5) &
               (speed <= 4.0) & (fov_rate <= 1.0))
    score = (yaw_rate / 2.0 + pitch_rate / 1.5 + roll_rate / 1.5 + speed / 4.0 + fov_rate)

    # Longer selections are intentionally reduced to the quietest window of at most five seconds.
    max_frames = max(2, int(round(5.0 * sample_rate)) + 1)
    if len(times) > max_frames:
        best_start, best_score = 0, np.inf
        for i in range(len(times) - max_frames + 1):
            window = score[i:i + max_frames - 1]
            value = float(np.mean(np.minimum(window, 20.0)) + 4.0 * np.mean(~edge_ok[i:i + max_frames - 1]))
            if value < best_score:
                best_start, best_score = i, value
        sl = slice(best_start, best_start + max_frames)
        times = times[sl]; poses = {key: value[sl] for key, value in poses.items()}
        dt = np.maximum(np.diff(times), 1e-6)
        yaw_rate = np.abs(_angle_delta(poses["yaw"])) / dt
        pitch_rate = np.abs(np.diff(poses["pitch"])) / dt
        roll_rate = np.abs(np.diff(poses["roll"])) / dt
        speed = np.linalg.norm(np.diff(poses["offset"], axis=0), axis=1) / dt
        fov_rate = np.abs(np.diff(poses["horizontal_fov_deg"])) / dt
        edge_ok = ((yaw_rate <= 2.0) & (pitch_rate <= 1.5) & (roll_rate <= 1.5) &
                   (speed <= 4.0) & (fov_rate <= 1.0))

    # Use the longest uninterrupted quiet run. A sharp edge splits the sequence.
    runs, run_start = [], 0
    for i, ok in enumerate(edge_ok):
        if not ok:
            runs.append((run_start, i + 1)); run_start = i + 1
    runs.append((run_start, len(times)))
    first, last = max(runs, key=lambda item: item[1] - item[0])
    selected = times[first:last]
    minimum_frames = max(4, int(np.ceil(sample_rate)))
    if len(selected) < minimum_frames:
        raise ValueError("На интервале нет хотя бы одной спокойной секунды: выберите участок без поворота и резкого движения дрона")
    selected_poses = {key: value[first:last] for key, value in poses.items()}
    position = selected_poses["offset"]
    yaw_span = float(np.ptp(np.unwrap(np.deg2rad(selected_poses["yaw"]))) * 180.0 / np.pi)
    report = {
        "requested_interval_s": [round(start_s, 3), round(end_s, 3)],
        "used_interval_s": [round(float(selected[0]), 3), round(float(selected[-1]), 3)],
        "sample_rate_hz": round(sample_rate, 2),
        "candidate_frames": int(requested_candidates), "used_frames": int(len(selected)),
        "rejected_frames": int(requested_candidates - len(selected)),
        "yaw_span_deg": round(yaw_span, 3),
        "pitch_span_deg": round(float(np.ptp(selected_poses["pitch"])), 3),
        "roll_span_deg": round(float(np.ptp(selected_poses["roll"])), 3),
        "position_span_m": round(float(np.max(np.linalg.norm(position - position[0], axis=1))), 3),
        "method": "georeferenced_weighted_temporal_mean",
    }
    return selected, report


def mosaic_selected_frames(video_path: str, times_s, project_dir: str, takeoff_height_m: float = 0.0,
                           horizontal_fov_deg: float = 0.0, resolution_m_per_px: float = 1.0,
                           yaw_offset_deg: float = 0.0, max_distance_m: float = 3000.0,
                           adjustments: dict | None = None, bottom_crop_percent: float = 0.0,
                           output_kind: str = "mosaic", result_metadata: dict | None = None,
                           refine: bool = True, attitude_corrections=None, region_m: dict | None = None,
                           interpolation: str = "linear", robust: bool | None = None,
                           name_span: tuple[float, float] | None = None, name_kind: str | None = None) -> dict:
    """North-up mosaic of individual frames for checking how neighbouring frames overlap.

    ``name_span`` / ``name_kind`` override the times and the type in the file name (``naming.py``): by default
    the first and the last frame and ``mosaic`` / ``average`` / ``enhanced``.

    The response mirrors ``project_selected_frame`` (bounds, resolution, coastline) and adds frame footprints.
    Accuracy falls with the square of ground distance, hence the default 3 km limit.
    ``adjustments`` affect the PNG and GeoTIFF only, as in ``project_selected_frame``.
    ``refine`` aligns the frames with each other by image features (see ``refine.py``): the logged
    attitude is quantised to 0.1° and drifts, which moves distant islands by tens of metres.
    ``attitude_corrections`` (N×3, degrees) replace the refinement, e.g. when a stored result is re-rendered;
    ``region_m`` clips the canvas; ``interpolation`` = "linear" or "lanczos" (sharper, for re-rendering at
    native resolution); ``robust`` (default: on for temporal averages) makes a second pass that drops samples
    farther than 2.5σ from the per-pixel mean in grey level, i.e. sun glints and breaking crests.
    ``resolution_m_per_px`` <= 0 means automatic: the finest sea-surface pixel of the selected frames,
    coarsened only when the canvas limit requires it.
    """
    from .georaster import save_georaster
    from .projection import _coastline_bbox, _coastline_to_enu, _load_local_coastline, land_layer
    from .viewer import safe_media_path

    cv2, geod = _dependencies()
    video = safe_media_path(video_path)
    context = load_context(video)
    meta = context.meta
    frame_indices = np.unique(np.round(np.clip(np.asarray(times_s, dtype=np.float64), 0, meta["duration"]) * meta["fps"]).astype(int))
    frame_indices = frame_indices[frame_indices < max(int(meta["frames"]) or 10 ** 9, 1)]
    if len(frame_indices) < 1:
        raise ValueError("Не выбрано ни одного кадра")
    if len(frame_indices) > 60:
        raise ValueError("Слишком много кадров для интерактивной мозаики (больше 60)")
    poses = frame_poses(context, frame_indices / meta["fps"], geod, yaw_offset_deg, takeoff_height_m, horizontal_fov_deg)
    width, height = meta["width"], meta["height"]
    bottom_row = int(round(height * (1 - np.clip(bottom_crop_percent, 0, 95) / 100)))
    margin = int(round(height * 0.02))
    robust = output_kind == "temporal_average" if robust is None else bool(robust)
    flags = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}[interpolation]
    # Decoded frames are kept for the later passes when they fit in ~1.6 GB (≤ 64 4K frames);
    # otherwise every pass decodes again.
    cache = [] if len(frame_indices) * width * height * 3 <= 1_600_000_000 else None
    cap = open_video(video)
    position = 10 ** 9

    def frames():
        nonlocal position
        if cache is not None and len(cache) == len(frame_indices):
            yield from enumerate(cache)
            return
        for n, frame_index in enumerate(frame_indices):
            frame, position = _read_frame(cv2, cap, int(frame_index), position)
            if cache is not None:
                cache.append(frame)
            yield n, frame
        position = 10 ** 9   # next pass seeks to the first frame again

    refinement = None
    try:
        if attitude_corrections is not None:
            from .refine import apply_corrections
            poses = apply_corrections(poses, np.asarray(attitude_corrections, dtype=np.float64).reshape(-1, 3))
            refinement = {"method": "stored_attitude_corrections"}
        elif refine and len(frame_indices) > 1:
            from .refine import apply_corrections, frame_features, refine_attitudes
            features = [frame_features(cv2, frame, bottom_row) for _, frame in frames()]
            corrections, refinement = refine_attitudes(cv2, features, poses, width, height)
            poses = apply_corrections(poses, corrections)
            del features
        crop_rows, corners, row_weight = _footprints(poses, width, height, margin, max_distance_m)
        native_resolution = min(finest_ground_resolution(
            width, height, poses["focal_px"][n], camera_basis(poses["yaw"][n], poses["pitch"][n], poses["roll"][n]),
            poses["altitude"][n], crop_rows[n], bottom_row) for n in range(len(frame_indices)))
        resolution_auto = not resolution_m_per_px or float(resolution_m_per_px) <= 0
        minimum, maximum, resolution, canvas_w, canvas_h = _canvas(
            corners, native_resolution if resolution_auto else float(resolution_m_per_px), MAX_CANVAS_PIXELS,
            MAX_CANVAS_SIDE, region_m)
        accumulator = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        weights = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        luma_sq = np.zeros((canvas_h, canvas_w), dtype=np.float32) if robust and len(frame_indices) >= 4 else None
        for n, frame in frames():
            valid_rows = bottom_row - crop_rows[n]
            if valid_rows <= 1:
                raise ValueError("Нижняя граница обрезки находится выше горизонта")
            _accumulate(cv2, frame[crop_rows[n]:, :], corners[n], row_weight[n], minimum, maximum, resolution,
                        accumulator, weights, valid_rows, flags, luma_sq)
        robust_report = None
        if luma_sq is not None:
            # Second pass: per-pixel weighted mean and σ of Gray from the first pass, samples beyond
            # 2.5σ (+2 grey levels for quantisation) get zero weight.
            with np.errstate(invalid="ignore", divide="ignore"):
                total = np.maximum(weights, 1e-6)
                mean = (accumulator @ LUMA) / total
                sigma = np.sqrt(np.maximum(luma_sq / total - mean ** 2, 0))
            limit = 2.5 * sigma + 2.0
            first_weight = weights.copy()
            del luma_sq

            def keep(y0, y1, x0, x1, warped):
                grey = cv2.transform(warped.astype(np.float32), LUMA.reshape(1, 3))
                deviation = cv2.absdiff(grey, mean[y0:y1, x0:x1])
                return (deviation <= limit[y0:y1, x0:x1]).astype(np.float32)

            accumulator[:] = 0
            weights[:] = 0
            for n, frame in frames():
                _accumulate(cv2, frame[crop_rows[n]:, :], corners[n], row_weight[n], minimum, maximum, resolution,
                            accumulator, weights, bottom_row - crop_rows[n], flags, keep=keep)
            covered = first_weight > 0
            if not covered.any() and not (region_m or {}).get("exact"):
                raise ValueError("В выбранной области нет данных кадров")
            # At most 1/2.5² = 16 % of a pixel's weight lies beyond 2.5σ (Chebyshev), so with ≥ 4 frames
            # no covered pixel loses all its samples.
            robust_report = {"method": "sigma_clipped_mean", "k_sigma": 2.5,
                             "rejected_weight_percent": round(float(100 * (1 - weights[covered].sum()
                                                                            / max(first_weight[covered].sum(), 1e-6))), 2)}
            del first_weight, mean, sigma, limit
    finally:
        cap.release()
        cache = None
    output_root = Path(project_dir).expanduser().resolve() if project_dir else video.parent / "DroneVision-project"
    output_dir = output_root / "mosaics"
    output_dir.mkdir(parents=True, exist_ok=True)
    label = name_kind or {"temporal_average": "average", "enhanced": "enhanced"}.get(output_kind, "mosaic")
    first_s, last_s = name_span or (frame_indices[0] / meta["fps"], frame_indices[-1] / meta["fps"])
    output = output_dir / f"{output_stem(output_dir, context, first_s, last_s, label, ('.png', '.geo.json'))}.png"
    if not (weights > 0).any() and not (region_m or {}).get("exact"):
        raise ValueError("В выбранной области нет данных кадров")
    raw_mosaic = _finish(accumulator, weights)
    mosaic, grid_step = overlay_metric_grid(cv2, apply_adjustments(raw_mosaic, adjustments), minimum, maximum,
                                            resolution)
    if not imwrite(output, mosaic):
        raise RuntimeError("Не удалось сохранить мозаику")
    lat0, lon0 = context.takeoff
    georaster = save_georaster(output, raw_mosaic, weights > 0, minimum, maximum, resolution, (lat0, lon0),
                               output_kind, adjustments, video=str(video),
                               times_s=[round(float(t), 3) for t in poses["time_s"]],
                               frame_indices=[int(i) for i in frame_indices], yaw_offset_deg=yaw_offset_deg,
                               takeoff_height_m=takeoff_height_m, bottom_crop_percent=bottom_crop_percent,
                               horizontal_fov_deg=horizontal_fov_deg, max_distance_m=max_distance_m,
                               attitude_corrections_deg=np.round(poses.get("attitude_correction",
                                                                            np.zeros((len(frame_indices), 3))), 4).tolist(),
                               interpolation=interpolation, robust=robust_report,
                               name_span_s=[round(float(first_s), 3), round(float(last_s), 3)],
                               **naming_metadata(context))
    coastline_layers = []
    try:
        lines, source = _load_local_coastline(_coastline_bbox(lat0, lon0, minimum, maximum))
        coastline_layers.append({"source": source, "color": "#ff5b45", "lines": _coastline_to_enu(lines, geod, lat0, lon0)})
    except Exception as exc:
        coastline_layers.append({"source": "Локальная береговая линия", "color": "#ff5b45", "lines": [], "error": str(exc)})
    span = maximum - minimum
    frames = [{"time_s": round(float(poses["time_s"][i]), 3), "frame_index": int(frame_indices[i]),
               "yaw_deg": round(float(poses["yaw"][i]), 2), "log_yaw_deg": round(float(poses["log_yaw"][i]), 2),
               "pitch_deg": round(float(poses["pitch"][i]), 2), "roll_deg": round(float(poses["roll"][i]), 3),
               "zoom": round(float(poses["zoom"][i]), 2),
               "attitude_correction_deg": np.round(poses["attitude_correction"][i], 3).tolist()
               if "attitude_correction" in poses else [0.0, 0.0, 0.0],
               "altitude_m": round(float(poses["altitude"][i]), 1), "camera_m": np.round(poses["offset"][i], 2).tolist(),
               "footprint": np.round(corners[i], 1).tolist()} for i in range(len(frame_indices))]
    result = {"path": str(output), "size_px": [canvas_w, canvas_h],
            "bounds_m": {"east": [float(minimum[0]), float(maximum[0])], "north": [float(minimum[1]), float(maximum[1])]},
            "resolution_m_per_px": round(resolution, 4), "grid_step_m": grid_step,
            "resolution_auto": resolution_auto, "native_resolution_m": native_resolution,
            "max_distance_m": max_distance_m, "yaw_offset_deg": yaw_offset_deg,
            "bottom_crop_percent": float(bottom_crop_percent),
            "coastline_layers": coastline_layers,
            "coastline": {"source": " + ".join(layer["source"] for layer in coastline_layers),
                          "segments": sum(len(layer["lines"]) for layer in coastline_layers)},
            "land": land_layer(geod, lat0, lon0, minimum, maximum), **georaster,
            "footprints": [f["footprint"] for f in frames], "frames": frames,
            "takeoff": {"latitude": lat0, "longitude": lon0}, "kind": output_kind,
            "refinement": refinement, "robust_average": robust_report, "interpolation": interpolation}
    if result_metadata:
        result.update(result_metadata)
    return result


def average_interval(video_path: str, start_s: float, end_s: float, project_dir: str,
                     takeoff_height_m: float = 0.0, horizontal_fov_deg: float = 0.0,
                     resolution_m_per_px: float = 1.0, yaw_offset_deg: float = 0.0,
                     max_distance_m: float = 3000.0, adjustments: dict | None = None,
                     bottom_crop_percent: float = 0.0, refine: bool = True) -> dict:
    """Georeference and temporally average a stable sample from the selected interval."""
    _, geod = _dependencies()
    video = Path(video_path).expanduser().resolve()
    context = load_context(video)
    times, report = _stable_temporal_sample(context, geod, start_s, end_s, yaw_offset_deg,
                                            takeoff_height_m, horizontal_fov_deg)
    return mosaic_selected_frames(
        str(video), times, project_dir, takeoff_height_m, horizontal_fov_deg, resolution_m_per_px,
        yaw_offset_deg, max_distance_m, adjustments, bottom_crop_percent,
        output_kind="temporal_average", result_metadata={"temporal_average": report}, refine=refine,
        name_span=(float(start_s), float(end_s)),
    )
