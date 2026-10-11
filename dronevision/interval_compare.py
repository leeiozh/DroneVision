"""«Сравнить 2 осреднённых интервала»: two temporal averages on one ENU canvas for flicker comparison.

The counterpart of ``projection.project_frame_pair`` for averages.  Every interval is sampled and averaged
exactly as by ``mosaic.average_interval`` (quiet ≤ 5 s window at ~6 Hz, sigma-clipped mean); both results
share bounds and resolution, so the viewer switches between them without moving and the contour / velocity
tools work as for two frames.  The time of an average is the mean time of its frames (the same convention
as in ``contour_tracking``).

Alignment has two levels.  Inside an interval the frames are aligned with each other as in a single average.
Between the intervals one common (yaw, pitch, roll) shift per interval is found from their middle frames and,
as for a frame pair, only when land is visible: over open water it would cancel the drift being measured.
"""
from __future__ import annotations


import numpy as np

from .cvio import open_video
from .geometry import camera_basis, finest_ground_resolution
from .mosaic import (MAX_CANVAS_PIXELS, MAX_CANVAS_SIDE, _canvas, _dependencies, _footprints, _read_frame,
                     horizon_margin_rows,
                     _stable_temporal_sample, mosaic_selected_frames)
from .pose import frame_poses, load_context


def _align_intervals(cv2, context, geod, video, samples: list[dict], bottom_row: int) -> dict:
    """Refine the attitudes of both samples in place; return the report of the interval-to-interval step."""
    from .projection import _visible_land_fraction
    from .refine import apply_corrections, frame_features, refine_attitudes

    width, height = context.meta["width"], context.meta["height"]
    middles = []
    cap = open_video(video)
    position = 10 ** 9
    try:
        for sample in samples:
            features = []
            for index in sample["indices"]:
                frame, position = _read_frame(cv2, cap, int(index), position)
                features.append(frame_features(cv2, frame, bottom_row))
            corrections, sample["refinement"] = refine_attitudes(cv2, features, sample["log_poses"], width, height)
            sample["poses"] = apply_corrections(sample["log_poses"], corrections)
            middles.append(features[len(features) // 2])
            del features
    finally:
        cap.release()
    pair = {key: np.array([sample["poses"][key][len(sample["indices"]) // 2] for sample in samples])
            for key in samples[0]["poses"]}
    land = _visible_land_fraction(context, pair, geod)
    if land is None or land < 0.005:
        note = ("в кадрах нет суши — выравнивание интервалов между собой отключено, иначе оно «съело» бы дрейф воды"
                if land is not None else
                "нет данных о суше для этого района — выравнивание интервалов между собой отключено, "
                "чтобы не «съесть» дрейф воды")
        return {"method": "image_feature_attitude_refinement", "frames": 2, "edges": 0, "skipped": True,
                "land_fraction": land, "note": note}
    shifts, report = refine_attitudes(cv2, middles, pair, width, height)
    for sample, shift in zip(samples, shifts):
        sample["poses"] = apply_corrections(sample["log_poses"], sample["poses"]["attitude_correction"] + shift)
    report["land_fraction"] = round(land, 4)
    return report


def compare_averaged_intervals(video_path: str, intervals, project_dir: str, takeoff_height_m: float = 0.0,
                               horizontal_fov_deg: float = 0.0, resolution_m_per_px: float = 0.0,
                               yaw_offset_deg: float = 0.0, max_distance_m: float = 0.0,
                               adjustments: dict | None = None, bottom_crop_percent: float = 0.0,
                               refine: bool = True, horizon_offset_percent: float = 0.0) -> dict:
    """Average two intervals onto one identical ENU canvas.

    ``intervals`` = [[start_s, end_s], [start_s, end_s]].  Returns {"variants": [earlier, later],
    "comparison": {...}}; each variant has the format of ``mosaic.average_interval`` plus ``time_s`` (mean
    time of the averaged frames) and ``interval_s`` (the window actually used).
    """
    from .viewer import safe_media_path

    if len(intervals) != 2:
        raise ValueError("Для сравнения нужно выбрать ровно два интервала")
    cv2, geod = _dependencies()
    video = safe_media_path(video_path)
    context = load_context(video)
    meta = context.meta
    width, height, fps = meta["width"], meta["height"], meta["fps"]
    bottom_row = int(round(height * (1 - np.clip(bottom_crop_percent, 0, 95) / 100)))
    samples = []
    for start_s, end_s in sorted([float(a), float(b)] for a, b in intervals):
        times, report = _stable_temporal_sample(context, geod, start_s, end_s, yaw_offset_deg,
                                                takeoff_height_m, horizontal_fov_deg)
        # The same frame selection as in mosaic_selected_frames, so the corrections stay frame for frame.
        indices = np.unique(np.round(np.clip(times, 0, meta["duration"]) * fps).astype(int))
        indices = indices[indices < max(int(meta["frames"]) or 10 ** 9, 1)]
        poses = frame_poses(context, indices / fps, geod, yaw_offset_deg, takeoff_height_m, horizontal_fov_deg)
        samples.append({"indices": indices, "report": report, "log_poses": poses, "poses": poses,
                        "refinement": None})
    if samples[1]["indices"][0] <= samples[0]["indices"][-1]:
        raise ValueError("Интервалы пересекаются: для сравнения выберите два отдельных интервала")
    alignment = _align_intervals(cv2, context, geod, video, samples, bottom_row) if refine else None

    # One canvas for both averages: the union of all footprints at the finest resolution of any frame.
    corners, native = [], []
    for sample in samples:
        poses = sample["poses"]
        crop_rows, sample_corners, _ = _footprints(poses, width, height, int(horizon_margin_rows(height, horizon_offset_percent)[0]), max_distance_m)
        corners.append(sample_corners)
        native += [finest_ground_resolution(
            width, height, poses["focal_px"][n], camera_basis(poses["yaw"][n], poses["pitch"][n], poses["roll"][n]),
            poses["altitude"][n], crop_rows[n], bottom_row) for n in range(len(crop_rows))]
    resolution_auto = not resolution_m_per_px or float(resolution_m_per_px) <= 0
    minimum, maximum, resolution, _, _ = _canvas(
        np.concatenate(corners), min(native) if resolution_auto else float(resolution_m_per_px),
        MAX_CANVAS_PIXELS, MAX_CANVAS_SIDE)
    # Half a pixel inside the grid lines: snapping in mosaic_selected_frames then returns the same bounds.
    half = resolution / 2
    region = {"east": [float(minimum[0] + half), float(maximum[0] - half)],
              "north": [float(minimum[1] + half), float(maximum[1] - half)], "exact": True}

    # Both files carry the whole span: start of the first interval — end of the second one (naming.py).
    ordered = sorted([float(a), float(b)] for a, b in intervals)
    variants = []
    for sample in samples:
        result = mosaic_selected_frames(
            str(video), sample["indices"] / fps, project_dir, takeoff_height_m, horizontal_fov_deg, resolution,
            yaw_offset_deg, max_distance_m, adjustments, bottom_crop_percent, output_kind="temporal_average",
            result_metadata={"temporal_average": sample["report"]}, refine=False,
            attitude_corrections=sample["poses"].get("attitude_correction"), region_m=region,
            name_span=(ordered[0][0], ordered[1][1]), name_kind=f"compare{len(variants) + 1}",
            horizon_offset_percent=horizon_offset_percent)
        result.update(time_s=round(float(np.mean(sample["indices"] / fps)), 3),
                      interval_s=sample["report"]["used_interval_s"], refinement=sample["refinement"],
                      resolution_auto=resolution_auto, native_resolution_m=min(native))
        variants.append(result)
    first, second = (sample["poses"] for sample in samples)
    mean_yaw = lambda poses: np.rad2deg(np.angle(np.exp(1j * np.deg2rad(poses["yaw"])).mean()))
    comparison = {
        "delta_time_s": round(variants[1]["time_s"] - variants[0]["time_s"], 3),
        "drone_displacement_m": round(float(np.linalg.norm(second["offset"].mean(axis=0)
                                                           - first["offset"].mean(axis=0))), 3),
        "yaw_delta_deg": round(float((mean_yaw(second) - mean_yaw(first) + 180) % 360 - 180), 3),
        "pitch_delta_deg": round(float(np.mean(second["pitch"]) - np.mean(first["pitch"])), 3),
        "refinement": alignment,
    }
    return {"variants": variants, "time_s": variants[0]["time_s"], "kind": "average_pair",
            "comparison": comparison}
