from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import urllib.request
import struct
import zipfile

import numpy as np

from .adjust import apply_adjustments
from .geometry import camera_basis, finest_ground_resolution, ground_corners, horizon_crop_row
from .georaster import save_georaster
from .grid import nice_grid_step, overlay_metric_grid
from .naming import metadata as naming_metadata, output_stem
from .pose import enu_offsets, frame_poses, load_context
from .viewer import safe_media_path
from .cvio import imread, imwrite, open_video
from .paths import coastline_path

MAX_GROUND_DISTANCE_M = 10_000.0
MAX_CANVAS_PIXELS = 48_000_000     # same limits as mosaic.py
MAX_CANVAS_SIDE = 10_000
LOCAL_COASTLINE = coastline_path()     # see paths.py: settings, DroneData/maps, then maps/ of the application


def _dependencies():
    try:
        import cv2
        from pyproj import Geod
    except ImportError as exc:
        raise RuntimeError("Для проекции нужны opencv-python и pyproj") from exc
    return cv2, Geod(ellps="WGS84")


def _nice_grid_step(span_m: float) -> int:
    return nice_grid_step(span_m)


def _coastline_bbox(lat0: float, lon0: float, minimum: np.ndarray, maximum: np.ndarray) -> tuple[float, float, float, float]:
    # Deliberately wider than the projected raster: the UI can pan beyond the image.
    margin = max(float((maximum - minimum).max()) * 0.35, 1500.0)
    south = lat0 + (minimum[1] - margin) / 111_000
    north = lat0 + (maximum[1] + margin) / 111_000
    lon_scale = 111_000 * np.cos(np.deg2rad(lat0))
    west = lon0 + (minimum[0] - margin) / lon_scale
    east = lon0 + (maximum[0] + margin) / lon_scale
    return west, south, east, north


def _read_shapefile(data: bytes, bbox: tuple[float, float, float, float],
                    clamp: bool = False) -> list[list[list[float]]]:
    """Parts of shapes intersecting ``bbox`` as [[lat, lon], ...].

    By default points farther than 0.05° outside the box are dropped (outlines); with ``clamp`` they are
    moved onto the widened box instead, which keeps polygon rings closed for filling.
    """
    west, south, east, north = bbox
    lines = []
    offset = 100
    while offset + 8 <= len(data):
        _, words = struct.unpack_from(">2i", data, offset)
        offset += 8
        content_end = offset + words * 2
        if content_end > len(data):
            break
        shape_type = struct.unpack_from("<i", data, offset)[0]
        if shape_type in (3, 5):
            xmin, ymin, xmax, ymax = struct.unpack_from("<4d", data, offset + 4)
            if xmax >= west and xmin <= east and ymax >= south and ymin <= north:
                num_parts, num_points = struct.unpack_from("<2i", data, offset + 36)
                parts = list(struct.unpack_from(f"<{num_parts}i", data, offset + 44)) + [num_points]
                points_offset = offset + 44 + 4 * num_parts
                points = [struct.unpack_from("<2d", data, points_offset + 16 * i) for i in range(num_points)]
                for start, stop in zip(parts[:-1], parts[1:]):
                    if clamp:
                        line = [[min(max(lat, south - 0.05), north + 0.05), min(max(lon, west - 0.05), east + 0.05)]
                                for lon, lat in points[start:stop]]
                        if len(line) >= 3:
                            lines.append(line)
                        continue
                    # Outlines: split where points leave the box (instead of joining the gap with a chord)
                    # and at edges along whole degrees, i.e. the tile borders of pre-cut land polygons.
                    line, previous = [], None
                    for lon, lat in points[start:stop]:
                        keep = west - 0.05 <= lon <= east + 0.05 and south - 0.05 <= lat <= north + 0.05
                        tile_edge = previous is not None and ((lon == previous[0] and float(lon).is_integer())
                                                             or (lat == previous[1] and float(lat).is_integer()))
                        if not keep or tile_edge:
                            if len(line) >= 2:
                                lines.append(line)
                            line = []
                        if keep:
                            line.append([lat, lon])
                        previous = (lon, lat)
                    if len(line) >= 2:
                        lines.append(line)
        offset = content_end
    return lines


def _load_coastline(cache_dir: Path,
                    bbox: tuple[float, float, float, float]) -> tuple[list[list[list[float]]], str]:
    """Download a global coastline dataset, then crop it locally.

    The remote request is identical for every location and does not disclose the survey coordinates.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "gshhg-shp-2.3.7.zip"
    if not archive.is_file():
        request = urllib.request.Request(
            "https://www.soest.hawaii.edu/pwessel/gshhg/gshhg-shp-2.3.7.zip",
            headers={"User-Agent": "DroneVision/0.1 coastline-validation"},
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            archive.write_bytes(response.read())
    with zipfile.ZipFile(archive) as bundle:
        # High resolution is substantially denser than Natural Earth 1:10m while
        # remaining practical for an interactive first run. L1 is land/ocean.
        shp_name = next(name for name in bundle.namelist() if name.endswith("GSHHS_h_L1.shp"))
        data = bundle.read(shp_name)
    return _read_shapefile(data, bbox), "GSHHG 2.3.7 high resolution"


def _land_file(bbox: tuple[float, float, float, float]) -> Path | None:
    """Land polygons covering ``bbox``: a regional extract (cut from the OSM planet file when needed)."""
    from .coastdata import land_file
    return land_file(bbox)


def _load_local_coastline(bbox: tuple[float, float, float, float]) -> tuple[list[list[list[float]]], str]:
    path = _land_file(bbox)
    if path is None:
        return [], "Локальная береговая линия"
    return _read_shapefile(path.read_bytes(), bbox), "Локальная OSM береговая линия"


def land_layer(geod, lat0: float, lon0: float, minimum: np.ndarray, maximum: np.ndarray) -> dict:
    """Closed land polygon rings in ENU metres for the "Суша" overlay (even-odd fill: lakes stay open)."""
    bbox = _coastline_bbox(lat0, lon0, minimum, maximum)
    path = _land_file(bbox)
    if path is None:
        return {"source": "Локальная береговая линия", "rings": [],
                "error": "для района нет полигонов суши (python -m dronevision coastline)"}
    try:
        rings = _read_shapefile(path.read_bytes(), bbox, clamp=True)
        return {"source": "Локальная OSM суша", "rings": _coastline_to_enu(rings, geod, lat0, lon0)}
    except Exception as exc:
        return {"source": "Локальная OSM суша", "rings": [], "error": str(exc)}


def _coastline_to_enu(lines: list[list[list[float]]], geod, lat0: float, lon0: float) -> list[list[list[float]]]:
    converted = []
    for line in lines:
        if len(line) >= 2:
            lat, lon = np.asarray(line, dtype=np.float64).T
            converted.append(np.round(enu_offsets(geod, lat0, lon0, lat, lon), 2).tolist())
    return converted


def project_selected_frame(video_path: str, time_s: float, project_dir: str,
                           takeoff_height_m: float = 0.0, horizontal_fov_deg: float = 0.0,
                           resolution_m_per_px: float = 2.0, horizon_offset_percent: float = 0.0,
                           yaw_offset_deg: float = 0.0, adjustments: dict | None = None,
                           bottom_crop_percent: float = 0.0, attitude_correction_deg=None,
                           max_distance_m: float = 0.0) -> dict:
    """Project one video frame onto the sea surface (north up, metres from the takeoff point).

    ``horizontal_fov_deg`` overrides the camera profile (FOV without zoom, 0 = automatic);
    ``yaw_offset_deg`` corrects the FlightRecord heading, which can be off by >10 deg after
    a takeoff from a boat (see docs/projection.md and ``python -m dronevision calibrate``).
    ``resolution_m_per_px`` <= 0 means automatic: the finest sea-surface pixel of this frame
    (``geometry.finest_ground_resolution``), coarsened only when the canvas limit requires it.
    Only the waterline is georeferenced correctly: trees and hills lean away from the camera.
    ``adjustments`` (see ``adjust.py``) change the displayed PNG and the GeoTIFF; the data raster
    used for intensity profiles keeps the original colours.
    ``attitude_correction_deg`` = (yaw, pitch, roll) added to the logged attitude, e.g. from
    ``refine.refine_attitudes`` when two frames are compared.
    """
    cv2, geod = _dependencies()
    video = safe_media_path(video_path)
    context = load_context(video)
    meta = context.meta
    time_s = float(np.clip(time_s, 0, meta["duration"]))
    pose = {key: values[0] for key, values in
            frame_poses(context, [time_s], geod, yaw_offset_deg, takeoff_height_m, horizontal_fov_deg).items()}
    lat0, lon0 = context.takeoff
    camera_offset = pose["offset"]
    altitude = float(pose["altitude"])
    focal_px = float(pose["focal_px"])
    correction = np.zeros(3) if attitude_correction_deg is None else np.asarray(attitude_correction_deg, dtype=np.float64)
    yaw = float((pose["yaw"] + correction[0]) % 360)
    pitch, roll = float(pose["pitch"] + correction[1]), float(pose["roll"] + correction[2])
    basis = camera_basis(yaw, pitch, roll)

    cap = open_video(video)
    try:
        frame_index = int(round(time_s * meta["fps"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise RuntimeError(f"Не удалось прочитать кадр {frame_index}")
    height, width = frame.shape[:2]

    # Geometric horizon plus a safety margin: everything below it is projected.  ``max_distance_m`` > 0 is an
    # optional cut (a far footprint forces a coarse pixel on the size-limited canvas); 0 = no limit.
    max_distance = max(float(max_distance_m or 0.0), 0.0)
    margin = int(round(height * max(0.02, horizon_offset_percent / 100)))
    crop_y = max(horizon_crop_row(width, height, focal_px, basis, altitude, np.inf, margin),
                 horizon_crop_row(width, height, focal_px, basis, altitude, max_distance))
    corners = camera_offset + ground_corners(width, height, focal_px, basis, altitude, crop_y)
    bottom_row = int(round(height * (1 - np.clip(bottom_crop_percent, 0, 95) / 100)))
    if bottom_row <= crop_y + 1:
        raise ValueError("Нижняя граница обрезки находится выше горизонта")

    native_resolution = finest_ground_resolution(width, height, focal_px, basis, altitude, crop_y, bottom_row)
    resolution_auto = not resolution_m_per_px or float(resolution_m_per_px) <= 0
    resolution = native_resolution if resolution_auto else float(resolution_m_per_px)
    span = np.ceil(corners.max(axis=0)) - np.floor(corners.min(axis=0))
    output_size = np.maximum(np.ceil(span / resolution).astype(int), 2)
    if output_size.prod() > MAX_CANVAS_PIXELS or output_size.max() > MAX_CANVAS_SIDE:
        resolution *= max(np.sqrt(output_size.prod() / MAX_CANVAS_PIXELS), output_size.max() / MAX_CANVAS_SIDE)
        resolution = float(np.ceil(resolution * 1000) / 1000)
    # Bounds on multiples of the resolution: projections of one video then share pixel centres and
    # can be overlaid (frame comparison) without resampling.
    minimum = np.floor(corners.min(axis=0) / resolution) * resolution
    maximum = np.ceil(corners.max(axis=0) / resolution) * resolution
    output_size = np.maximum(np.round((maximum - minimum) / resolution).astype(int), 2)

    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - crop_y - 1], [0, height - crop_y - 1]])
    target = np.float32([[(east - minimum[0]) / resolution, (maximum[1] - north) / resolution]
                         for east, north in corners])
    transform = cv2.getPerspectiveTransform(source, target)
    projected = cv2.warpPerspective(frame[crop_y:, :], transform, tuple(map(int, output_size)),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    source_coverage = np.full((height - crop_y, width), 255, np.uint8)
    source_coverage[bottom_row - crop_y:] = 0
    coverage = cv2.warpPerspective(source_coverage, transform,
                                   tuple(map(int, output_size)), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT) >= 128
    projected[~coverage] = 0
    raw_projected = projected
    projected = apply_adjustments(raw_projected, adjustments)
    output_root = Path(project_dir).expanduser().resolve() if project_dir else video.parent / "DroneVision-project"
    output_dir = output_root / "projections"
    output_dir.mkdir(parents=True, exist_ok=True)
    bbox = _coastline_bbox(lat0, lon0, minimum, maximum)
    coastline_layers = []
    try:
        coastline, coastline_source = _load_local_coastline(bbox)
        coastline_layers.append({"source": coastline_source, "color": "#ff5b45",
                                 "lines": _coastline_to_enu(coastline, geod, lat0, lon0)})
    except Exception as exc:
        coastline_layers.append({"source": "Локальная береговая линия", "color": "#ff5b45",
                                 "lines": [], "error": str(exc)})
    projected, grid_step = overlay_metric_grid(cv2, projected, minimum, maximum, resolution)
    orientation = f"yaw{yaw_offset_deg:+.1f}"
    output = output_dir / f"{output_stem(output_dir, context, time_s, time_s, 'projection', ('.png', '.geo.json'))}.png"
    if not imwrite(output, projected):
        raise RuntimeError("Не удалось сохранить проекцию")
    georaster = save_georaster(output, raw_projected, coverage, minimum, maximum, resolution, (lat0, lon0),
                               "projection", adjustments, video=str(video), times_s=[round(time_s, 3)],
                               frame_indices=[frame_index], yaw_offset_deg=yaw_offset_deg,
                               takeoff_height_m=takeoff_height_m, bottom_crop_percent=bottom_crop_percent,
                               horizontal_fov_deg=horizontal_fov_deg, max_distance_m=max_distance,
                               attitude_corrections_deg=[np.round(correction, 4).tolist()],
                               **naming_metadata(context))
    gimbal_yaw, body_yaw = float(pose["gimbal_yaw"]), float(pose["log_yaw"])
    return {
        "path": str(output), "time_s": round(time_s, 3), "frame_index": frame_index,
        "size_px": [int(output_size[0]), int(output_size[1])],
        "bounds_m": {"east": [float(minimum[0]), float(maximum[0])],
                     "north": [float(minimum[1]), float(maximum[1])]},
        "resolution_m_per_px": round(resolution, 4), "grid_step_m": grid_step,
        "resolution_auto": resolution_auto, "native_resolution_m": native_resolution,
        "crop_row": crop_y, "bottom_crop_row": bottom_row, "bottom_crop_percent": float(bottom_crop_percent),
        "source_height": height,
        "orientation": orientation,
        "orientation_label": f"Курс корпуса {'без поправки' if not yaw_offset_deg else f'{yaw_offset_deg:+.1f}°'}"
                             f" · pitch/roll подвеса · FOV {pose['horizontal_fov_deg']:.1f}°"
                             + ("" if pose["gps_valid"] else
                                " · GPS потерян: позиция по счислению"
                                + (f" (±{pose['gps_uncertainty_m']:.0f} м)" if np.isfinite(pose["gps_uncertainty_m"])
                                   else " (точность неизвестна)")),
        "coastline_layers": coastline_layers,
        "coastline": {"source": " + ".join(layer["source"] for layer in coastline_layers),
                      "segments": sum(len(layer["lines"]) for layer in coastline_layers)},
        "land": land_layer(geod, lat0, lon0, minimum, maximum),
        **georaster,
        "telemetry": {"latitude": float(pose["lat"]), "longitude": float(pose["lon"]),
                      "height_above_water_m": altitude, "flight_altitude_m": float(pose["flight_altitude"]),
                      "takeoff_height_m": takeoff_height_m,
                      "altitude_source": "FlightRecord относительно взлёта + высота взлёта над водой",
                      "yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll,
                      "body_yaw_deg": body_yaw, "yaw_offset_deg": yaw_offset_deg,
                      "attitude_correction_deg": np.round(correction, 3).tolist(),
                      "gimbal_yaw_deg": gimbal_yaw,
                      "gimbal_relative_yaw_deg": ((gimbal_yaw - body_yaw + 180) % 360) - 180,
                      "body_pitch_deg": float(pose["body_pitch"]), "body_roll_deg": float(pose["body_roll"]),
                      "zoom": float(pose["zoom"]), "focal_length_mm": float(pose["focal_mm"]),
                      "focal_length_px": focal_px, "horizontal_fov_deg": float(pose["horizontal_fov_deg"]),
                      "camera_profile": context.profile,
                      "gps_valid": bool(pose["gps_valid"]),
                      "gps_uncertainty_m": None if not np.isfinite(pose["gps_uncertainty_m"])
                      else round(float(pose["gps_uncertainty_m"]), 1),
                      "log_sync": {"utc_offset_hours": context.utc_offset_hours,
                                   "clock_offset_seconds": context.clock_offset_seconds}},
        "takeoff": {"latitude": lat0, "longitude": lon0},
    }


def _visible_land_fraction(context, poses, geod, max_distance_m: float = 5000.0) -> float | None:
    """Share of the frames' sea-surface footprints (≤ ``max_distance_m``) covered by OSM land.

    None when the local land polygons do not cover the area (then static details cannot be vouched for).
    """
    from .calibration import _land_rings
    from .geometry import camera_basis, ground_corners, horizon_crop_row

    cv2, _ = _dependencies()
    width, height = context.meta["width"], context.meta["height"]
    polygons = []
    for i in range(len(poses["time_s"])):
        basis = camera_basis(poses["yaw"][i], poses["pitch"][i], poses["roll"][i])
        try:
            crop = max(horizon_crop_row(width, height, poses["focal_px"][i], basis, poses["altitude"][i], np.inf,
                                        int(0.02 * height)),
                       horizon_crop_row(width, height, poses["focal_px"][i], basis, poses["altitude"][i], max_distance_m))
            polygons.append(poses["offset"][i] + ground_corners(width, height, poses["focal_px"][i], basis,
                                                                poses["altitude"][i], crop))
        except ValueError:
            continue
    if not polygons:
        return None
    lat0, lon0 = context.takeoff
    points = np.concatenate(polygons)
    lower, upper = points.min(axis=0) - 50, points.max(axis=0) + 50
    west, south, east, north = _coastline_bbox(lat0, lon0, lower, upper)
    land_path = _land_file((west, south, east, north))
    if land_path is None:
        return None
    cell = 20.0
    size = np.ceil((upper - lower) / cell).astype(int) + 1
    to_grid = lambda xy: np.round((xy - lower) / cell).astype(np.int32).reshape(-1, 1, 2)
    footprint = np.zeros((size[1], size[0]), np.uint8)
    for polygon in polygons:
        cv2.fillPoly(footprint, [to_grid(polygon)], 1)
    land = np.zeros_like(footprint)
    for ring, outer in _land_rings(land_path, (west, south, east, north)):
        cv2.fillPoly(land, [to_grid(enu_offsets(geod, lat0, lon0, ring[:, 1], ring[:, 0]))], 1 if outer else 0)
    covered = footprint.astype(bool)
    return float(land[covered].mean()) if covered.any() else None


def _pair_corrections(video_path: str, frames: list[dict], takeoff_height_m: float, horizontal_fov_deg: float,
                      yaw_offset_deg: float) -> tuple[np.ndarray, dict]:
    """Attitude corrections that make two frames agree on static details (islands, shore, clouds)."""
    from .refine import frame_features, refine_attitudes

    cv2, geod = _dependencies()
    context = load_context(safe_media_path(video_path))
    meta = context.meta
    times = [float(np.clip(float(item["time_s"]), 0, meta["duration"])) for item in frames]
    poses = frame_poses(context, times, geod, yaw_offset_deg, takeoff_height_m, horizontal_fov_deg)
    # Over open water the only matchable details are the water itself: aligning on them would cancel the
    # very drift the comparison is meant to measure.  Require visible land (OSM) for the pair alignment.
    land = _visible_land_fraction(context, poses, geod)
    if land is None or land < 0.005:
        note = ("в кадрах нет суши — выравнивание пары по изображению отключено, иначе оно «съело» бы дрейф воды"
                if land is not None else
                "нет данных о суше для этого района — выравнивание пары отключено, чтобы не «съесть» дрейф воды")
        return np.zeros((len(frames), 3)), {"method": "image_feature_attitude_refinement", "frames": len(frames),
                                            "edges": 0, "skipped": True, "land_fraction": land, "note": note}
    features = []
    cap = open_video(context.video)
    try:
        for item, time_s in zip(frames, times):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(time_s * meta["fps"])))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Не удалось прочитать кадр {time_s:.3f} с")
            bottom = float(item.get("bottom_crop_percent") or 0.0)
            features.append(frame_features(cv2, frame, int(round(frame.shape[0] * (1 - np.clip(bottom, 0, 95) / 100)))))
    finally:
        cap.release()
    corrections, report = refine_attitudes(cv2, features, poses, meta["width"], meta["height"])
    report["land_fraction"] = round(land, 4)
    return corrections, report


def project_frame_pair(video_path: str, frames: list[dict], project_dir: str,
                       takeoff_height_m: float = 0.0, horizontal_fov_deg: float = 0.0,
                       resolution_m_per_px: float = 2.0, yaw_offset_deg: float = 0.0,
                       refine: bool = True, max_distance_m: float = 0.0) -> dict:
    """Project two frames onto one identical ENU canvas for flicker comparison.

    With ``refine`` the two attitudes are first aligned by image features (``refine.py``): otherwise the
    0.1° quantisation and drift of the logged angles alone move distant water by tens of metres, which
    would be read as slick motion.
    """
    if len(frames) != 2:
        raise ValueError("Для сравнения нужно выбрать ровно два кадра")
    ordered = sorted(frames, key=lambda item: float(item["time_s"]))
    corrections, refinement = [None, None], None
    if refine:
        corrections, refinement = _pair_corrections(video_path, ordered, takeoff_height_m, horizontal_fov_deg,
                                                    yaw_offset_deg)
    variants = [project_selected_frame(
        video_path, float(item["time_s"]), project_dir, takeoff_height_m, horizontal_fov_deg,
        resolution_m_per_px, float(item.get("horizon_offset_percent") or 0.0), yaw_offset_deg,
        adjustments=item.get("adjustments"),
        bottom_crop_percent=float(item.get("bottom_crop_percent") or 0.0),
        attitude_correction_deg=correction, max_distance_m=max_distance_m,
    ) for item, correction in zip(ordered, corrections)]

    cv2, geod = _dependencies()
    # Each source bound is already snapped to its resolution. Preserve that common pixel grid: rounding
    # the union to whole metres would shift the georeference by a fraction of a pixel.
    minimum = np.min([[v["bounds_m"]["east"][0], v["bounds_m"]["north"][0]] for v in variants], axis=0)
    maximum = np.max([[v["bounds_m"]["east"][1], v["bounds_m"]["north"][1]] for v in variants], axis=0)
    resolution = max(float(v["resolution_m_per_px"]) for v in variants)
    size = np.maximum(np.ceil((maximum - minimum) / resolution).astype(int), 2)
    if size.prod() > MAX_CANVAS_PIXELS or size.max() > MAX_CANVAS_SIDE:
        resolution *= max(np.sqrt(size.prod() / MAX_CANVAS_PIXELS), size.max() / MAX_CANVAS_SIDE)
        size = np.maximum(np.ceil((maximum - minimum) / resolution).astype(int), 2)
    canvas_w, canvas_h = map(int, size)
    lat0, lon0 = variants[0]["takeoff"]["latitude"], variants[0]["takeoff"]["longitude"]
    bbox = _coastline_bbox(lat0, lon0, minimum, maximum)
    try:
        coastline, source = _load_local_coastline(bbox)
        coastline_layers = [{"source": source, "color": "#ff5b45",
                             "lines": _coastline_to_enu(coastline, geod, lat0, lon0)}]
    except Exception as exc:
        coastline_layers = [{"source": "Локальная береговая линия", "color": "#ff5b45",
                             "lines": [], "error": str(exc)}]
    shared_land = land_layer(geod, lat0, lon0, minimum, maximum)
    output_dir = Path(variants[0]["path"]).parent
    context = load_context(safe_media_path(video_path))
    span = float(ordered[0]["time_s"]), float(ordered[1]["time_s"])
    aligned = []
    for index, (item, variant) in enumerate(zip(ordered, variants), 1):
        source_rgba = imread(variant["data_path"], cv2.IMREAD_UNCHANGED)
        if source_rgba is None:
            raise RuntimeError("Не удалось прочитать растр проекции для сравнения")
        source_resolution = float(variant["resolution_m_per_px"])
        if not np.isclose(source_resolution, resolution):
            target_size = (max(1, int(round(source_rgba.shape[1] * source_resolution / resolution))),
                           max(1, int(round(source_rgba.shape[0] * source_resolution / resolution))))
            source_rgba = cv2.resize(source_rgba, target_size, interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
        x0 = int(round((variant["bounds_m"]["east"][0] - minimum[0]) / resolution))
        y0 = int(round((maximum[1] - variant["bounds_m"]["north"][1]) / resolution))
        x1, y1 = min(canvas_w, x0 + source_rgba.shape[1]), min(canvas_h, y0 + source_rgba.shape[0])
        if x1 > x0 and y1 > y0:
            canvas[y0:y1, x0:x1] = source_rgba[:y1 - y0, :x1 - x0]
        valid = canvas[..., 3] >= 128
        raw = canvas[..., :3].copy(); raw[~valid] = 0
        display = apply_adjustments(raw, item.get("adjustments"))
        display, grid_step = overlay_metric_grid(cv2, display, minimum, maximum, resolution)
        output = output_dir / f"{output_stem(output_dir, context, *span, f'compare{index}', ('.png', '.geo.json'))}.png"
        if not imwrite(output, display):
            raise RuntimeError("Не удалось сохранить сравниваемую проекцию")
        georaster = save_georaster(
            output, raw, valid, minimum, maximum, resolution, (lat0, lon0), "projection_comparison",
            item.get("adjustments"), video=str(Path(video_path).expanduser().resolve()),
            times_s=[round(float(item["time_s"]), 3)], frame_indices=[variant["frame_index"]],
            yaw_offset_deg=yaw_offset_deg, takeoff_height_m=takeoff_height_m,
            bottom_crop_percent=float(item.get("bottom_crop_percent") or 0.0),
            horizontal_fov_deg=horizontal_fov_deg, max_distance_m=max_distance_m,
            attitude_corrections_deg=[variant["telemetry"]["attitude_correction_deg"]],
            pair_times_s=[round(span[0], 3), round(span[1], 3)], **naming_metadata(context),
        )
        variant.update({
            "path": str(output), "size_px": [canvas_w, canvas_h],
            "bounds_m": {"east": [float(minimum[0]), float(maximum[0])],
                         "north": [float(minimum[1]), float(maximum[1])]},
            "resolution_m_per_px": round(float(resolution), 4), "grid_step_m": grid_step,
            "coastline_layers": coastline_layers,
            "coastline": {"source": " + ".join(layer["source"] for layer in coastline_layers),
                          "segments": sum(len(layer["lines"]) for layer in coastline_layers)},
            "land": shared_land, **georaster,
        })
        aligned.append(variant)
    first, second = aligned
    _, _, displacement = geod.inv(first["telemetry"]["longitude"], first["telemetry"]["latitude"],
                                  second["telemetry"]["longitude"], second["telemetry"]["latitude"])
    comparison = {
        "delta_time_s": round(second["time_s"] - first["time_s"], 3),
        "drone_displacement_m": round(float(displacement), 3),
        "yaw_delta_deg": round(float((second["telemetry"]["yaw_deg"] - first["telemetry"]["yaw_deg"] + 180) % 360 - 180), 3),
        "pitch_delta_deg": round(second["telemetry"]["pitch_deg"] - first["telemetry"]["pitch_deg"], 3),
        "refinement": refinement,
    }
    return {"variants": aligned, "time_s": first["time_s"], "frame_index": first["frame_index"],
            "comparison": comparison}
