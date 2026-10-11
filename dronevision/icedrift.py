"""«Дрейф льдин» (experimental): velocity field of drifting ice floes from the video itself.

Method: corner points on the floes are followed through the interval with pyramidal Lucas–Kanade tracking
(forward–backward check), every position is put on the sea surface with the camera pose of its frame, and a
straight line fitted to east(t), north(t) of a track gives its velocity.  Tracks are then gathered on a ground
grid: the median of a cell is its drift vector, the spread of its tracks is its uncertainty.

Everything is automatic: the tracking scale follows the pixel footprint on the water, the cell size follows
the size of the frame on the water.  What it cannot check: with open water all around there is nothing fixed
in the frame, so the logged heading and GPS position of the drone are trusted.  A heading drift of 0.01°/s is
a false 0.05 m/s at 300 m from the nadir; the fitted rotation of the whole field (``rotation_deg_s``) shows
how much of this — or of a real turning of the ice — is present.

Checked on DJI_0204 and DJI_0331 (2023-07-24, nadir from ~480 m, sparse ice).
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .cvio import imwrite, open_video
from .geometry import camera_basis, ground_points, pixel_rays
from .naming import metadata as naming_metadata, output_stem
from .pose import frame_poses, load_context
from .wavefield import _compass, _dependencies, _footprint

SAMPLE_RATE_HZ = 2.0
MAX_SECONDS = 120.0
MIN_SECONDS = 10.0
TRACK_WIDTH = 1920            # frames are tracked at this width
MAX_POINTS = 4000
MAX_DRAWN_TRACKS = 700


def _ice_mask(cv2, gray: np.ndarray) -> np.ndarray:
    """Bright compact objects on dark water: Otsu threshold, opened to drop sun glitter of a few pixels."""
    level, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (gray > max(level, 90)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))


def _to_ground(points_px: np.ndarray, poses: dict, i: int, width: int, height: int) -> np.ndarray:
    """ENU (relative to takeoff) of full-resolution image points on the sea surface for frame ``i``; NaN above the horizon."""
    basis = camera_basis(poses["yaw"][i], poses["pitch"][i], poses["roll"][i])
    rays = pixel_rays(points_px[:, 0], points_px[:, 1], width, height, float(poses["focal_px"][i]), basis)
    ground = ground_points(rays, float(poses["altitude"][i]))[:, :2]
    ground[~np.isfinite(ground).all(axis=1)] = np.nan
    return ground + poses["offset"][i]


def _nice(value: float) -> float:
    power = 10 ** np.floor(np.log10(max(value, 1e-6)))
    fraction = value / power
    return float((1 if fraction <= 1.5 else 2 if fraction <= 3.5 else 5 if fraction <= 7.5 else 10) * power)


def _rigid(east, north, u, v, weight) -> dict | None:
    """Mean drift at the weighted centre plus a rotation of the whole field: v = U + Ω ẑ × (r − centre)."""
    if len(east) < 3:
        return None
    centre = np.array([np.average(east, weights=weight), np.average(north, weights=weight)])
    x, y, root = east - centre[0], north - centre[1], np.sqrt(weight)
    design = np.concatenate([np.column_stack([np.ones_like(x), np.zeros_like(x), -y]),
                             np.column_stack([np.zeros_like(x), np.ones_like(x), x])]) * np.r_[root, root][:, None]
    target = np.r_[u, v] * np.r_[root, root]
    if np.linalg.matrix_rank(design) < 3:
        return None
    solution = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = (design @ solution - target) / np.r_[root, root]
    return {"u": float(solution[0]), "v": float(solution[1]), "omega_rad_s": float(solution[2]), "centre": centre,
            "residual_m_s": float(np.sqrt((residual ** 2).reshape(2, -1).sum(axis=0).mean()))}


def analyse(video_path: str, start_s: float | None, end_s: float | None, project_dir: str,
            yaw_offset_deg: float = 0.0, takeoff_height_m: float = 0.0, horizontal_fov_deg: float = 0.0,
            progress=lambda share, text: None) -> dict:
    """Drift vectors of ice floes on a ground grid for an interval of the video."""
    from .georaster import _transformer, aeqd_crs, save_georaster
    from .viewer import safe_media_path

    cv2, geod = _dependencies()
    video = safe_media_path(video_path)
    context = load_context(video)
    meta = context.meta
    width, height, fps = meta["width"], meta["height"], meta["fps"]
    start = 0.0 if start_s is None else max(0.0, float(start_s))
    end = meta["duration"] if end_s is None else min(float(end_s), meta["duration"])
    notes = []
    if end - start > MAX_SECONDS:
        end = start + MAX_SECONDS
        notes.append(f"интервал сокращён до первых {MAX_SECONDS:.0f} с")
    if end - start < MIN_SECONDS:
        raise ValueError(f"Нужно не меньше {MIN_SECONDS:.0f} с записи (лучше 30–120 с): льдины должны сместиться "
                         "хотя бы на несколько пикселей")
    step = max(1, int(round(fps / SAMPLE_RATE_HZ)))
    indices = np.arange(int(np.ceil(start * fps)), int(np.floor(end * fps)), step)
    indices = indices[indices < max(int(meta["frames"]) or 10 ** 9, 1)]
    times = indices / fps
    poses = frame_poses(context, times, geod, yaw_offset_deg, takeoff_height_m, horizontal_fov_deg)
    shrink = TRACK_WIDTH / width if width > TRACK_WIDTH else 1.0
    small = (int(round(width * shrink)), int(round(height * shrink)))
    lucas = dict(winSize=(25, 25), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    cap = open_video(video)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(indices[0]))
        ok, first_frame = cap.read()
        if not ok:
            raise RuntimeError("Не удалось прочитать первый кадр интервала")
        previous = cv2.cvtColor(cv2.resize(first_frame, small, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        mask = _ice_mask(cv2, previous)
        ice_share = float((mask > 0).mean())
        corners = cv2.goodFeaturesToTrack(previous, MAX_POINTS, 0.01, 10, mask=mask, blockSize=7)
        if corners is None or len(corners) < 20:
            raise ValueError("В кадре не найдено льдин: нужны светлые объекты на тёмной воде")
        points = corners.reshape(-1, 2).astype(np.float32)
        alive = np.ones(len(points), bool)
        # ground[n, track] — position at sample n, NaN after the track is lost
        ground = np.full((len(indices), len(points), 2), np.nan)
        ground[0] = _to_ground(points / shrink, poses, 0, width, height)
        position = int(indices[0]) + 1
        for n, index in enumerate(indices[1:], 1):
            while position < index:
                if not cap.grab():
                    raise RuntimeError(f"Не удалось прочитать кадр {position}")
                position += 1
            ok, frame = cap.read()
            position += 1
            if not ok:
                raise RuntimeError(f"Не удалось прочитать кадр {index}")
            gray = cv2.cvtColor(cv2.resize(frame, small, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            live = np.flatnonzero(alive)
            if len(live) < 10:
                break
            moved, found, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points[live].reshape(-1, 1, 2), None, **lucas)
            back, found_back, _ = cv2.calcOpticalFlowPyrLK(gray, previous, moved, None, **lucas)
            moved, back = moved.reshape(-1, 2), back.reshape(-1, 2)
            good = ((found.ravel() == 1) & (found_back.ravel() == 1)
                    & (np.linalg.norm(back - points[live], axis=1) < 0.7)
                    & (moved[:, 0] > 4) & (moved[:, 0] < small[0] - 4) & (moved[:, 1] > 4) & (moved[:, 1] < small[1] - 4))
            alive[live[~good]] = False
            points[live[good]] = moved[good]
            ground[n, live[good]] = _to_ground(moved[good] / shrink, poses, n, width, height)
            previous = gray
            if n % 20 == 0:
                progress(0.05 + 0.75 * n / len(indices), f"Слежу за льдинами: кадр {n} из {len(indices)}, точек {int(alive.sum())}")
    finally:
        cap.release()

    # --- velocity of every track: straight line through its positions ---
    progress(0.82, "Скорости льдин")
    duration = float(times[-1] - times[0])
    seen = np.isfinite(ground).all(axis=2)
    long_enough = seen.sum(axis=0) >= max(6, int(0.6 * len(indices)))
    rows = []
    for j in np.flatnonzero(long_enough):
        t, track = times[seen[:, j]], ground[seen[:, j], j]
        slope_e, start_e = np.polyfit(t - t[0], track[:, 0], 1)
        slope_n, start_n = np.polyfit(t - t[0], track[:, 1], 1)
        misfit = np.hypot(track[:, 0] - (start_e + slope_e * (t - t[0])), track[:, 1] - (start_n + slope_n * (t - t[0])))
        rows.append((track[0, 0], track[0, 1], track[-1, 0], track[-1, 1], slope_e, slope_n, float(np.sqrt((misfit ** 2).mean())),
                     float(t[-1] - t[0])))
    if len(rows) < 10:
        raise ValueError("Льдины не удалось проследить: слишком мало устойчивых точек (блики, пена или слишком быстрая съёмка)")
    tracks = np.array(rows)
    # Wobbling points (glitter, foam, the edge of the frame) fit a line badly.
    limit = max(3 * float(np.median(tracks[:, 6])), 0.15)
    tracks = tracks[tracks[:, 6] <= limit]

    # --- grid ---
    middle = len(indices) // 2
    reach = 25 * float(poses["altitude"][middle])
    footprint = _footprint(poses, middle, width, height, reach)
    lower, upper = footprint.min(axis=0), footprint.max(axis=0)
    cell_m = _nice(float(np.max(upper - lower)) / 10)
    lat0, lon0 = context.takeoff
    to_lonlat = _transformer(aeqd_crs(lat0, lon0), "EPSG:4326")
    column = np.floor((tracks[:, 0] - lower[0]) / cell_m).astype(int)
    row = np.floor((tracks[:, 1] - lower[1]) / cell_m).astype(int)
    cells = []
    for key in sorted(set(zip(column.tolist(), row.tolist()))):
        inside = (column == key[0]) & (row == key[1])
        u, v = tracks[inside, 4], tracks[inside, 5]
        if inside.sum() < 3:
            continue
        mu, mv = float(np.median(u)), float(np.median(v))
        spread = float(1.4826 * np.median(np.hypot(u - mu, v - mv)))
        centre = lower + (np.array(key) + 0.5) * cell_m
        lon, lat = to_lonlat.transform(float(centre[0]), float(centre[1]))
        cells.append({"east_m": round(float(centre[0]), 1), "north_m": round(float(centre[1]), 1),
                      "latitude": round(float(lat), 7), "longitude": round(float(lon), 7), "size_m": cell_m,
                      "u_east_m_s": round(mu, 3), "v_north_m_s": round(mv, 3), "speed_m_s": round(float(np.hypot(mu, mv)), 3),
                      "direction_to_deg": round(float(np.rad2deg(np.arctan2(mu, mv)) % 360), 1),
                      "tracks": int(inside.sum()), "spread_m_s": round(spread, 3),
                      "quality": "good" if inside.sum() >= 6 and spread < 0.05 else "weak"})
    if not cells:
        raise ValueError("Точек на льдинах слишком мало, чтобы собрать поле скоростей")

    # --- whole field: mean drift, rotation, what stands out ---
    east, north = np.array([c["east_m"] for c in cells]), np.array([c["north_m"] for c in cells])
    u, v = np.array([c["u_east_m_s"] for c in cells]), np.array([c["v_north_m_s"] for c in cells])
    weight = np.array([c["tracks"] for c in cells], float)
    median = np.array([np.median(tracks[:, 4]), np.median(tracks[:, 5])])
    camera_position = poses["offset"].mean(axis=0)
    regular = np.ones(len(cells), bool)
    rigid = None
    for _ in range(3):
        rigid = _rigid(east[regular], north[regular], u[regular], v[regular], weight[regular]) if regular.sum() >= 3 else None
        if rigid is None:
            break
        model_u = rigid["u"] - rigid["omega_rad_s"] * (north - rigid["centre"][1])
        model_v = rigid["v"] + rigid["omega_rad_s"] * (east - rigid["centre"][0])
        apart = np.hypot(u - model_u, v - model_v)
        # A ship, a floe pushed by it or a mistracked patch: far from what the rest of the field does.
        regular = apart <= max(5 * float(1.4826 * np.median(apart[regular])), 0.08)
    for c, ordinary in zip(cells, regular):
        c["outlier"] = bool(not ordinary)
    # The drift under the drone does not depend on a turn of the camera about its axis: the turn moves nothing there.
    omega = rigid["omega_rad_s"] if rigid else 0.0
    if rigid:
        mean = np.array([rigid["u"] - omega * (camera_position[1] - rigid["centre"][1]),
                         rigid["v"] + omega * (camera_position[0] - rigid["centre"][0])])
    else:
        mean = median
    for c in cells:
        c.update(measured_u_east_m_s=c["u_east_m_s"], measured_v_north_m_s=c["v_north_m_s"])
        cu = c["u_east_m_s"] + omega * (c["north_m"] - camera_position[1])
        cv = c["v_north_m_s"] - omega * (c["east_m"] - camera_position[0])
        c.update(u_east_m_s=round(float(cu), 3), v_north_m_s=round(float(cv), 3), speed_m_s=round(float(np.hypot(cu, cv)), 3),
                 direction_to_deg=round(float(np.rad2deg(np.arctan2(cu, cv)) % 360), 1))
    camera_position = poses["offset"].mean(axis=0)
    drone_velocity = (poses["offset"][-1] - poses["offset"][0]) / max(duration, 1e-6)
    drone_speed = float(np.hypot(*drone_velocity))
    edge = float(np.max(np.linalg.norm(footprint - camera_position, axis=1)))
    reasons = []
    if drone_speed > 0.5:
        reasons.append(f"дрон летит со скоростью {drone_speed:.1f} м/с: ошибка курса камеры в 1° даёт "
                       f"{drone_speed * 0.0175:.2f} м/с ложного дрейфа — нужно висение")
    if float(np.ptp(poses["altitude"])) > 0.03 * float(poses["altitude"].mean()):
        reasons.append("высота меняется за интервал: масштаб кадра плывёт")
    if len(tracks) < 60:
        reasons.append(f"прослежено только {len(tracks)} точек")
    summary = {"u_east_m_s": round(float(mean[0]), 3), "v_north_m_s": round(float(mean[1]), 3),
               "speed_m_s": round(float(np.hypot(*mean)), 3),
               "direction_to_deg": round(float(np.rad2deg(np.arctan2(mean[0], mean[1])) % 360), 0),
               "uncertainty_m_s": None if rigid is None else round(float(max(rigid["residual_m_s"] / np.sqrt(max(regular.sum(), 1)), 0.01)), 3),
               "measured_median_u_m_s": round(float(median[0]), 3), "measured_median_v_m_s": round(float(median[1]), 3),
               "rotation_deg_s": None if rigid is None else round(float(np.rad2deg(rigid["omega_rad_s"])), 4),
               "rotation_edge_m_s": None if rigid is None else round(abs(rigid["omega_rad_s"]) * edge, 3),
               "field_residual_m_s": None if rigid is None else round(rigid["residual_m_s"], 3),
               "tracks": int(len(tracks)), "points_started": int(len(points)), "cells": len(cells),
               "outlier_cells": int((~regular).sum()), "ice_share_percent": round(100 * ice_share, 1),
               "drone_speed_m_s": round(drone_speed, 2),
               "reliability": "unreliable" if drone_speed > 0.5 else "weak" if reasons else "good", "reasons": reasons}
    summary["direction_to_text"] = _compass(summary["direction_to_deg"])
    # One step of the tracker is ~0.1 px; over the record this is the floor of a track's speed error.
    pixel_m = float(np.median(np.linalg.norm(ground[0, 1:] - ground[0, :-1], axis=1) /
                              np.maximum(np.linalg.norm((corners.reshape(-1, 2)[1:] - corners.reshape(-1, 2)[:-1]) / shrink, axis=1), 1e-6)))
    summary["pixel_m"] = round(pixel_m, 3)
    summary["speed_floor_m_s"] = round(0.3 * pixel_m / shrink / max(duration, 1e-6), 4)
    if rigid is not None and summary["rotation_edge_m_s"] > 0.03:
        notes.append(f"всё поле поворачивается на {abs(summary['rotation_deg_s']):.3f}°/с ({summary['rotation_edge_m_s']:.2f} м/с на краю "
                     "кадра). В открытой воде нет неподвижных ориентиров, поэтому увод курса камеры, которого нет в логе, "
                     "неотличим от настоящего вращения льда; для свободно дрейфующих льдин вероятнее первое. Скорость под "
                     "дроном от этого не зависит, а векторы показаны за вычетом поворота (галочка над картой возвращает его)")
    else:
        notes.append("в открытой воде нет неподвижных ориентиров: курс и положение дрона взяты из лога")
    if summary["outlier_cells"]:
        notes.append(f"участков, выпадающих из общего движения: {summary['outlier_cells']} (судно, льдина у борта "
                     "или ошибка слежения) — они показаны красным и не входят в общий дрейф")

    # --- files ---
    progress(0.9, "Сохраняю результаты")
    output_root = Path(project_dir).expanduser().resolve() if project_dir else video.parent / "DroneVision-project"
    folder = output_root / "ice"
    folder.mkdir(parents=True, exist_ok=True)
    stem = output_stem(folder, context, times[0], times[-1], "icedrift", (".png", ".json", ".csv", ".geo.json"))
    pad = 0.04 * float(np.max(upper - lower))
    lower, upper = lower - pad, upper + pad
    map_resolution = float(max(pixel_m, np.max(upper - lower) / 1600))
    map_w, map_h = (np.ceil((upper - lower) / map_resolution)).astype(int)
    basis = camera_basis(poses["yaw"][0], poses["pitch"][0], poses["roll"][0])
    relative = lower - poses["offset"][0]
    to_ground = np.array([[map_resolution, 0, relative[0]], [0, -map_resolution, relative[1] + map_h * map_resolution],
                          [0, 0, -poses["altitude"][0]]], dtype=np.float64)
    camera = np.array([[poses["focal_px"][0], 0, width / 2], [0, poses["focal_px"][0], height / 2], [0, 0, 1]])
    matrix = camera @ np.stack([basis[1], basis[2], basis[0]]) @ to_ground
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    overview = cv2.warpPerspective(first_frame, matrix, (int(map_w), int(map_h)), flags=flags)
    valid = cv2.warpPerspective(np.full((height, width), 255, np.uint8), matrix, (int(map_w), int(map_h)), flags=flags) > 127
    try:
        outline = _footprint(poses, 0, width, height, reach)
        inside = np.zeros((map_h, map_w), np.uint8)
        cv2.fillConvexPoly(inside, np.round(np.column_stack([(outline[:, 0] - lower[0]) / map_resolution,
                                                             (lower[1] + map_h * map_resolution - outline[:, 1]) / map_resolution])).astype(np.int32), 1)
        valid &= inside > 0
    except ValueError:
        pass
    overview[~valid] = 0
    image_path = folder / f"{stem}.png"
    if not imwrite(image_path, overview):
        raise RuntimeError("Не удалось сохранить карту")
    georaster = save_georaster(image_path, overview, valid, lower, lower + np.array([map_w, map_h]) * map_resolution,
                               map_resolution, (lat0, lon0), "ice_drift_map", None, video=str(video),
                               times_s=[round(float(times[0]), 3), round(float(times[-1]), 3)],
                               yaw_offset_deg=yaw_offset_deg, takeoff_height_m=takeoff_height_m,
                               **naming_metadata(context))
    utc_start = datetime.fromtimestamp(context.start_epoch + float(times[0]) - context.utc_offset_hours * 3600,
                                       timezone.utc).isoformat(timespec="seconds")
    drawn = tracks[np.linspace(0, len(tracks) - 1, min(len(tracks), MAX_DRAWN_TRACKS)).astype(int)]
    result = {
        "kind": "ice_drift", "video": str(video), "interval_s": [round(float(times[0]), 3), round(float(times[-1]), 3)],
        "utc_start": utc_start, **naming_metadata(context),
        "frames": int(len(indices)), "frame_rate_hz": round(fps / step, 2), "cell_m": cell_m, "cells": cells,
        "summary": summary, "notes": notes, "yaw_offset_deg": yaw_offset_deg, "takeoff_height_m": takeoff_height_m,
        "takeoff": {"latitude": lat0, "longitude": lon0}, "camera_m": np.round(camera_position, 1).tolist(),
        "camera_altitude_m": round(float(poses["altitude"].mean()), 1),
        # start east, start north, u, v of a sample of the tracks, for the picture
        "tracks": np.round(drawn[:, [0, 1, 4, 5]], 3).tolist(),
        "map": {"path": str(image_path), "size_px": [int(map_w), int(map_h)], "resolution_m_per_px": map_resolution,
                "bounds_m": {"east": [float(lower[0]), float(lower[0] + map_w * map_resolution)],
                             "north": [float(lower[1]), float(lower[1] + map_h * map_resolution)]},
                "geo_grid": georaster["geo_grid"], "footprint_m": np.round(footprint, 1).tolist()},
        "method": "Lucas-Kanade tracking of corners on ice floes, ground positions from the logged pose, line fit per track",
    }
    json_path, csv_path, tracks_path = folder / f"{stem}.json", folder / f"{stem}.csv", folder / f"{stem}_tracks.csv"
    result.update(path=str(json_path), csv_path=str(csv_path), tracks_path=str(tracks_path))
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    header = ["latitude", "longitude", "east_m", "north_m", "u_east_m_s", "v_north_m_s", "speed_m_s", "direction_to_deg",
              "measured_u_east_m_s", "measured_v_north_m_s", "tracks", "spread_m_s", "quality", "outlier"]
    lines = [";".join(header)] + [";".join(str(c.get(key, "")) for key in header) for c in cells]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    track_lines = ["start_east_m;start_north_m;end_east_m;end_north_m;u_east_m_s;v_north_m_s;fit_rms_m;duration_s"]
    track_lines += [";".join(f"{value:.3f}" for value in track) for track in tracks]
    tracks_path.write_text("\n".join(track_lines) + "\n", encoding="utf-8-sig")
    progress(1.0, "Готово")
    return result
