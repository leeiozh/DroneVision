"""Follow hand-drawn closed contours (ice floes, foam patches) and open polylines from one georeferenced raster to another.

Every contour is matched on its own: translation and rotation of the image inside the contour are found by
masked normalised cross-correlation of the two data rasters (``<stem>_data.png``).  Nothing is shared
between contours, so floes in different parts of the frame may drift and spin differently.

The search runs twice: around the common shift of the two rasters (phase correlation), and — when that
match is poor — in a wide window around zero limited by ``max_speed_m_s``.  The match is then refined at
a finer scale.  Rotation is positive clockwise seen from above (like a heading).
"""
from __future__ import annotations

import math

import numpy as np

from .georaster import load_data, load_georaster

LUMA_BGR = np.array([0.114, 0.587, 0.299], dtype=np.float32)
COARSE_TEMPLATE_PX = 72        # longer side of the contour box in the coarse search
FINE_TEMPLATE_PX = 320         # ... and in the refinement
MAX_SEARCH_PX = 1500           # side of the coarse search window; the scale is coarsened to respect it
ANGLES_COARSE = np.arange(-24.0, 24.01, 3.0)


def _grey(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return image[..., :3].astype(np.float32) @ LUMA_BGR, image[..., 3] > 0


def _window(meta: dict, grey: np.ndarray, valid: np.ndarray, centre, half_m: float, resolution: float):
    """Square north-up window around ``centre`` (east, north) resampled to ``resolution`` m/px.

    Pixel (i, j) of the window has its centre at (east0 + i*res, north0 - j*res).
    """
    import cv2

    size = 2 * int(math.ceil(half_m / resolution)) + 1
    east0, north0 = centre[0] - (size // 2) * resolution, centre[1] + (size // 2) * resolution
    source = float(meta["resolution_m_per_px"])
    scale = resolution / source
    # window pixel -> raster pixel
    matrix = np.float32([[scale, 0, (east0 - meta["bounds_m"]["east"][0]) / source],
                         [0, scale, (meta["bounds_m"]["north"][1] - north0) / source]])
    x0, y0 = int(math.floor(matrix[0, 2] - 2 * scale)) - 2, int(math.floor(matrix[1, 2] - 2 * scale)) - 2
    x1, y1 = int(math.ceil(matrix[0, 2] + (size + 2) * scale)) + 2, int(math.ceil(matrix[1, 2] + (size + 2) * scale)) + 2
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, grey.shape[1]), min(y1, grey.shape[0])
    if cx1 - cx0 < 2 or cy1 - cy0 < 2:
        return np.zeros((size, size), np.float32), np.zeros((size, size), np.float32), (east0, north0)
    crop, crop_valid = grey[cy0:cy1, cx0:cx1], valid[cy0:cy1, cx0:cx1].astype(np.float32)
    if scale > 1.5:      # low-pass before decimation
        sigma = 0.45 * scale
        crop, crop_valid = cv2.GaussianBlur(crop, (0, 0), sigma), cv2.GaussianBlur(crop_valid, (0, 0), sigma)
    matrix[:, 2] -= (cx0, cy0)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    return (cv2.warpAffine(crop, matrix, (size, size), flags=flags, borderValue=0),
            cv2.warpAffine(crop_valid, matrix, (size, size), flags=flags, borderValue=0), (east0, north0))


def _common_shift(meta_a, grey_a, valid_a, meta_b, grey_b, valid_b) -> np.ndarray:
    """Shift (east, north) of the scene from A to B by phase correlation of the overlapping area."""
    import cv2

    east = [max(meta_a["bounds_m"]["east"][0], meta_b["bounds_m"]["east"][0]),
            min(meta_a["bounds_m"]["east"][1], meta_b["bounds_m"]["east"][1])]
    north = [max(meta_a["bounds_m"]["north"][0], meta_b["bounds_m"]["north"][0]),
             min(meta_a["bounds_m"]["north"][1], meta_b["bounds_m"]["north"][1])]
    span = min(east[1] - east[0], north[1] - north[0])
    if span <= 0:
        return np.zeros(2)
    centre = ((east[0] + east[1]) / 2, (north[0] + north[1]) / 2)
    resolution = max(span / 768, float(meta_a["resolution_m_per_px"]), float(meta_b["resolution_m_per_px"]))
    a, mask_a, _ = _window(meta_a, grey_a, valid_a, centre, span / 2, resolution)
    b, mask_b, _ = _window(meta_b, grey_b, valid_b, centre, span / 2, resolution)
    both = (mask_a > 0.99) & (mask_b > 0.99)
    if both.mean() < 0.2:
        return np.zeros(2)
    a = np.where(both, a - a[both].mean(), 0).astype(np.float32)
    b = np.where(both, b - b[both].mean(), 0).astype(np.float32)
    (dx, dy), response = cv2.phaseCorrelate(a, b, cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F))
    if response < 0.03 or math.hypot(dx, dy) > 0.45 * a.shape[0]:
        return np.zeros(2)
    return np.array([dx * resolution, -dy * resolution])


def _match(template, mask, search, search_valid, angles):
    """Best (score, angle, dx, dy) of the rotated template inside ``search``; dx, dy in pixels from the centre.

    The score is the masked zero-mean normalised cross-correlation.  Returns also the second peak
    (outside the neighbourhood of the best one) to judge ambiguity.
    """
    import cv2

    size = template.shape[0]
    centre = ((size - 1) / 2, (size - 1) / 2)
    best = (-2.0, 0.0, 0.0, 0.0, -2.0)
    for angle in angles:
        # OpenCV angle is counter-clockwise on screen; ours is clockwise seen from above (= on a north-up map).
        rotation = cv2.getRotationMatrix2D(centre, -float(angle), 1.0)
        t = cv2.warpAffine(template, rotation, (size, size), flags=cv2.INTER_LINEAR)
        m = cv2.warpAffine(mask, rotation, (size, size), flags=cv2.INTER_LINEAR)
        m = (m > 0.5).astype(np.float32)
        if m.sum() < 16:
            continue
        response = cv2.matchTemplate(search, t, cv2.TM_CCOEFF_NORMED, mask=m)
        response = np.nan_to_num(response, nan=-1.0, posinf=-1.0, neginf=-1.0)
        response[np.abs(response) > 1.0001] = -1.0
        # positions where the template would hang over missing data do not count
        coverage = cv2.matchTemplate(search_valid, m, cv2.TM_CCORR) / m.sum()
        response[coverage < 0.97] = -1.0
        y, x = np.unravel_index(int(response.argmax()), response.shape)
        score = float(response[y, x])
        if score > best[0]:
            # sub-pixel peak by parabolas
            ox = oy = 0.0
            if 0 < x < response.shape[1] - 1:
                l, c, r = response[y, x - 1], score, response[y, x + 1]
                if min(l, r) > -1 and c - 0.5 * (l + r) > 1e-9:
                    ox = 0.25 * (l - r) / (0.5 * (l + r) - c)
            if 0 < y < response.shape[0] - 1:
                u, c, d = response[y - 1, x], score, response[y + 1, x]
                if min(u, d) > -1 and c - 0.5 * (u + d) > 1e-9:
                    oy = 0.25 * (u - d) / (0.5 * (u + d) - c)
            other = response.copy()
            radius = max(3, size // 3)
            other[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1] = -1.0
            half = (response.shape[1] - 1) / 2, (response.shape[0] - 1) / 2
            best = (score, float(angle), x + float(np.clip(ox, -1, 1)) - half[0], y + float(np.clip(oy, -1, 1)) - half[1],
                    float(other.max()))
    return best


def _template(meta, grey, valid, polygon, target_px: int, minimum_resolution: float, closed: bool = True):
    """Window around the shape with its mask, the working resolution and the centre.

    Closed contour: its interior plus a water rim.  Open polyline: a band along the line (the line usually
    follows an edge of a slick or of foam, the band holds both sides of that edge).
    """
    import cv2

    lower, upper = polygon.min(axis=0), polygon.max(axis=0)
    centre = (lower + upper) / 2
    extent = float((upper - lower).max())
    resolution = max(extent / target_px, minimum_resolution)
    rim = max(0.12 * extent, 2.5 * resolution)             # water rim: the outline carries the shape
    half = 0.5 * math.hypot(*(upper - lower)) + rim         # room to rotate
    window, window_valid, (east0, north0) = _window(meta, grey, valid, centre, half, resolution)
    pixels = np.column_stack([(polygon[:, 0] - east0) / resolution, (north0 - polygon[:, 1]) / resolution])
    mask = np.zeros(window.shape, np.uint8)
    grow = max(1, int(round(rim / resolution)))
    if closed:
        cv2.fillPoly(mask, [np.round(pixels * 16).astype(np.int32)], 1, shift=4)
    else:
        cv2.polylines(mask, [np.round(pixels * 16).astype(np.int32)], False, 1, thickness=1, shift=4)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    mask = (mask > 0) & (window_valid > 0.99)
    return window, mask.astype(np.float32), resolution, centre, extent


def _line_centre(line: np.ndarray) -> np.ndarray:
    """Mean position along an open polyline (weighted by length, not by vertex count)."""
    length = np.linalg.norm(np.diff(line, axis=0), axis=1)
    if length.sum() <= 0:
        return line.mean(axis=0)
    return ((line[:-1] + line[1:]) / 2 * length[:, None]).sum(axis=0) / length.sum()


def _track_one(meta_a, grey_a, valid_a, meta_b, grey_b, valid_b, polygon, guess, wide_radius_m: float,
               closed: bool = True) -> dict:
    finest = max(float(meta_a["resolution_m_per_px"]), float(meta_b["resolution_m_per_px"]))
    area = _area_centroid(polygon)[0] if closed else 0.0
    template, mask, resolution, centre, extent = _template(meta_a, grey_a, valid_a, polygon, COARSE_TEMPLATE_PX, finest, closed)
    if mask.sum() < 25 or float(np.ptp(template[mask > 0])) < 4:
        raise ValueError("внутри контура нет деталей (или нет данных кадра)" if closed else
                         "вдоль ломаной нет деталей (или нет данных кадра)")

    def coarse(origin, radius):
        scale_limit = (2 * radius + template.shape[0] * resolution) / MAX_SEARCH_PX
        if scale_limit > resolution:      # very wide search: coarsen everything
            t, m, res, _, _ = _template(meta_a, grey_a, valid_a, polygon, COARSE_TEMPLATE_PX, scale_limit, closed)
        else:
            t, m, res = template, mask, resolution
        half = radius + t.shape[0] * res / 2
        search, search_valid, _ = _window(meta_b, grey_b, valid_b, centre + origin, half, res)
        score, angle, dx, dy, second = _match(t, m, search, search_valid, ANGLES_COARSE)
        return score, angle, origin + np.array([dx * res, -dy * res]), second, res

    near_radius = max(1.5 * extent, 8.0, 0.35 * float(np.hypot(*guess)))
    attempts = [coarse(np.asarray(guess, dtype=np.float64), near_radius)]
    if attempts[0][0] < 0.75 and wide_radius_m > near_radius:
        attempts.append(coarse(np.zeros(2), wide_radius_m))
    score, angle, shift, second, coarse_resolution = max(attempts, key=lambda item: item[0])
    if score <= 0:
        raise ValueError("контур не найден на втором кадре (вышел за кадр?)")

    # Refinement at a finer scale: ±3 coarse pixels, angles in two passes.
    template, mask, resolution, centre, extent = _template(meta_a, grey_a, valid_a, polygon, FINE_TEMPLATE_PX, finest, closed)
    radius = 3 * coarse_resolution + 2 * resolution
    half = radius + template.shape[0] * resolution / 2
    for step in (1.0, 0.2):
        search, search_valid, _ = _window(meta_b, grey_b, valid_b, centre + shift, half, resolution)
        span = 3.0 if step == 1.0 else 1.0
        fine = _match(template, mask, search, search_valid, angle + np.arange(-span, span + 1e-6, step))
        if fine[0] <= -1:
            break
        score, angle = fine[0], fine[1]
        shift = shift + np.array([fine[2] * resolution, -fine[3] * resolution])
        half = 4 * resolution + template.shape[0] * resolution / 2

    theta = math.radians(angle)                      # clockwise from above: (e, n) -> (e cos + n sin, -e sin + n cos)
    rotation = np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])
    moved = (polygon - centre) @ rotation.T + centre + shift
    centroid_a = _area_centroid(polygon)[1] if closed else _line_centre(polygon)
    centroid_b = _area_centroid(moved)[1] if closed else _line_centre(moved)
    vector = centroid_b - centroid_a
    quality = "good" if score >= 0.85 and score - second >= 0.15 else "acceptable" if score >= 0.75 and score - second >= 0.1 else "low"
    warning = ""
    if not closed:
        from .motion import _total_turn
        if _total_turn(polygon) < 8.0:      # aperture problem, as in motion.estimate_curve_displacement
            quality = "low"
            warning = "Ломаная почти прямая: смещение вдоль неё определяется неоднозначно; добавьте характерный изгиб"
    return {"east_m": round(float(vector[0]), 3), "north_m": round(float(vector[1]), 3),
            "displacement_m": round(float(np.hypot(*vector)), 3),
            "azimuth_deg": round((math.degrees(math.atan2(vector[0], vector[1])) + 360) % 360, 1),
            "rotation_deg": round(float(angle), 2), "manual": False, "score": round(float(score), 3),
            "second_peak": round(float(max(second, -1.0)), 3), "quality": quality, "warning": warning,
            "area_m2": round(area, 1), "length_m": round(float(np.linalg.norm(np.diff(polygon, axis=0), axis=1).sum()), 2), "extent_m": round(extent, 2), "resolution_m": round(float(resolution), 4),
            "centroid_a": np.round(centroid_a, 3).tolist(), "centroid_b": np.round(centroid_b, 3).tolist(),
            "points_b": np.round(moved, 3).tolist()}


def _ring(points) -> np.ndarray:
    polygon = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(polygon) >= 2 and np.allclose(polygon[0], polygon[-1]):
        polygon = polygon[:-1]
    if len(polygon) < 3 or not np.all(np.isfinite(polygon)):
        raise ValueError("контур должен содержать минимум три точки")
    return polygon


def _area_centroid(polygon: np.ndarray) -> tuple[float, np.ndarray]:
    x, y = polygon[:, 0], polygon[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    signed = 0.5 * float(cross.sum())
    if abs(signed) < 1e-9:
        return 0.0, polygon.mean(axis=0)
    centroid = np.array([((x + np.roll(x, -1)) * cross).sum(), ((y + np.roll(y, -1)) * cross).sum()]) / (6 * signed)
    return abs(signed), centroid


def _resample_ring(polygon: np.ndarray, count: int = 240) -> np.ndarray:
    closed = np.vstack([polygon, polygon[:1]])
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    samples = np.linspace(0, distance[-1], count, endpoint=False)
    return np.column_stack([np.interp(samples, distance, closed[:, axis]) for axis in range(2)])


def _rotation(angle_deg: float) -> np.ndarray:
    theta = math.radians(angle_deg)       # clockwise from above: north turns towards east
    return np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])


def match_drawn_contours(points_a, points_b) -> dict:
    """Displacement and rotation between two hand-drawn outlines of the same object (A earlier, B later).

    The shift is the difference of the area centroids.  The rotation minimises the symmetric distance
    between the outlines once the centroids coincide; it is reported only when it fits clearly better than
    no rotation (outlines of round objects carry no orientation).  ``points_b`` in the result is outline A
    carried by the estimated motion: drawn over frame B it should coincide with outline B.
    """
    a, b = _ring(points_a), _ring(points_b)
    area_a, centre_a = _area_centroid(a)
    area_b, centre_b = _area_centroid(b)
    if min(area_a, area_b) <= 0:
        raise ValueError("контур вырожден (нулевая площадь)")
    ring_a, ring_b = _resample_ring(a) - centre_a, _resample_ring(b) - centre_b

    def cost(angle: float) -> float:
        moved = ring_a @ _rotation(angle).T
        distance = np.linalg.norm(moved[:, None, :] - ring_b[None], axis=2)
        return float(np.sqrt(0.5 * ((distance.min(axis=1) ** 2).mean() + (distance.min(axis=0) ** 2).mean())))

    coarse = np.arange(-60.0, 60.01, 1.0)
    best = float(coarse[int(np.argmin([cost(angle) for angle in coarse]))])
    fine = best + np.arange(-1.0, 1.001, 0.1)
    best = float(fine[int(np.argmin([cost(angle) for angle in fine]))])
    unrotated, residual = cost(0.0), cost(best)
    rotation_found = unrotated - residual > 0.15 * unrotated
    if not rotation_found:
        best, residual = 0.0, unrotated
    vector = centre_b - centre_a
    extent = float(np.ptp(a, axis=0).max())
    ratio = area_b / area_a
    relative = residual / max(extent, 1e-6)
    quality = ("good" if relative <= 0.04 and 0.85 <= ratio <= 1.18 else
               "acceptable" if relative <= 0.10 and 0.7 <= ratio <= 1.4 else "low")
    moved = (a - centre_a) @ _rotation(best).T + centre_b
    return {"east_m": round(float(vector[0]), 3), "north_m": round(float(vector[1]), 3),
            "displacement_m": round(float(np.hypot(*vector)), 3),
            "azimuth_deg": round((math.degrees(math.atan2(vector[0], vector[1])) + 360) % 360, 1),
            "rotation_deg": round(best, 2), "rotation_found": bool(rotation_found), "score": None,
            "residual_m": round(residual, 3), "quality": quality, "manual": True,
            "area_m2": round(area_a, 1), "area_b_m2": round(area_b, 1), "extent_m": round(extent, 2),
            "centroid_a": np.round(centre_a, 3).tolist(), "centroid_b": np.round(centre_b, 3).tolist(),
            "points_b": np.round(moved, 3).tolist()}


def track_contours(path_a: str, path_b: str, shapes: list[dict], max_speed_m_s: float = 1.5) -> dict:
    """Follow closed contours drawn on raster A (ENU metres) to raster B.

    ``shapes`` = [{"slot": id, "points": [[east, north], ...], "type": "polygon" | "polyline"}, ...].
    An open ``polyline`` is searched for by the image band along it.  A contour that also has ``points_b`` (the
    same object outlined by hand on B) is not searched for: its motion comes from the two outlines
    (``match_drawn_contours``).  Each result has the displacement of the
    contour's centroid, its rotation (clockwise from above), speed, the matching score and ``points_b`` —
    the contour carried to B, to be drawn over B for a visual check.  Failures are reported per contour.
    """
    meta_a, meta_b = load_georaster(path_a), load_georaster(path_b)
    times_a, times_b = meta_a.get("times_s") or [], meta_b.get("times_s") or []
    if not times_a or not times_b:
        raise ValueError("Для изображений не сохранено время кадров; постройте сравнение заново")
    dt = float(np.mean(times_b) - np.mean(times_a))
    if abs(dt) < 1e-6:
        raise ValueError("Оба изображения относятся к одному моменту времени")
    common = np.zeros(2)
    if any(not shape.get("points_b") for shape in shapes):       # rasters are needed only for the search
        grey_a, valid_a = _grey(load_data(meta_a))
        if meta_a["data"] == meta_b["data"]:
            grey_b, valid_b = grey_a, valid_a
        else:
            grey_b, valid_b = _grey(load_data(meta_b))
        common = _common_shift(meta_a, grey_a, valid_a, meta_b, grey_b, valid_b)
    wide = float(np.clip(max_speed_m_s * abs(dt), 10.0, 600.0))
    results = []
    for shape in shapes:
        entry = {"slot": shape.get("slot")}
        try:
            if shape.get("type") == "polyline":       # open line: found on B by the image band along it
                line = np.asarray(shape["points"], dtype=np.float64).reshape(-1, 2)
                if len(line) < 2 or not np.all(np.isfinite(line)) or np.ptp(line, axis=0).max() <= 0:
                    raise ValueError("ломаная должна содержать минимум две разные точки")
                entry.update(_track_one(meta_a, grey_a, valid_a, meta_b, grey_b, valid_b, line, common, wide, closed=False))
            elif shape.get("points_b"):
                entry.update(match_drawn_contours(_ring(shape["points"]), shape["points_b"]))
            else:
                entry.update(_track_one(meta_a, grey_a, valid_a, meta_b, grey_b, valid_b, _ring(shape["points"]),
                                        common, wide))
            entry.update(delta_time_s=round(dt, 3), speed_m_s=round(entry["displacement_m"] / abs(dt), 4),
                         rotation_deg_per_min=round(entry["rotation_deg"] / abs(dt) * 60, 3))
        except ValueError as exc:
            entry["error"] = str(exc)
        results.append(entry)
    return {"results": results, "delta_time_s": round(dt, 3),
            "common_shift_m": np.round(common, 2).tolist(), "path_a": meta_a["image"], "path_b": meta_b["image"],
            "method": "per-contour masked NCC, translation + rotation"}


def export_motion(path_a: str, path_b: str, items: list[dict]) -> dict:
    """Write the displacement vectors of a frame pair to ``<project>/motion/*.txt`` (tab separated).

    ``items`` = [{"name", "type": "polygon" | "polyline", "method", "points": shape on the earlier frame
    (ENU metres), "east_m", "north_m": displacement to the later
    frame, "rotation_deg", "quality"}, ...].  Area, length, velocity and geographic coordinates are derived
    here; the time comes from the frame times stored with the rasters and the SRT clock of the video.
    """
    from datetime import datetime, timedelta, timezone
    from pathlib import Path

    from .georaster import enu_to_lonlat
    from .naming import flight_start_utc, output_stem
    from .pose import load_context

    meta_a, meta_b = load_georaster(path_a), load_georaster(path_b)
    if not (meta_a.get("times_s") and meta_b.get("times_s") and meta_a.get("video")):
        raise ValueError("Для изображений не сохранено время кадров; постройте сравнение заново")
    if float(np.mean(meta_b["times_s"])) < float(np.mean(meta_a["times_s"])):
        meta_a, meta_b = meta_b, meta_a
    time_a, time_b = float(np.mean(meta_a["times_s"])), float(np.mean(meta_b["times_s"]))
    duration = time_b - time_a
    if duration <= 0:
        raise ValueError("Оба изображения относятся к одному моменту времени")
    if not items:
        raise ValueError("Нет рассчитанных векторов: сначала нажмите «Рассчитать смещение и скорость»")
    context = load_context(meta_a["video"])
    # The SRT clock is the camera's local time; the FlightRecord (UTC) differs from it by whole hours.
    camera = datetime.fromtimestamp(context.start_epoch + time_a, timezone.utc).replace(tzinfo=None)
    utc = camera - timedelta(hours=context.utc_offset_hours)
    stamp = lambda value: value.isoformat(sep=" ", timespec="milliseconds")

    def number(value, digits):
        return "NaN" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"

    rows = []
    for item in items:
        points = np.asarray(item["points"], dtype=np.float64).reshape(-1, 2)
        closed = item.get("type") == "polygon"
        if closed:
            ring = _ring(points)
            area, centre = _area_centroid(ring)
            length = float(np.linalg.norm(np.diff(np.vstack([ring, ring[:1]]), axis=0), axis=1).sum())
        else:
            area, length = float("nan"), float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            centre = _line_centre(points)
        east, north = float(item["east_m"]), float(item["north_m"])
        lon, lat = enu_to_lonlat(meta_a, centre[:1], centre[1:])
        rotation = item.get("rotation_deg")
        rows.append("\t".join([
            str(item.get("name") or "object"), "contour" if closed else "polyline", str(item.get("method") or ""),
            stamp(utc), stamp(camera), f"{duration:.3f}",
            f"{centre[0]:.2f}", f"{centre[1]:.2f}", f"{float(lat[0]):.8f}", f"{float(lon[0]):.8f}",
            number(area, 1), f"{length:.2f}", f"{east / duration:.4f}", f"{north / duration:.4f}",
            f"{math.hypot(east, north) / duration:.4f}", f"{(math.degrees(math.atan2(east, north)) + 360) % 360:.1f}",
            f"{east:.3f}", f"{north:.3f}", f"{math.hypot(east, north):.3f}",
            number(None if rotation is None else float(rotation), 2), str(item.get("quality") or "")]))
    header = [
        "# DroneVision — векторы смещения (столбцы разделены табуляцией)",
        f"# Создано (UTC):\t{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"# Видео:\t{meta_a['video']}",
        f"# Frames_s — моменты кадров от начала видео, с:\t{time_a:.3f}\t{time_b:.3f}",
        f"# Изображения:\t{meta_a['image']}\t{meta_b['image']}",
        f"# Точка отсчёта ENU (место взлёта):\tlat {meta_a['reference']['latitude']:.8f}\tlon {meta_a['reference']['longitude']:.8f}",
        f"# CRS координат E/N:\t{meta_a['reference']['crs']}",
        f"# Поправка курса камеры, deg:\t{float(meta_a.get('yaw_offset_deg') or 0.0):+.2f}",
        f"# Начало полёта (UTC, первая запись FlightRecord):\t{flight_start_utc(context).isoformat(timespec='seconds')}"
        f"\t{Path(context.log_path).name}",
        f"# Сдвиг часов лога относительно видео:\t{context.utc_offset_hours:+g} h\t{context.clock_offset_seconds:+.2f} s",
        "# DateTime_UTC — момент первого кадра по часам FlightRecord (время камеры минус "
        f"{context.utc_offset_hours:+g} h); DateTime_camera — тот же момент по часам SRT видео",
        "# Duration_s — интервал между кадрами; velocity = displacement / Duration_s",
        "# Center_* — центр контура (центроид площади) или ломаной (середина по её длине) на первом кадре",
        "# Area_m2 — площадь внутри контура (NaN для ломаной); Length_m — периметр контура или длина ломаной",
        "# Ve_m_s, Vn_m_s — восточная и северная компоненты velocity; Azimuth_deg — направление движения по часовой стрелке от севера",
        "# dE_m, dN_m — компоненты displacement; Displacement_m — полный модуль смещения между кадрами",
        "# Rotation_deg — поворот объекта между кадрами по часовой стрелке при взгляде сверху (NaN — не определён)",
        "# Method: manual — объект отмечен вручную на обоих кадрах; auto — найден на втором кадре по изображению",
        "# Quality — оценка надёжности сопоставления: good / acceptable / low",
        "",
        "Name\tType\tMethod\tDateTime_UTC\tDateTime_camera\tDuration_s\tCenter_E_m\tCenter_N_m\tCenter_Lat\tCenter_Lon\t"
        "Area_m2\tLength_m\tVe_m_s\tVn_m_s\tSpeed_m_s\tAzimuth_deg\tdE_m\tdN_m\tDisplacement_m\tRotation_deg\tQuality",
    ]
    text = "\n".join(header + rows) + "\n"
    image = Path(meta_a["image"])
    output_dir = image.parent.parent / "motion"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{output_stem(output_dir, context, time_a, time_b, 'motion', ('.txt',))}.txt"
    output.write_text(text, encoding="utf-8")
    return {"path": str(output), "filename": output.name, "text": text, "count": len(rows)}
