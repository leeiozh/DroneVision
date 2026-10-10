from __future__ import annotations

import math

import numpy as np


def _polyline(points) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or len(value) < 2 or not np.all(np.isfinite(value)):
        raise ValueError("Ломаная должна содержать минимум две корректные точки")
    lengths = np.linalg.norm(np.diff(value, axis=0), axis=1)
    keep = np.r_[True, lengths > 1e-6]
    value = value[keep]
    if len(value) < 2:
        raise ValueError("Длина ломаной равна нулю")
    return value


def _resample(points: np.ndarray, spacing: float) -> tuple[np.ndarray, float]:
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distance = np.r_[0.0, np.cumsum(segment)]
    total = float(distance[-1])
    # The same arc-length step on both curves: the index shift below is an arc-length shift only then.
    samples = np.arange(0.0, total + spacing * 1e-6, spacing)
    if total - samples[-1] > spacing * 0.25:
        samples = np.append(samples, total)
    return np.column_stack([np.interp(samples, distance, points[:, axis]) for axis in range(2)]), total


def _total_turn(points: np.ndarray) -> float:
    vectors = np.diff(points, axis=0)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    if len(vectors) < 2:
        return 0.0
    cosine = np.clip(np.sum(vectors[:-1] * vectors[1:], axis=1), -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)).sum())


def _closest_on_polyline(points: np.ndarray, line: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closest points on ``line``, unit normals there, and whether the foot is inside (not a line end)."""
    start, direction = line[:-1], np.diff(line, axis=0)
    length2 = np.maximum((direction ** 2).sum(axis=1), 1e-12)
    t = np.clip(((points[:, None, :] - start[None]) * direction[None]).sum(axis=2) / length2[None], 0.0, 1.0)
    feet = start[None] + t[..., None] * direction[None]
    distance = np.linalg.norm(points[:, None, :] - feet, axis=2)
    best = np.argmin(distance, axis=1)
    rows = np.arange(len(points))
    tangent = direction[best] / np.sqrt(length2[best])[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    tb = t[rows, best]
    inside = ~(((best == 0) & (tb <= 0.0)) | ((best == len(direction) - 1) & (tb >= 1.0)))
    return feet[rows, best], normal, inside


# How the translation of a drawn line is found (estimate_curve_displacement)
# ---------------------------------------------------------------------------
# The line is assumed to keep its shape and only shift between the two frames.  Two tracings of it
# never start and end at the same place, so points are not matched to points: every point of one line
# is compared with the *nearest place* on the other line, and only the distance across the line counts.
# A straight stretch therefore says nothing about a shift along itself (the aperture problem); bends do.
#
# Among all translations that put one line onto the other equally well, the shortest one is reported.
# For a nearly straight slick this is the displacement across it; for a line with clear bends the
# bends fix the along-line part as well.  An earlier version matched the lines by arc length first and
# kept that shift along straight lines: two tracings of one slick of different length then "moved" by
# hundreds of metres along the slick (DJI_0469: 1200 m instead of ~12 m).

WEAK_DIRECTION = 0.10      # eigenvalue ratio of the normals below which the along-line shift is not trusted
MIN_OVERLAP = 0.30         # share of the shorter line that must lie on the other one


def _across_gaps(points: np.ndarray, line: np.ndarray):
    """Signed distances of ``points`` across ``line``, the unit normals there and the "inside" mask."""
    feet, normal, inside = _closest_on_polyline(points, line)
    return ((points - feet) * normal).sum(axis=1), normal, inside


def _register(a: np.ndarray, b: np.ndarray, start, iterations: int = 60) -> tuple[np.ndarray, float]:
    """Point-to-line registration: the translation ``t`` that moves line A onto line B, from ``start``.

    Both directions are used (points of B against A shifted, points of A shifted against B), which keeps
    the result independent of which frame was traced more densely.  Where the normals of the lines span
    one direction only, the shift along the line stays at its start value: with ``start`` = 0 this gives
    the shortest translation.  Returns ``t`` and the eigenvalue ratio of the normals (0 = straight line).
    """
    t = np.asarray(start, dtype=np.float64).copy()
    prior = t.copy()
    ratio = 0.0
    for _ in range(iterations):
        gap_b, normal_b, inside_b = _across_gaps(b - t, a)          # B in the frame of A
        gap_a, normal_a, inside_a = _across_gaps(a + t, b)          # A in the frame of B
        gaps = np.r_[gap_b, -gap_a]
        normals = np.vstack([normal_b, normal_a])
        inside = np.r_[inside_b, inside_a]
        if inside.sum() < 4:
            break
        scale = max(1.4826 * float(np.median(np.abs(gaps[inside] - np.median(gaps[inside])))), 1e-3)
        centred = gaps - np.median(gaps[inside])
        w = np.where(inside, 1.0 / (1.0 + (centred / (3.0 * scale)) ** 2), 0.0)
        matrix = (normals * w[:, None]).T @ normals
        values, vectors = np.linalg.eigh(matrix)                    # ascending: vectors[:, 0] = along the line
        ratio = float(values[0] / max(values[1], 1e-12))
        # The weakly determined direction is held at the prior; no penalty across the line (no bias there).
        # Hand tracing puts the vertices of a gentle bend anywhere, so below WEAK_DIRECTION the bend is
        # not allowed to pull the line along itself (DJI_0469: it asked for 370 m).
        hold = w.sum() * max(0.0, 1.0 - ratio / WEAK_DIRECTION) ** 2
        along = vectors[:, 0]
        matrix = matrix + hold * np.outer(along, along)
        vector = (normals * (w * gaps)[:, None]).sum(axis=0) + hold * along * float(along @ (prior - t))
        step = np.linalg.solve(matrix + 1e-9 * np.eye(2), vector)
        t += step
        if np.hypot(*step) < 1e-4:
            break
    return t, ratio


def _misfit(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> tuple[float, float, np.ndarray]:
    """(robust rms distance between the lines after the shift, overlap share, distances of B's points)."""
    gap_b, _, inside_b = _across_gaps(b - t, a)
    gap_a, _, inside_a = _across_gaps(a + t, b)
    overlap = min(float(inside_b.mean()), 1.0) if len(a) >= len(b) else min(float(inside_a.mean()), 1.0)
    overlap = max(overlap, min(float(inside_b.mean()), float(inside_a.mean())))
    distances = np.abs(np.r_[gap_b[inside_b], gap_a[inside_a]])
    if len(distances) < 4:
        return float("inf"), 0.0, np.abs(gap_b)
    clipped = np.minimum(distances, np.percentile(distances, 90))
    return float(np.sqrt(np.mean(clipped ** 2))), overlap, np.abs(gap_b[inside_b])


def _arc_length_starts(a: np.ndarray, b_forward: np.ndarray, keep: int = 6) -> list[np.ndarray]:
    """Rough translations from matching the lines by arc length (both orders): starting points only.

    They matter for a strongly bent line that moved farther than the size of its bends, where the
    registration from zero would lock onto the wrong bend.
    """
    minimum = max(6, int(math.ceil(min(len(a), len(b_forward)) * MIN_OVERLAP)))
    found = []
    for b in (b_forward, b_forward[::-1]):
        for shift in range(-(len(b) - minimum), len(a) - minimum + 1):
            ia, ib = max(0, shift), max(0, -shift)
            count = min(len(a) - ia, len(b) - ib)
            if count < minimum:
                continue
            difference = b[ib:ib + count] - a[ia:ia + count]
            translation = np.median(difference, axis=0)
            errors = np.linalg.norm(difference - translation, axis=1)
            found.append((float(np.sqrt(np.mean(np.minimum(errors, np.percentile(errors, 90)) ** 2))), translation))
    found.sort(key=lambda item: item[0])
    starts: list[np.ndarray] = []
    for _, translation in found:
        if all(np.hypot(*(translation - other)) > 1e-6 + 0.05 * np.hypot(*other) for other in starts):
            starts.append(translation)
        if len(starts) >= keep:
            break
    return starts


def estimate_curve_displacement(points_a, points_b, delta_time_s: float) -> dict:
    """Translation that moves line A (earlier frame) onto line B, assuming the line keeps its shape.

    The lines may be traced with different lengths, vertex counts and directions.  Of all translations
    that align the shapes equally well the shortest is returned (see the notes above ``_register``).
    """
    dt = float(delta_time_s)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Разница времени между кадрами должна быть положительной")
    a0, b0 = _polyline(points_a), _polyline(points_b)
    length_a = float(np.linalg.norm(np.diff(a0, axis=0), axis=1).sum())
    length_b = float(np.linalg.norm(np.diff(b0, axis=0), axis=1).sum())
    shorter = min(length_a, length_b)
    if shorter < 2.0:
        raise ValueError("Обведите хотя бы 2 метра кривой для устойчивого сопоставления")
    spacing = max(shorter / 140.0, 0.25)
    a, _ = _resample(a0, spacing)
    b, _ = _resample(b0, spacing)
    # Candidates: the shortest translation (start at zero) and registrations started from arc-length matches.
    starts = [np.zeros(2)] + _arc_length_starts(a, b)
    candidates = []
    for start in starts:
        translation, ratio = _register(a, b, start)
        misfit, overlap, distances = _misfit(a, b, translation)
        if overlap >= MIN_OVERLAP and np.isfinite(misfit):
            candidates.append({"t": translation, "ratio": ratio, "misfit": misfit, "overlap": overlap,
                               "distances": distances, "from_zero": not np.any(start)})
    if not candidates:
        raise ValueError("Не удалось найти общий участок двух ломаных")
    # The shortest translation stands unless a longer one aligns the shapes clearly better over at least the
    # same stretch, and only for a line whose bends really fix the shift along it.  A short overlap of two
    # almost straight lines always "fits" and must not win (two tracings of one 185 m segment gave 125 m).
    score = lambda c: c["misfit"] / (0.65 + 0.35 * c["overlap"])
    shortest = next((c for c in candidates if c["from_zero"]), None)
    rivals = [c for c in candidates if not c["from_zero"] and c["ratio"] >= WEAK_DIRECTION]
    if shortest is not None:
        rivals = [c for c in rivals if score(c) < 0.6 * score(shortest) and c["overlap"] >= 0.8 * shortest["overlap"]]
    if rivals:
        best = min(score(c) for c in rivals)
        chosen = min((c for c in rivals if score(c) <= 1.25 * best + 0.25 * spacing), key=lambda c: float(np.hypot(*c["t"])))
    elif shortest is not None:
        chosen = shortest
    else:
        chosen = min(candidates, key=score)
    translation, residual = chosen["t"], chosen["misfit"]
    east, north = map(float, translation)
    displacement = float(math.hypot(east, north))
    overlap_length = float(chosen["overlap"] * shorter)
    relative_error = residual / max(overlap_length, spacing)
    shape_turn = min(_total_turn(a), _total_turn(b))
    along_determined = chosen["ratio"] >= WEAK_DIRECTION
    quality = ("good" if overlap_length >= 12 * spacing and relative_error <= 0.025 else
               "acceptable" if overlap_length >= 8 * spacing and relative_error <= 0.06 else "low")
    warning = ""
    if not along_determined:
        warning = ("Линия почти прямая: показано смещение поперёк линии; сдвиг вдоль неё по форме не определяется"
                   if chosen["ratio"] < 0.02 else
                   "Линия слабо изогнута: смещение вдоль неё определено неуверенно, поперёк — надёжно")
    if residual > 0.5 * displacement:
        # e.g. 21 m found from two tracings that differ by 16 m rms: the direction is right, the value is rough.
        warning += ("; " if warning else "") + (f"смещение ({displacement:.1f} м) сравнимо с расхождением самих линий "
                                                 f"({residual:.1f} м): обведите точнее или возьмите кадры дальше по времени")
        quality = "acceptable" if quality == "good" else quality
    azimuth = (math.degrees(math.atan2(east, north)) + 360.0) % 360.0
    uncertainty = float(np.percentile(chosen["distances"], 75)) if len(chosen["distances"]) else residual
    return {
        "delta_time_s": round(dt, 6),
        "east_m": round(east, 3), "north_m": round(north, 3),
        "displacement_m": round(displacement, 3),
        "speed_m_s": round(displacement / dt, 5),
        "speed_cm_s": round(displacement / dt * 100.0, 3),
        "azimuth_deg": round(azimuth, 2),
        "overlap_length_m": round(overlap_length, 2),
        "residual_m": round(residual, 3), "uncertainty_m": round(uncertainty, 3),
        "length_a_m": round(length_a, 2), "length_b_m": round(length_b, 2),
        "shape_turn_deg": round(shape_turn, 2), "quality": quality,
        "along_determined": bool(along_determined), "direction_ratio": round(chosen["ratio"], 4),
        "warning": warning,
        "method": "shortest translation aligning the shapes (symmetric point-to-line registration)",
    }
