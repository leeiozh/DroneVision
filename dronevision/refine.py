"""Frame-to-frame camera attitude from the images themselves.

Why: FlightRecord attitude is good only on average.  Measured on DJI_0371 (SIFT matches between frames,
see ``attitude_check``): angles are quantised to 0.1° (the log "jumps" while the picture stands still),
body yaw over-reports turns by ~3 % (6° after 200° of turns), gimbal pitch drifts ~2° per 100 s and roll
by ±1.5°.  The log time sync and the focal length were confirmed (best shift 0.00 s, f = 2940 px at 24 mm).
At 1.5–2 km on the sea surface 0.1° of pitch is ~20 m, so projected islands wander between frames.

How: features are matched between frames; a matched pixel of frame ``a`` is cast onto the sea plane
with pose ``a`` and reprojected with pose ``b`` (directions above the horizon or beyond 20 km are
transferred as directions).  Small additive corrections of yaw/pitch/roll per frame minimise all
reprojection errors jointly.  Their mean is held at zero, so the absolute georeference stays that of
the log plus the user's heading offset, while the frames become mutually consistent (~0.01°).
"""
from __future__ import annotations

import numpy as np

from .geometry import camera_basis, pixel_rays

FEATURE_WIDTH = 1600          # features are detected on a downscaled copy; points are in full-resolution px
MAX_FEATURES = 4000
MAX_POINTS_PER_EDGE = 500
MIN_INLIERS = 40
FAR_M = 20_000.0              # sea points farther than this are treated as directions
INLIER_PX = 4.0               # full-resolution pixels


def frame_features(cv2, frame: np.ndarray, valid_rows: int | None = None):
    """SIFT keypoints (full-resolution pixel coordinates) and descriptors of one BGR frame."""
    height, width = frame.shape[:2]
    scale = min(1.0, FEATURE_WIDTH / width)
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # Haze and glare leave distant islands at low contrast; local equalisation keeps their features.
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    mask = np.zeros(gray.shape, np.uint8)
    rows = gray.shape[0] if valid_rows is None else int(np.clip(round(valid_rows * scale), 0, gray.shape[0]))
    mask[8:max(rows - 8, 8), 8:-8] = 255
    keypoints, descriptors = cv2.SIFT_create(MAX_FEATURES).detectAndCompute(gray, mask)
    points = (np.float64([k.pt for k in keypoints]).reshape(-1, 2) + 0.5) / scale - 0.5
    return points, descriptors


def _pose(poses: dict, i: int, correction=(0.0, 0.0, 0.0)) -> dict:
    return {"yaw": float(poses["yaw"][i]) + correction[0], "pitch": float(poses["pitch"][i]) + correction[1],
            "roll": float(poses["roll"][i]) + correction[2], "focal_px": float(poses["focal_px"][i]),
            "altitude": float(poses["altitude"][i]), "offset": np.asarray(poses["offset"][i], dtype=np.float64)}


def transfer(points: np.ndarray, pose_a: dict, pose_b: dict, width: int, height: int) -> np.ndarray:
    """Pixels of frame ``a`` -> pixels of frame ``b`` through the sea plane (or as directions)."""
    basis_a = camera_basis(pose_a["yaw"], pose_a["pitch"], pose_a["roll"])
    rays = pixel_rays(points[:, 0], points[:, 1], width, height, pose_a["focal_px"], basis_a)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = pose_a["altitude"] / -rays[:, 2]
    ground = rays[:, :2] * scale[:, None]
    sea = (rays[:, 2] < -1e-6) & (np.hypot(ground[:, 0], ground[:, 1]) < FAR_M)
    vectors = rays.copy()
    vectors[sea, :2] = ground[sea] + pose_a["offset"] - pose_b["offset"]
    vectors[sea, 2] = -pose_b["altitude"]
    forward, right, down = camera_basis(pose_b["yaw"], pose_b["pitch"], pose_b["roll"])
    depth = vectors @ forward
    with np.errstate(divide="ignore", invalid="ignore"):
        u = width / 2 + pose_b["focal_px"] * (vectors @ right) / depth
        v = height / 2 + pose_b["focal_px"] * (vectors @ down) / depth
    out = np.column_stack([u, v])
    out[depth <= 0] = np.nan
    return out


def _residual(points_a, points_b, pose_a, pose_b, width, height) -> np.ndarray:
    return transfer(points_a, pose_a, pose_b, width, height) - points_b


def _weights(residual: np.ndarray, scale: float) -> np.ndarray:
    """Cauchy weights per point (both coordinates share the point's weight)."""
    norm = np.hypot(residual[:, 0], residual[:, 1])
    weight = np.where(np.isfinite(norm), 1.0 / (1.0 + (norm / scale) ** 2), 0.0)
    return np.repeat(weight, 2)


def _jacobian(function, x: np.ndarray, step: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    r0 = function(x).ravel()
    columns = [(function(x + step * e).ravel() - r0) / step for e in np.eye(len(x))]
    return r0, np.column_stack(columns)


def match_edge(cv2, features_a, features_b, pose_a: dict, pose_b: dict, width: int, height: int):
    """Matches between two frames that are consistent with a small attitude change of frame ``b``.

    Returns (points_a, points_b, median residual with the logged poses) or None. Graduated robust scales
    (60 -> 2 px) tolerate log errors of a couple of degrees and reject moving water and glitter.
    """
    (pa, da), (pb, db) = features_a, features_b
    if da is None or db is None or len(pa) < MIN_INLIERS or len(pb) < MIN_INLIERS:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m[0] for m in pairs if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < MIN_INLIERS:
        return None
    p1 = pa[[m.queryIdx for m in good]]
    p2 = pb[[m.trainIdx for m in good]]
    before = _residual(p1, p2, pose_a, pose_b, width, height)

    def residual(x):
        shifted = dict(pose_b, yaw=pose_b["yaw"] + x[0], pitch=pose_b["pitch"] + x[1], roll=pose_b["roll"] + x[2])
        return _residual(p1, p2, pose_a, shifted, width, height)

    x = np.zeros(3)
    # Coarse start: the dominant image shift, to survive log errors of several degrees.
    finite = np.all(np.isfinite(before), axis=1)
    if finite.sum() < MIN_INLIERS:
        return None
    for scale in (60.0, 20.0, 6.0, 2.0):
        for _ in range(4):
            r0, jac = _jacobian(lambda v: residual(v), x)
            w = _weights(r0.reshape(-1, 2), scale)
            ok = np.isfinite(r0) & np.all(np.isfinite(jac), axis=1)
            if ok.sum() < 2 * MIN_INLIERS:
                return None
            sw = np.sqrt(w[ok])
            dx = np.linalg.lstsq(jac[ok] * sw[:, None], -r0[ok] * sw, rcond=None)[0]
            x += dx
            if np.abs(dx).max() < 1e-4:
                break
    error = np.hypot(*residual(x).T)
    inliers = np.isfinite(error) & (error < INLIER_PX)
    if inliers.sum() < MIN_INLIERS:
        return None
    index = np.flatnonzero(inliers)
    if len(index) > MAX_POINTS_PER_EDGE:
        # Keep a spatially spread subset: sort by image row so all distances stay represented.
        index = index[np.argsort(p1[index, 1])][np.linspace(0, len(index) - 1, MAX_POINTS_PER_EDGE).astype(int)]
    log_error = np.hypot(*before[index].T)
    return p1[index], p2[index], float(np.nanmedian(log_error))


def refine_attitudes(cv2, features: list, poses: dict, width: int, height: int) -> tuple[np.ndarray, dict]:
    """Per-frame (yaw, pitch, roll) corrections in degrees and a report.

    Edges join consecutive frames, frames two apart and every frame with the middle one. Frames that do
    not connect to the largest group keep the logged attitude.
    """
    count = len(features)
    corrections = np.zeros((count, 3))
    report = {"method": "image_feature_attitude_refinement", "frames": count, "edges": 0,
              "connected_frames": 1 if count else 0, "median_log_error_px": None, "median_residual_px": None,
              "corrections_deg": corrections.round(3).tolist(), "note": ""}
    if count < 2:
        return corrections, report
    hub = count // 2
    candidates = sorted({tuple(sorted(e)) for i in range(count)
                         for e in ((i, i + 1), (i, i + 2), (i, hub)) if e[1] < count and e[0] != e[1]})
    base = [_pose(poses, i) for i in range(count)]
    edges = []
    for a, b in candidates:
        matched = match_edge(cv2, features[a], features[b], base[a], base[b], width, height)
        if matched is not None:
            edges.append((a, b, *matched))
    if not edges:
        report["note"] = "Нет общих неподвижных деталей между кадрами: оставлена ориентация из лога"
        return corrections, report

    # Largest connected group of frames.
    parent = list(range(count))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, *_ in edges:
        parent[root(a)] = root(b)
    groups: dict[int, list[int]] = {}
    for i in range(count):
        groups.setdefault(root(i), []).append(i)
    group = max(groups.values(), key=len)
    if len(group) < 2:
        report["note"] = "Кадры не связаны общими деталями: оставлена ориентация из лога"
        return corrections, report
    column = {frame: k for k, frame in enumerate(group)}
    edges = [edge for edge in edges if edge[0] in column and edge[1] in column]
    size = 3 * len(group)
    x = np.zeros(size)
    # Gauge: mean correction of the group is zero (the log is right on average, the heading offset aside).
    gauge = np.zeros((3, size))
    for k in range(len(group)):
        gauge[:, 3 * k:3 * k + 3] = np.eye(3) / len(group)

    def edge_residual(edge, params):
        a, b, p1, p2, _ = edge
        ia, ib = column[a], column[b]
        return _residual(p1, p2, _pose(poses, a, params[3 * ia:3 * ia + 3]), _pose(poses, b, params[3 * ib:3 * ib + 3]),
                         width, height)

    for scale in (4.0, 2.0, 2.0, 2.0):
        normal = 1e6 * gauge.T @ gauge + 1e-6 * np.eye(size)
        gradient = 1e6 * gauge.T @ (gauge @ x)
        for edge in edges:
            ia, ib = column[edge[0]], column[edge[1]]
            local = np.r_[x[3 * ia:3 * ia + 3], x[3 * ib:3 * ib + 3]]

            def residual(v, edge=edge, ia=ia, ib=ib):
                params = x.copy()
                params[3 * ia:3 * ia + 3], params[3 * ib:3 * ib + 3] = v[:3], v[3:]
                return edge_residual(edge, params)

            r0, jac = _jacobian(residual, local)
            w = _weights(r0.reshape(-1, 2), scale)
            ok = np.isfinite(r0) & np.all(np.isfinite(jac), axis=1)
            jw = jac[ok] * w[ok, None]
            index = np.r_[3 * ia:3 * ia + 3, 3 * ib:3 * ib + 3]
            normal[np.ix_(index, index)] += jw.T @ jac[ok]
            gradient[index] += jw.T @ r0[ok]
        step = np.linalg.solve(normal, -gradient)
        x += step
        if np.abs(step).max() < 1e-4:
            break

    residuals = [np.hypot(*edge_residual(edge, x).T) for edge in edges]
    for frame, k in column.items():
        corrections[frame] = x[3 * k:3 * k + 3]
    unconnected = count - len(group)
    report.update({
        "edges": len(edges), "connected_frames": len(group),
        "median_log_error_px": round(float(np.median([edge[4] for edge in edges])), 2),
        "median_residual_px": round(float(np.median(np.concatenate(residuals))), 2),
        "corrections_deg": corrections.round(3).tolist(),
        "max_correction_deg": round(float(np.abs(corrections).max()), 3),
        "note": f"{unconnected} кадр(ов) без общих деталей оставлены по логу" if unconnected else "",
    })
    return corrections, report


def apply_corrections(poses: dict, corrections: np.ndarray) -> dict:
    """Poses with the corrections added to yaw, pitch and roll (the log values stay in ``log_yaw`` etc.)."""
    corrected = dict(poses)
    corrected["yaw"] = (np.asarray(poses["yaw"]) + corrections[:, 0]) % 360
    corrected["pitch"] = np.asarray(poses["pitch"]) + corrections[:, 1]
    corrected["roll"] = np.asarray(poses["roll"]) + corrections[:, 2]
    corrected["attitude_correction"] = corrections
    return corrected
