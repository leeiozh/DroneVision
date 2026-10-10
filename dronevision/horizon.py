"""Camera pitch and roll from the sea horizon, and frame-to-frame drift from what is seen along it.

The sea horizon is the most accurate attitude reference in an oblique frame: its dip below the horizontal
depends only on the camera height (``coastdata.horizon_dip_deg``), and a line fitted over a thousand
columns is known to ~0.1 px, i.e. ~0.003° — against 0.1° quantisation and degree-level offsets of the
logged gimbal pitch.  Checked on DJI_0531 (2023-07-18): log −5.7°, horizon −7.0°; with the horizon value
the ground scale agrees with the wind-wave dispersion relation to 2 %.

The fit is a RANSAC line through per-column edge candidates with a 1.5 px tolerance: land on the horizon
has an irregular skyline and a waterline below the true horizon, so it does not gather inliers.
"""
from __future__ import annotations

import numpy as np

from .coastdata import horizon_dip_deg

COLUMN_STEP = 8
INLIER_PX = 1.5
MIN_INLIER_SHARE = 0.12


def expected_row(height: int, focal_px: float, pitch_deg: float, altitude_m: float) -> float:
    """Image row of the sea horizon at the centre column for a camera pitch (roll neglected)."""
    return height / 2 - focal_px * np.tan(np.deg2rad(-pitch_deg - horizon_dip_deg(altitude_m)))


def _candidates(cv2, gray: np.ndarray, top: int, bottom: int, per_column: int = 3):
    """Strongest vertical brightness steps of every sampled column: (x, sub-pixel row, strength)."""
    band = cv2.GaussianBlur(gray[top:bottom].astype(np.float32), (0, 0), sigmaX=6, sigmaY=1.2)
    step = np.abs(band[2:] - band[:-2])
    xs = np.arange(COLUMN_STEP // 2, gray.shape[1], COLUMN_STEP)
    columns = step[:, xs]
    found = []
    rows = np.arange(columns.shape[0])
    for k, x in enumerate(xs):
        c = columns[:, k]
        peak = (c[1:-1] >= c[:-2]) & (c[1:-1] > c[2:])
        index = rows[1:-1][peak]
        if not len(index):
            continue
        index = index[np.argsort(c[index])[::-1][:per_column]]
        for i in index:
            a, b, d = c[i - 1], c[i], c[i + 1]
            offset = 0.5 * (a - d) / (a - 2 * b + d) if a - 2 * b + d else 0.0
            found.append((x, top + 1 + i + offset, b))
    return np.asarray(found, dtype=np.float64).reshape(-1, 3), len(xs)


def detect(cv2, frame: np.ndarray, around_row: float | None = None, search_rows: float | None = None,
           seed: int = 0) -> dict | None:
    """Sea horizon line of a BGR frame: row at the centre column, slope, rms, share of columns on it.

    ``around_row`` ± ``search_rows`` limits the search (default: ±10 % of the height around the expected
    row, or the whole frame).  None when no straight edge collects enough columns.
    """
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if around_row is None:
        top, bottom = 2, height - 2
    else:
        reach = 0.10 * height if search_rows is None else float(search_rows)
        top, bottom = int(max(2, around_row - reach)), int(min(height - 2, around_row + reach))
    if bottom - top < 12:
        return None
    points, columns = _candidates(cv2, gray, top, bottom)
    if len(points) < 20:
        return None
    # Weak steps (haze banding, waves) only slow the search down.
    points = points[points[:, 2] >= max(1.0, 0.25 * np.percentile(points[:, 2], 90))]
    x, y = points[:, 0], points[:, 1]
    limit = np.tan(np.deg2rad(8.0))
    # First a strict tolerance: an open-sea horizon is straight to a fraction of a pixel, and a strict fit
    # cannot bridge it with the haze or land edge next to it.  A rough or hazy horizon needs the loose one.
    for tolerance in (0.5, INLIER_PX):
        rng = np.random.default_rng(seed)
        best, best_count = None, 0
        for _ in range(1500 if tolerance < 1 else 500):
            i, j = rng.integers(0, len(points), 2)
            if abs(x[i] - x[j]) < 0.08 * width:
                continue
            slope = (y[j] - y[i]) / (x[j] - x[i])
            if abs(slope) > limit:
                continue
            near = np.abs(y - (y[i] + slope * (x - x[i]))) < tolerance
            count = len(np.unique(x[near]))
            if count > best_count:
                best, best_count = near, count
        if best is None or best_count < max(12, MIN_INLIER_SHARE * columns):
            continue
        for _ in range(3):
            slope, intercept = np.polyfit(x[best], y[best], 1)
            residual = y - (intercept + slope * x)
            best = np.abs(residual) < tolerance
            if best.sum() < 12:
                break
        inlier_x = np.unique(x[best])
        if len(inlier_x) < max(12, MIN_INLIER_SHARE * columns) or np.ptp(inlier_x) < 0.12 * width:
            continue
        return {"row": float(intercept + slope * width / 2), "slope": float(slope),
                "rms_px": round(float(np.sqrt(np.mean(residual[best] ** 2))), 3),
                "columns_share": round(len(inlier_x) / columns, 3),
                "x_range": [float(inlier_x.min()), float(inlier_x.max())], "tolerance_px": tolerance}
    return None


def attitude(line: dict, height: int, focal_px: float, altitude_m: float) -> tuple[float, float]:
    """(pitch, roll) in degrees of the camera that sees the horizon on ``line`` (geometry.py conventions)."""
    roll = -np.arctan(line["slope"])
    distance = (height / 2 - line["row"]) * np.cos(roll)      # principal point to the line, + = line above
    pitch = -(np.rad2deg(np.arctan(distance / focal_px)) + horizon_dip_deg(altitude_m))
    return float(pitch), float(np.rad2deg(roll))


MAX_LOG_PITCH_ERROR_DEG = 3.0      # larger disagreement with the log means the line is not the horizon


def measure(cv2, frame: np.ndarray, focal_px: float, altitude_m: float, guess_pitch_deg: float | None = None) -> dict | None:
    """Horizon line plus the attitude it implies; the search is centred on the guessed (logged) pitch.

    A straight edge that would put the camera more than ``MAX_LOG_PITCH_ERROR_DEG`` away from the guess is
    rejected: with the camera looking down, the "horizon" would otherwise be a slick edge or a ship's wake.
    """
    height = frame.shape[0]
    around = expected_row(height, focal_px, guess_pitch_deg, altitude_m) if guess_pitch_deg is not None else None
    reach = None
    if around is not None:
        if not -0.02 * height < around < 1.02 * height:
            return None                                    # the horizon is outside the frame
        reach = focal_px * np.tan(np.deg2rad(MAX_LOG_PITCH_ERROR_DEG))
    line = detect(cv2, frame, around, reach)
    if line is None:
        return None
    pitch, roll = attitude(line, height, focal_px, altitude_m)
    if guess_pitch_deg is not None and (abs(pitch - guess_pitch_deg) > MAX_LOG_PITCH_ERROR_DEG or abs(roll) > 5.0):
        return None
    return {**line, "pitch_deg": round(pitch, 4), "roll_deg": round(roll, 4)}


class DriftTracker:
    """Frame-to-frame rotation of a nearly fixed camera from the strip along the horizon.

    Pitch and roll come from the horizon line of every frame.  Yaw comes from the horizontal shift of the
    strip around the horizon against the first frame — land and distant clouds are at infinity for this
    purpose.  A strip without texture (open sea under a clear sky) gives no yaw: ``yaw_tracked`` is False.
    """

    def __init__(self, cv2, frame: np.ndarray, focal_px: float, altitude_m: float, guess_pitch_deg: float):
        self.cv2, self.focal, self.altitude = cv2, float(focal_px), float(altitude_m)
        self.height, self.width = frame.shape[:2]
        self.first = measure(cv2, frame, focal_px, altitude_m, guess_pitch_deg)
        self.template = self.origin = None
        self.yaw_tracked = False
        self.last = self.first
        if self.first is None:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        row = int(round(self.first["row"]))
        top, bottom = max(row - 48, 0), min(row + 24, self.height)
        left, right = int(0.08 * self.width), int(0.92 * self.width)
        if bottom - top >= 24:
            self.template, self.origin = self._detail(gray[top:bottom, left:right]), (left, top)
            self.yaw_tracked = True          # decided per frame by the sharpness of the match

    def _detail(self, gray: np.ndarray) -> np.ndarray:
        """What varies along the horizon: the horizon edge itself is the same at any shift and is removed."""
        smooth = self.cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.5)
        return smooth - self.cv2.blur(smooth, (161, 1))

    def update(self, frame: np.ndarray) -> dict | None:
        """{"pitch_deg", "roll_deg", "yaw_shift_deg", "yaw_ok"} of this frame, None without a horizon."""
        if self.first is None:
            return None
        cv2 = self.cv2
        line = detect(cv2, frame, self.last["row"], 0.04 * self.height)
        if line is None:
            return None
        pitch, roll = attitude(line, self.height, self.focal, self.altitude)
        self.last = line
        result = {"pitch_deg": pitch, "roll_deg": roll, "yaw_shift_deg": 0.0, "yaw_ok": False}
        if self.yaw_tracked:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            left, top = self.origin
            reach_x, reach_y = int(0.05 * self.width), 40
            part = self.template.shape[1] // 3
            shifts, matches = [], []
            # Three parts of the strip separately: a shift of d px at x off the axis is a turn of
            # d·f / (f² + x²), and detail (an island, a cloud) usually sits in one part only.
            for n in range(3):
                piece = self.template[:, n * part:(n + 1) * part]
                x_left = left + n * part
                y0, x0 = max(top - reach_y, 0), max(x_left - reach_x, 0)
                window = self._detail(gray[y0:top + piece.shape[0] + reach_y, x0:x_left + part + reach_x])
                score = cv2.matchTemplate(window, piece, cv2.TM_CCOEFF_NORMED)
                _, peak, _, (j, i) = cv2.minMaxLoc(score)
                # Open sea under a clear sky has no detail along the horizon: the match is then weak and flat.
                side = max(float(score[i, max(j - 12, 0)]), float(score[i, min(j + 12, score.shape[1] - 1)]))
                if peak > 0.35 and peak - side > 0.1 and 0 < j < score.shape[1] - 1:
                    a, b, c = score[i, j - 1], score[i, j], score[i, j + 1]
                    shift = j + (0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c else 0.0) - (x_left - x0)
                    off_axis = x_left + part / 2 - self.width / 2
                    # Content moving right means the camera turned left.
                    shifts.append(-np.rad2deg(shift * self.focal / (self.focal ** 2 + off_axis ** 2)))
                    matches.append(float(peak))
            if shifts:
                result.update(yaw_shift_deg=float(np.median(shifts)), yaw_ok=True, match=round(max(matches), 3))
        return result
