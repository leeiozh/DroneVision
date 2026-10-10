"""Camera model shared by the single-frame projection and the mosaic.

Conventions (checked on DJI_0371, Mavic 2 Zoom, against OSM waterlines and image motion,
see docs/projection.md):

* ENU metres: x east, y north, z up; the sea surface is z = -altitude below the camera.
* yaw is the aircraft heading ``osd.yaw`` (clockwise from true north). ``gimbal.yaw`` in
  FlightRecord has an arbitrary reference (DJI_0371: body - 48.5 deg) and must not be used.
* pitch and roll are the gimbal attitude in the earth frame (``gimbal.pitch``, ``gimbal.roll``),
  pitch negative below the horizon. They are not added to the body attitude.
* rotation order yaw -> pitch -> roll (intrinsic Z-Y-X), pinhole without distortion.
"""
from __future__ import annotations

import numpy as np

# 35 mm-equivalent width of the recorded 16:9 video frame: focal_px = width * focal_mm / value.
CAMERA_PROFILES = {
    # f = 2950 px at 24 mm for 3840 px (image motion + coastline fit, DJI_0371) -> HFOV 66.1 deg.
    "mavic2_zoom": {"label": "Mavic 2 Zoom", "base_focal_mm": 24.0, "frame_width_35mm": 31.24, "calibrated": True},
    # 1" sensor, full-width 4K, 28 mm lens: nominal HFOV 65.5 deg, not yet checked on real control data.
    "mavic2_pro": {"label": "Mavic 2 Pro", "base_focal_mm": 28.0, "frame_width_35mm": 36.0, "calibrated": False},
}


MIN_AUTO_RESOLUTION_M = 0.01


def camera_profile(focal_mm_values) -> str:
    """Guess the camera from SRT focal lengths: Mavic 2 Pro records a constant 28 mm."""
    values = np.asarray(focal_mm_values, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) and np.all(np.abs(values - 28.0) < 0.05):
        return "mavic2_pro"
    return "mavic2_zoom"


def focal_length_px(width_px: int, focal_mm: float, profile: str = "mavic2_zoom",
                    base_horizontal_fov_deg: float = 0.0) -> float:
    """Pinhole focal length in pixels for an SRT 35 mm-equivalent focal length.

    ``base_horizontal_fov_deg`` overrides the profile: horizontal FOV without zoom (at base focal).
    """
    spec = CAMERA_PROFILES[profile]
    if base_horizontal_fov_deg and base_horizontal_fov_deg > 0:
        base_px = width_px / (2 * np.tan(np.deg2rad(base_horizontal_fov_deg) / 2))
        return float(base_px * focal_mm / spec["base_focal_mm"])
    return float(width_px * focal_mm / spec["frame_width_35mm"])


def horizontal_fov_deg(width_px: int, focal_px: float) -> float:
    return float(np.rad2deg(2 * np.arctan(width_px / (2 * focal_px))))


def camera_basis(yaw_deg: float, pitch_deg: float, roll_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward, image-right and image-down unit vectors in ENU."""
    yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    forward = np.array([np.cos(pitch) * np.sin(yaw), np.cos(pitch) * np.cos(yaw), np.sin(pitch)])
    right = np.array([np.cos(yaw), -np.sin(yaw), 0.0])
    down = np.cross(forward, right)
    return forward, np.cos(roll) * right + np.sin(roll) * down, -np.sin(roll) * right + np.cos(roll) * down


def pixel_rays(px, py, width: int, height: int, focal_px: float, basis) -> np.ndarray:
    forward, right, down = basis
    x = (np.asarray(px, dtype=np.float64) - width / 2) / focal_px
    y = (np.asarray(py, dtype=np.float64) - height / 2) / focal_px
    return forward + x[..., None] * right + y[..., None] * down


def ground_points(rays: np.ndarray, altitude_m: float) -> np.ndarray:
    """Intersection with the sea plane relative to the camera (east, north); NaN above the horizon."""
    rays = np.asarray(rays, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(rays[..., 2] < -1e-9, -altitude_m / rays[..., 2], np.nan)
    return rays[..., :2] * scale[..., None]


def project_to_image(east_north, altitude_m: float, width: int, height: int, focal_px: float, basis) -> tuple[np.ndarray, np.ndarray]:
    """Sea-level points (relative east, north) to pixels; second value is depth along the optical axis."""
    forward, right, down = basis
    points = np.asarray(east_north, dtype=np.float64)
    vectors = np.concatenate([points, np.full(points.shape[:-1] + (1,), -altitude_m)], axis=-1)
    depth = vectors @ forward
    with np.errstate(divide="ignore", invalid="ignore"):
        u = width / 2 + focal_px * (vectors @ right) / depth
        v = height / 2 + focal_px * (vectors @ down) / depth
    return np.stack([u, v], axis=-1), depth


def horizon_crop_row(width: int, height: int, focal_px: float, basis, altitude_m: float,
                     max_distance_m: float = 10_000.0, extra_rows: int = 0) -> int:
    """First image row whose left and right edge pixels hit the sea closer than ``max_distance_m``."""
    rows = np.arange(height)
    corners = ground_points(np.stack([pixel_rays(np.zeros(height), rows, width, height, focal_px, basis),
                                      pixel_rays(np.full(height, width - 1), rows, width, height, focal_px, basis)]),
                            altitude_m)
    distance = np.linalg.norm(corners, axis=-1).max(axis=0)
    valid = np.flatnonzero(np.isfinite(distance) & (distance <= max_distance_m))
    if not len(valid):
        raise ValueError("Камера не видит поверхность воды ближе допустимой дальности")
    return int(min(valid[0] + max(extra_rows, 0), height - 2))


def ground_corners(width: int, height: int, focal_px: float, basis, altitude_m: float, crop_row: int) -> np.ndarray:
    """Sea-level ENU offsets of the cropped frame corners: top-left, top-right, bottom-right, bottom-left."""
    px = np.array([0, width - 1, width - 1, 0], dtype=np.float64)
    py = np.array([crop_row, crop_row, height - 1, height - 1], dtype=np.float64)
    corners = ground_points(pixel_rays(px, py, width, height, focal_px, basis), altitude_m)
    if not np.all(np.isfinite(corners)):
        raise ValueError("Часть кадра смотрит выше горизонта; увеличьте отступ от горизонта")
    return corners


def finest_ground_resolution(width: int, height: int, focal_px: float, basis, altitude_m: float,
                             top_row: int = 0, bottom_row: int | None = None) -> float:
    """Smallest sea-surface size of one image pixel between two rows, m (geometric mean of both axes).

    This is the default raster resolution ("auto"): nothing the frame resolves is lost, and a low flight
    over a near object gets centimetres instead of a fixed 0.25 m.  Rounded up to two significant digits.
    """
    bottom = height - 1 if bottom_row is None else min(int(bottom_row), height - 1)
    px, py = np.meshgrid(np.linspace(0, width - 2, 5), np.linspace(min(top_row, bottom), bottom - 1, 9))
    origin, along_x, along_y = (ground_points(pixel_rays(px + dx, py + dy, width, height, focal_px, basis), altitude_m)
                                for dx, dy in ((0, 0), (1, 0), (0, 1)))
    a, b = along_x - origin, along_y - origin
    size = np.sqrt(np.abs(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]))
    size = size[np.isfinite(size) & (size > 0)]
    if not len(size):
        raise ValueError("Камера не видит поверхность воды")
    value = max(float(size.min()), MIN_AUTO_RESOLUTION_M)
    power = 10 ** (np.floor(np.log10(value)) - 1)
    return float(np.ceil(value / power - 1e-9) * power)


def pixel_ground_resolution(altitude_m: float, width: int, horizontal_fov_deg: float) -> float:
    return 2 * altitude_m * np.tan(np.deg2rad(horizontal_fov_deg) / 2) / width
