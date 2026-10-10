"""«Улучшить»: re-render a part of an existing projection / mosaic / average at the native resolution.

The interactive rasters are limited to ~48 MP, so a 3–10 km footprint is drawn at 0.3–1.3 m/px while the 4K
frame near the camera resolves 0.1–0.2 m.  Here only the visible region is re-projected from the original
video frames with the stored poses (including the image-based attitude corrections), at the resolution the
frames actually support there, with Lanczos interpolation.  No detail is invented: this is resampling,
not learned super-resolution; for averages several sub-pixel-shifted frames add some real detail.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry import camera_basis, pixel_rays, project_to_image
from .georaster import load_georaster

MAX_PIXELS = 36_000_000
MAX_SIDE = 9000
FINEST_M = 0.03


def _native_resolution(poses: dict, i: int, point_enu: np.ndarray, width: int, height: int) -> float | None:
    """Ground size (geometric mean of both axes) of one image pixel of frame ``i`` at ``point_enu``."""
    basis = camera_basis(poses["yaw"][i], poses["pitch"][i], poses["roll"][i])
    relative = np.asarray(point_enu, dtype=np.float64) - poses["offset"][i]
    uv, depth = project_to_image(relative[None], poses["altitude"][i], width, height, poses["focal_px"][i], basis)
    u, v = uv[0]
    if depth[0] <= 0 or not (0 <= u < width and 0 <= v < height):
        return None
    rays = pixel_rays(np.array([u, u + 1, u]), np.array([v, v, v + 1]), width, height, poses["focal_px"][i], basis)
    if np.any(rays[:, 2] >= 0):
        return None
    ground = rays[:, :2] * (poses["altitude"][i] / -rays[:, 2])[:, None]
    (ax, ay), (bx, by) = ground[1] - ground[0], ground[2] - ground[0]
    area = abs(ax * by - ay * bx)
    return float(np.sqrt(area))


def _nice(value: float) -> float:
    """Round up to two significant digits (0.172 -> 0.18)."""
    power = 10 ** (np.floor(np.log10(value)) - 1)
    return float(np.ceil(value / power - 1e-9) * power)


def enhance_region(paths: list[str], region: dict, adjustments: dict | None = None,
                   resolution_m_per_px: float | None = None) -> dict:
    """Re-render ``region`` ({"east": [a, b], "north": [c, d]}) of one or two stored results.

    Several paths (a frame comparison) share one canvas and resolution, so flicker comparison keeps working.
    Returns {"variants": [...]} in the format of ``mosaic.mosaic_selected_frames``.
    """
    from pyproj import Geod

    from .mosaic import mosaic_selected_frames
    from .pose import frame_poses, load_context
    from .refine import apply_corrections

    if not paths:
        raise ValueError("Нет изображения для улучшения")
    metas = [load_georaster(path) for path in paths]
    east = sorted(map(float, region["east"]))
    north = sorted(map(float, region["north"]))
    # The visible window may extend beyond the raster: keep the part covered by the stored results.
    east = [max(east[0], min(m["bounds_m"]["east"][0] for m in metas)), min(east[1], max(m["bounds_m"]["east"][1] for m in metas))]
    north = [max(north[0], min(m["bounds_m"]["north"][0] for m in metas)), min(north[1], max(m["bounds_m"]["north"][1] for m in metas))]
    span = np.array([east[1] - east[0], north[1] - north[0]])
    if np.any(span <= 0):
        raise ValueError("Видимая область не пересекается с изображением")
    # 5×5 probe points: the finest detail anywhere in the window sets the resolution.
    probes = np.stack(np.meshgrid(np.linspace(*east, 5), np.linspace(*north, 5)), axis=-1).reshape(-1, 2)

    native, jobs = [], []
    geod = Geod(ellps="WGS84")
    for meta in metas:
        if not meta.get("video") or not meta.get("times_s"):
            raise ValueError("Для этого результата не сохранены кадры; постройте его заново")
        context = load_context(meta["video"])
        fps = context.meta["fps"]
        times = np.round(np.asarray(meta["times_s"], dtype=np.float64) * fps) / fps
        poses = frame_poses(context, times, geod, float(meta.get("yaw_offset_deg") or 0.0),
                            float(meta.get("takeoff_height_m") or 0.0), float(meta.get("horizontal_fov_deg") or 0.0))
        corrections = meta.get("attitude_corrections_deg")
        if corrections is not None and len(corrections) == len(times):
            poses = apply_corrections(poses, np.asarray(corrections, dtype=np.float64).reshape(-1, 3))
        else:
            corrections = np.zeros((len(times), 3)).tolist()
        for i in range(len(times)):
            for point in probes:
                value = _native_resolution(poses, i, point, context.meta["width"], context.meta["height"])
                if value:
                    native.append(value)
        jobs.append((meta, times, corrections))

    if resolution_m_per_px:
        resolution = float(resolution_m_per_px)
    else:
        # Finest detail any frame offers at the region centre, but not finer than the canvas limits allow.
        finest = min(native) if native else min(float(m["resolution_m_per_px"]) for m in metas) / 4
        if max(len(m.get("times_s") or []) for m in metas) >= 4:
            finest /= 1.25   # many sub-pixel-shifted frames (averages) carry detail finer than one frame
        limit = max(np.sqrt(span.prod() / MAX_PIXELS), span.max() / MAX_SIDE)
        resolution = _nice(max(finest, limit, FINEST_M))
        stored = min(float(m["resolution_m_per_px"]) for m in metas)
        if resolution >= stored * 0.95:
            if limit > finest:
                raise ValueError(f"Область {span[0]:.0f}×{span[1]:.0f} м слишком большая для улучшения: приблизьте участок "
                                 f"(до ~{MAX_SIDE * finest:.0f} м по стороне)")
            raise ValueError(f"Изображение уже в нативном разрешении кадра ({stored:.2f} м/пкс против {finest:.2f} м)")

    # One canvas for several results (frame comparison), so they can still be flicker-compared.
    target = {"east": east, "north": north, "exact": len(metas) > 1}
    variants = []
    for meta, times, corrections in jobs:
        project_dir = Path(meta["image"]).parent.parent
        result = mosaic_selected_frames(
            meta["video"], times, str(project_dir), float(meta.get("takeoff_height_m") or 0.0),
            float(meta.get("horizontal_fov_deg") or 0.0), resolution, float(meta.get("yaw_offset_deg") or 0.0),
            float(meta.get("max_distance_m") or 10_000.0), adjustments if adjustments is not None else meta.get("adjustments"),
            float(meta.get("bottom_crop_percent") or 0.0), output_kind="enhanced",
            result_metadata={"enhanced_from": meta["image"], "source_kind": meta.get("kind"),
                             "native_resolution_m": None if not native else round(min(native), 3)},
            refine=False, attitude_corrections=corrections, region_m=target,
            interpolation="lanczos", robust=meta.get("kind") == "temporal_average",
        )
        result["time_s"] = round(float(times[0]), 3)
        variants.append(result)
    return {"variants": variants, "resolution_m_per_px": variants[0]["resolution_m_per_px"],
            "native_resolution_m": None if not native else round(min(native), 3)}
