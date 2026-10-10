"""Intensity profiles along polylines drawn on a projection or a frame mosaic.

Sampling follows SNAP's transect export: one sample per raster pixel along every segment,
nearest pixel, consecutive repeats of the same pixel dropped.  Values come from the data
raster (``<stem>_data.png``: no grid, no colour adjustments); Gray = 0.299 R + 0.587 G + 0.114 B.
"""
from __future__ import annotations

from datetime import datetime, timezone
import re
from pathlib import Path

import numpy as np

from .adjust import describe
from .georaster import enu_to_lonlat, enu_to_pixel, load_data, load_georaster, pixel_to_enu

MAX_VERTICES = 500


def sample_profile(image_path: str, points_enu) -> tuple[dict, dict]:
    """Return (sidecar, samples) for a polyline given in metres east/north of the takeoff point."""
    meta = load_georaster(image_path)
    points = np.asarray(points_enu, dtype=np.float64).reshape(-1, 2)
    if not 2 <= len(points) <= MAX_VERTICES:
        raise ValueError(f"Нужна линия из 2–{MAX_VERTICES} точек")
    image = load_data(meta)
    height, width = image.shape[:2]
    px, py = enu_to_pixel(meta, points[:, 0], points[:, 1])
    xs, ys, segment, along = [], [], [], []
    distance = 0.0
    res = meta["resolution_m_per_px"]
    for k in range(len(points) - 1):
        length_px = float(np.hypot(px[k + 1] - px[k], py[k + 1] - py[k]))
        count = max(int(np.ceil(length_px)), 1)
        t = np.linspace(0, 1, count + 1)[:-1] if k < len(points) - 2 else np.linspace(0, 1, count + 1)
        xs.append(px[k] + (px[k + 1] - px[k]) * t)
        ys.append(py[k] + (py[k + 1] - py[k]) * t)
        segment.append(np.full(len(t), k + 1))
        along.append(distance + t * length_px * res)
        distance += length_px * res
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    segment, along = np.concatenate(segment), np.concatenate(along)
    ix, iy = np.rint(xs).astype(int), np.rint(ys).astype(int)
    keep = np.ones(len(ix), dtype=bool)
    keep[1:] = (ix[1:] != ix[:-1]) | (iy[1:] != iy[:-1])
    ix, iy, segment, along = ix[keep], iy[keep], segment[keep], along[keep]
    inside = (ix >= 0) & (iy >= 0) & (ix < width) & (iy < height)
    values = np.full((len(ix), 4), np.nan)
    values[inside] = image[iy[inside], ix[inside]][:, [2, 1, 0, 3]]  # BGRA -> R, G, B, alpha
    valid = inside & (values[:, 3] > 0)
    rgb = np.where(valid[:, None], values[:, :3], np.nan)
    gray = rgb @ np.array([0.299, 0.587, 0.114])
    east, north = pixel_to_enu(meta, ix, iy)
    lon, lat = enu_to_lonlat(meta, east, north)
    samples = {"x": ix, "y": iy, "segment": segment, "distance_m": along, "east_m": east, "north_m": north,
               "lon": lon, "lat": lat, "r": rgb[:, 0], "g": rgb[:, 1], "b": rgb[:, 2], "gray": gray,
               "valid": valid, "length_m": distance}
    return meta, samples


def samples_json(samples: dict) -> dict:
    def clean(array, digits):
        return [None if not np.isfinite(v) else round(float(v), digits) for v in array]
    return {"distance_m": clean(samples["distance_m"], 2), "r": clean(samples["r"], 0), "g": clean(samples["g"], 0),
            "b": clean(samples["b"], 0), "gray": clean(samples["gray"], 2),
            "east_m": clean(samples["east_m"], 2), "north_m": clean(samples["north_m"], 2),
            "lat": clean(samples["lat"], 7), "lon": clean(samples["lon"], 7),
            "count": int(len(samples["x"])), "valid": int(samples["valid"].sum()),
            "length_m": round(float(samples["length_m"]), 2)}


def profile_text(meta: dict, samples: dict, points_enu, name: str, shape_type: str, color: str = "") -> str:
    points = np.asarray(points_enu, dtype=np.float64).reshape(-1, 2)
    vlon, vlat = enu_to_lonlat(meta, points[:, 0], points[:, 1])
    times = meta.get("times_s") or []
    lines = [
        "# DroneVision transect export (SNAP-like, tab separated)",
        f"# Created:\t{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"# Object:\t{name}\t{shape_type}\t{color}",
        f"# Image:\t{meta['image']}",
        f"# Source:\t{meta.get('kind', '')}\t{meta.get('video', '')}",
        f"# Frame time, s:\t{' '.join(f'{t:.3f}' for t in times)}",
        *([f"# Flight start, UTC:\t{meta['flight_start_utc']}\t{meta.get('flight_log', '')}"]
          if meta.get("flight_start_utc") else []),
        *([f"# Heading correction, deg:\t{float(meta['yaw_offset_deg']):+.2f}"]
          if meta.get("yaw_offset_deg") is not None else []),
        *([f"# Log clock shift:\t{meta['log_sync']['utc_offset_hours']:+g} h"
           f"\t{meta['log_sync']['clock_offset_seconds']:+.2f} s"] if meta.get("log_sync") else []),
        f"# Reference (takeoff):\tlat {meta['reference']['latitude']:.8f}\tlon {meta['reference']['longitude']:.8f}",
        f"# CRS of E/N:\t{meta['reference']['crs']}",
        f"# Resolution, m/px:\t{meta['resolution_m_per_px']:.4f}",
        "# Values:\tdata raster without grid and without colour adjustments "
        f"(display adjustments: {describe(meta.get('adjustments'))}); Gray = 0.299R + 0.587G + 0.114B; NaN = no data",
        f"# Length, m:\t{samples['length_m']:.2f}",
        "# Vertices (E_m N_m Lat Lon):",
        *[f"#\t{e:.2f}\t{n:.2f}\t{la:.8f}\t{lo:.8f}" for (e, n), la, lo in zip(points, vlat, vlon)],
        "",
        "Name\tIndex\tSegment\tDistance_m\tX\tY\tE_m\tN_m\tLat\tLon\tR\tG\tB\tGray",
    ]

    def value(v, digits=0):
        return "NaN" if not np.isfinite(v) else f"{v:.{digits}f}"

    for i in range(len(samples["x"])):
        lines.append("\t".join([
            name, str(i + 1), str(int(samples["segment"][i])), f"{samples['distance_m'][i]:.2f}",
            f"{samples['x'][i] + 0.5:.1f}", f"{samples['y'][i] + 0.5:.1f}",
            f"{samples['east_m'][i]:.2f}", f"{samples['north_m'][i]:.2f}",
            f"{samples['lat'][i]:.8f}", f"{samples['lon'][i]:.8f}",
            value(samples["r"][i]), value(samples["g"][i]), value(samples["b"][i]), value(samples["gray"][i], 2)]))
    return "\n".join(lines) + "\n"


def export_profile(image_path: str, points_enu, name: str = "line", shape_type: str = "polyline",
                   color: str = "") -> dict:
    """Write the transect next to the project (``<project>/profiles``) and return its text."""
    meta, samples = sample_profile(image_path, points_enu)
    safe = re.sub(r"[^\w.-]+", "_", name, flags=re.UNICODE).strip("_") or "line"
    image = Path(meta["image"])
    output_dir = image.parent.parent / "profiles"
    output_dir.mkdir(parents=True, exist_ok=True)
    # <image name>_<lines | contours>_<object>: the image name already holds flight, video, times and type.
    output = output_dir / f"{image.stem}_{'contours' if shape_type == 'polygon' else 'lines'}_{safe}.txt"
    text = profile_text(meta, samples, points_enu, name, shape_type, color)
    output.write_text(text, encoding="utf-8")
    return {"path": str(output), "filename": output.name, "text": text, "count": int(len(samples["x"]))}
