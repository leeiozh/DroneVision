"""Georeferenced rasters produced by the projection and the frame mosaic.

Every displayed PNG (with the burned-in grid) gets two companions:

* ``<stem>_data.png`` — the same raster without grid and without colour adjustments, BGRA,
  alpha 0 where no frame covers the canvas.  Intensity profiles are sampled from it;
* ``<stem>.geo.json`` — the georeference: the canvas is north up in metres east/north of the
  takeoff point, i.e. an azimuthal equidistant projection (geodesic azimuth + distance, as in
  ``pose.enu_offsets``) centred on the takeoff point.  Pixel (i, j) has its centre at
  ``(east_min + i * res, north_max - j * res)`` — the OpenCV convention used by the warps.

GeoTIFF export resamples the adjusted raster into UTM or WGS84 and writes the GeoTIFF tags
directly (no GDAL dependency).
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import struct
import threading
import zlib

import numpy as np

from .adjust import apply_adjustments, describe, normalize
from .cvio import imread, imwrite

_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


def aeqd_crs(lat0: float, lon0: float) -> str:
    return f"+proj=aeqd +lat_0={lat0:.9f} +lon_0={lon0:.9f} +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"


def _transformer(source: str, target: str):
    from pyproj import Transformer
    return Transformer.from_crs(source, target, always_xy=True)


def geo_grid(lat0: float, lon0: float, minimum, maximum, nodes: int = 17) -> dict:
    """Latitude/longitude on a regular ENU node grid for bilinear lookup in the browser.

    With nodes every ≤ 1 km the interpolation error is below a millimetre.
    """
    east = np.linspace(float(minimum[0]), float(maximum[0]), nodes)
    north = np.linspace(float(minimum[1]), float(maximum[1]), nodes)
    e, n = np.meshgrid(east, north)
    lon, lat = _transformer(aeqd_crs(lat0, lon0), "EPSG:4326").transform(e, n)
    return {"east": [float(east[0]), float(east[-1])], "north": [float(north[0]), float(north[-1])], "nodes": nodes,
            "lat": np.round(lat, 8).tolist(), "lon": np.round(lon, 8).tolist()}


def save_georaster(display_path: Path, data_bgr: np.ndarray, valid: np.ndarray, minimum, maximum,
                   resolution: float, takeoff, kind: str, adjustments: dict | None = None, **extra) -> dict:
    """Write the data raster and the sidecar next to ``display_path``; return fields for the web response."""
    import cv2

    display_path = Path(display_path)
    data_path = display_path.with_name(f"{display_path.stem}_data.png")
    rgba = np.dstack([data_bgr[..., :3], np.where(valid, 255, 0).astype(np.uint8)])
    if not imwrite(data_path, rgba):
        raise RuntimeError("Не удалось сохранить растр данных")
    lat0, lon0 = map(float, takeoff)
    height, width = data_bgr.shape[:2]
    sidecar = {
        "image": str(display_path), "data": str(data_path), "kind": kind,
        "size_px": [int(width), int(height)],
        "bounds_m": {"east": [float(minimum[0]), float(maximum[0])], "north": [float(minimum[1]), float(maximum[1])]},
        "resolution_m_per_px": float(resolution),
        "reference": {"latitude": lat0, "longitude": lon0, "crs": aeqd_crs(lat0, lon0),
                      "meaning": "метры на восток/север от точки взлёта, азимутальная равнопромежуточная проекция"},
        "pixel_centre": "(east_min + i*res, north_max - j*res)",
        "adjustments": normalize(adjustments),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }
    geo_path = display_path.with_suffix(".geo.json")
    geo_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"data_path": str(data_path), "geo_path": str(geo_path), "adjustments": sidecar["adjustments"],
            "geo_grid": geo_grid(lat0, lon0, minimum, maximum)}


def load_georaster(image_path: str) -> dict:
    """Sidecar of a displayed projection/mosaic PNG; only rasters written by ``save_georaster`` are accepted."""
    path = Path(image_path).expanduser().resolve()
    geo_path = path.with_suffix(".geo.json")
    if path.suffix.lower() != ".png" or not geo_path.is_file():
        raise ValueError("Для этого изображения нет геопривязки; постройте проекцию заново")
    meta = json.loads(geo_path.read_text(encoding="utf-8"))
    if Path(meta["data"]).resolve().parent != path.parent:
        raise ValueError("Повреждён файл геопривязки")
    return meta


def load_data(meta: dict) -> np.ndarray:
    """BGRA data raster with a small cache (profiles are requested repeatedly for one image)."""
    import cv2

    path = Path(meta["data"])
    key = (str(path), path.stat().st_mtime_ns)
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]
    image = imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Не удалось прочитать {path.name}")
    if image.ndim == 2 or image.shape[2] == 3:
        image = np.dstack([image if image.ndim == 3 else np.dstack([image] * 3), np.full(image.shape[:2], 255, np.uint8)])
    with _CACHE_LOCK:
        while len(_CACHE) >= 3:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = image
    return image


def pixel_to_enu(meta: dict, x, y) -> tuple[np.ndarray, np.ndarray]:
    res = meta["resolution_m_per_px"]
    return meta["bounds_m"]["east"][0] + np.asarray(x) * res, meta["bounds_m"]["north"][1] - np.asarray(y) * res


def enu_to_pixel(meta: dict, east, north) -> tuple[np.ndarray, np.ndarray]:
    res = meta["resolution_m_per_px"]
    return (np.asarray(east) - meta["bounds_m"]["east"][0]) / res, (meta["bounds_m"]["north"][1] - np.asarray(north)) / res


def enu_to_lonlat(meta: dict, east, north) -> tuple[np.ndarray, np.ndarray]:
    return _transformer(meta["reference"]["crs"], "EPSG:4326").transform(np.asarray(east, float), np.asarray(north, float))


# --- GeoTIFF -----------------------------------------------------------------------------------------------

def _utm_epsg(lat: float, lon: float) -> int:
    zone = int(math.floor((lon + 180) / 6)) % 60 + 1
    # Norway/Svalbard exceptions of the UTM grid.
    if 56 <= lat < 64 and 3 <= lon < 12:
        zone = 32
    if 72 <= lat < 84 and 0 <= lon < 42:
        zone = 31 if lon < 9 else 33 if lon < 21 else 35 if lon < 33 else 37
    return (32600 if lat >= 0 else 32700) + zone


def _target_grid(meta: dict, crs: str):
    """Output CRS, its EPSG code, pixel size and upper-left corner covering the whole raster."""
    lat0, lon0 = meta["reference"]["latitude"], meta["reference"]["longitude"]
    width, height = meta["size_px"]
    res = meta["resolution_m_per_px"]
    # Outline of the raster (outer pixel edges), densified: straight edges become slightly curved.
    u = np.linspace(0, 1, 65)
    xs = np.concatenate([-0.5 + u * width, -0.5 + u * width, np.full(65, -0.5), np.full(65, width - 0.5)])
    ys = np.concatenate([np.full(65, -0.5), np.full(65, height - 0.5), -0.5 + u * height, -0.5 + u * height])
    east, north = pixel_to_enu(meta, xs, ys)
    if crs == "wgs84":
        epsg = 4326
        lon, lat = enu_to_lonlat(meta, east, north)
        # Same ground sampling as the source: metres -> degrees at the reference latitude.
        a, f = 6378137.0, 1 / 298.257223563
        e2 = f * (2 - f)
        phi = math.radians(lat0)
        w = math.sqrt(1 - e2 * math.sin(phi) ** 2)
        meridional, normal = a * (1 - e2) / w ** 3, a / w
        step = (math.degrees(res / (normal * math.cos(phi))), math.degrees(res / meridional))
        x_min, y_max = lon.min(), lat.max()
        x_max, y_min = lon.max(), lat.min()
    else:
        epsg = _utm_epsg(lat0, lon0)
        x, y = _transformer(meta["reference"]["crs"], f"EPSG:{epsg}").transform(east, north)
        step = (res, res)
        x_min, x_max = math.floor(x.min() / res) * res, math.ceil(x.max() / res) * res
        y_min, y_max = math.floor(y.min() / res) * res, math.ceil(y.max() / res) * res
    out_w = int(math.ceil((x_max - x_min) / step[0]))
    out_h = int(math.ceil((y_max - y_min) / step[1]))
    return epsg, step, (float(x_min), float(y_max)), out_w, out_h


def _resample(meta: dict, rgba: np.ndarray, epsg: int, step, origin, out_w: int, out_h: int,
              coarse: int = 8, strip: int = 512) -> np.ndarray:
    """Inverse-map every output pixel centre into the source raster (exact transform on a coarse grid,
    bilinear in between — the map is smooth, the error is far below a pixel)."""
    import cv2

    to_enu = _transformer(f"EPSG:{epsg}", meta["reference"]["crs"])
    output = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    cols = np.arange(0, out_w + coarse, coarse)
    for r0 in range(0, out_h, strip):
        r1 = min(r0 + strip, out_h)
        rows = np.arange(r0 - r0 % coarse, r1 + coarse, coarse)
        gx, gy = np.meshgrid(origin[0] + (cols + 0.5) * step[0], origin[1] - (rows + 0.5) * step[1])
        east, north = to_enu.transform(gx, gy)
        px, py = enu_to_pixel(meta, east, north)
        # Bilinear expansion of the coarse maps to every output pixel of the strip.
        yy = (np.arange(r0, r1) - rows[0]) / coarse
        xx = np.arange(out_w) / coarse
        iy, fy = np.minimum(yy.astype(int), len(rows) - 2), (yy % 1)[:, None]
        ix, fx = np.minimum(xx.astype(int), len(cols) - 2), (xx % 1)[None, :]

        def expand(grid):
            top = grid[iy][:, ix] * (1 - fx) + grid[iy][:, ix + 1] * fx
            bottom = grid[iy + 1][:, ix] * (1 - fx) + grid[iy + 1][:, ix + 1] * fx
            return (top * (1 - fy) + bottom * fy).astype(np.float32)

        output[r0:r1] = cv2.remap(rgba, expand(px), expand(py), cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    # Uncovered source pixels are black, so a partly covered output pixel is premultiplied by its alpha:
    # restore the colour, then keep a hard mask (half covered or more).
    alpha = output[..., 3:].astype(np.float32)
    colour = np.clip(output[..., :3] * (255 / np.maximum(alpha, 1)), 0, 255)
    return np.dstack([np.where(alpha >= 128, colour, 0).astype(np.uint8),
                      np.where(alpha >= 128, 255, 0).astype(np.uint8)])


def write_geotiff(path: Path, rgba: np.ndarray, pixel_scale, tiepoint, geokeys: list[tuple[int, int]],
                  citation: str, description: str = "") -> None:
    """Minimal little-endian GeoTIFF: 8-bit RGBA, Adobe deflate + horizontal predictor, 64 rows per strip."""
    height, width = rgba.shape[:2]
    rows_per_strip = 64
    strips = []
    for r0 in range(0, height, rows_per_strip):
        block = np.ascontiguousarray(rgba[r0:r0 + rows_per_strip], dtype=np.uint8)
        diff = block.copy()
        diff[:, 1:] = block[:, 1:] - block[:, :-1]  # uint8 wrap-around, as TIFF predictor 2 expects
        strips.append(zlib.compress(diff.tobytes(), 6))

    ascii_params = (citation + "|").encode("ascii", "replace") + b"\0"
    geo_directory = [1, 1, 0, len(geokeys) + 1]
    for key, value in sorted(geokeys + [(1026, None)]):  # 1026 GTCitationGeoKey lives in GeoAsciiParams
        geo_directory += [1026, 34737, len(ascii_params) - 1, 0] if key == 1026 else [key, 0, 1, value]

    def entries(strip_offsets):
        # tag, type (2 ASCII, 3 SHORT, 4 LONG, 12 DOUBLE), values; sorted by tag as TIFF requires
        return [
            (256, 4, [width]), (257, 4, [height]), (258, 3, [8, 8, 8, 8]), (259, 3, [8]), (262, 3, [2]),
            (270, 2, description.encode("utf-8") + b"\0"), (273, 4, strip_offsets), (277, 3, [4]),
            (278, 4, [rows_per_strip]), (279, 4, [len(s) for s in strips]), (284, 3, [1]),
            (305, 2, b"DroneVision\0"), (317, 3, [2]), (338, 3, [2]), (339, 3, [1, 1, 1, 1]),
            (33550, 12, [float(pixel_scale[0]), float(pixel_scale[1]), 0.0]),
            (33922, 12, [0.0, 0.0, 0.0, float(tiepoint[0]), float(tiepoint[1]), 0.0]),
            (34735, 3, geo_directory), (34737, 2, ascii_params),
        ]

    def layout(strip_offsets):
        items = entries(strip_offsets)
        cursor = 8 + 2 + 12 * len(items) + 4
        directory, blobs = [struct.pack("<H", len(items))], []
        for tag, kind, values in items:
            data = bytes(values) if kind == 2 else struct.pack(f"<{len(values)}{dict(((3, 'H'), (4, 'I'), (12, 'd')))[kind]}", *values)
            directory.append(struct.pack("<HHI", tag, kind, len(values)))
            if len(data) <= 4:
                directory.append(data.ljust(4, b"\0"))
            else:
                directory.append(struct.pack("<I", cursor))
                data += b"\0" * (len(data) % 2)
                blobs.append(data)
                cursor += len(data)
        return b"II*\0" + struct.pack("<I", 8) + b"".join(directory) + struct.pack("<I", 0) + b"".join(blobs)

    header_size = len(layout([0] * len(strips)))
    offsets = list(np.cumsum([header_size] + [len(s) for s in strips[:-1]]).astype(int))
    if header_size + sum(len(s) for s in strips) >= 2 ** 32:
        raise ValueError("GeoTIFF больше 4 ГБ; увеличьте метров/пиксель")
    with open(path, "wb") as stream:
        stream.write(layout([int(o) for o in offsets]))
        for data in strips:
            stream.write(data)


def export_geotiff(image_path: str, crs: str = "utm") -> dict:
    """Write ``<stem>_<crs>.tif`` next to the PNG: adjusted colours, no grid, alpha = coverage."""
    import cv2

    meta = load_georaster(image_path)
    crs = "wgs84" if str(crs).lower() in ("wgs84", "4326", "epsg:4326", "latlon") else "utm"
    bgra = load_data(meta)
    adjusted = apply_adjustments(bgra, meta.get("adjustments"))
    rgba = cv2.cvtColor(adjusted, cv2.COLOR_BGRA2RGBA)
    rgba[rgba[..., 3] == 0] = 0  # adjustments may lift the empty background; resampling relies on it being black
    epsg, step, origin, out_w, out_h = _target_grid(meta, crs)
    if out_w * out_h > 120_000_000:
        raise ValueError("GeoTIFF получается слишком большим")
    resampled = _resample(meta, rgba, epsg, step, origin, out_w, out_h)
    if epsg == 4326:
        geokeys = [(1024, 2), (1025, 1), (2048, 4326), (2054, 9102)]
        citation = "WGS 84"
    else:
        geokeys = [(1024, 1), (1025, 1), (3072, epsg), (3076, 9001)]
        zone = epsg % 100
        citation = f"WGS 84 / UTM zone {zone}{'N' if epsg < 32700 else 'S'}"
    source = Path(meta["image"])
    output = source.with_name(f"{source.stem}_{'wgs84' if epsg == 4326 else f'utm{epsg % 100}'}.tif")
    description = (f"DroneVision {meta['kind']} · {Path(meta.get('video', '')).name} · "
                   f"adjustments: {describe(meta.get('adjustments'))}")
    write_geotiff(output, resampled, step, origin, geokeys, citation, description)
    return {"path": str(output), "epsg": epsg, "crs": citation, "size_px": [out_w, out_h],
            "pixel_size": [float(step[0]), float(step[1])], "upper_left": list(origin)}
