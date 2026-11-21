"""Local GEBCO bathymetry, read lazily for the small box shown by a projection.

The configured source is a CF NetCDF file or a directory of NetCDF tiles.  Only the
latitude/longitude slice around the current raster is read; the full (potentially multi-GB)
elevation array is never loaded.  GEBCO elevations are metres relative to mean sea level,
so a requested 50 m isobath is the ``elevation == -50`` contour.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .paths import bathymetry_path

DEFAULT_LEVELS_M = [5, 10, 20, 50, 100]
MAX_LEVELS = 20
MAX_DEPTH_M = 12_000
MAX_AXIS_SAMPLES = 1600
COLORS = ("#32c7ff", "#20aee8", "#168fcf", "#276fba", "#4459a9", "#633f98")


def parse_levels(values) -> list[float]:
    """Positive, unique isobath depths in metres, safe for a compact web response."""
    if isinstance(values, str):
        values = values.replace(";", ",").split(",")
    try:
        result = sorted({float(value) for value in values if str(value).strip()})
    except (TypeError, ValueError) as exc:
        raise ValueError("Глубины изолиний задаются числами через запятую") from exc
    if not result:
        raise ValueError("Укажите хотя бы одну глубину изолинии")
    if len(result) > MAX_LEVELS or result[0] <= 0 or result[-1] > MAX_DEPTH_M:
        raise ValueError(f"Допустимо 1–{MAX_LEVELS} глубин от 0 до {MAX_DEPTH_M} м")
    return result


def _files(path: Path | None = None) -> list[Path]:
    source = path or bathymetry_path()
    if source is None:
        return []
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted({*source.glob("*.nc"), *source.glob("*.nc4")})
    return []


def status() -> dict:
    source = bathymetry_path()
    files = _files(source)
    return {"path": None if source is None else str(source), "available": bool(files),
            "files": len(files), "default_levels_m": DEFAULT_LEVELS_M,
            "error": None if source is None or files else "В выбранной папке нет файлов .nc/.nc4"}


def _netcdf():
    try:
        import netCDF4
    except ImportError as exc:
        raise RuntimeError("Для батиметрии нужен пакет netCDF4; перезапустите DroneVision.bat") from exc
    return netCDF4


def _variable(dataset, names: tuple[str, ...], standard_name: str | None = None):
    for name in names:
        if name in dataset.variables:
            return dataset.variables[name]
    for variable in dataset.variables.values():
        if standard_name and getattr(variable, "standard_name", "") == standard_name:
            return variable
    return None


def _index_span(coordinate: np.ndarray, low: float, high: float) -> tuple[int, int] | None:
    finite = np.isfinite(coordinate)
    indices = np.flatnonzero(finite & (coordinate >= low) & (coordinate <= high))
    if not len(indices):
        return None
    # One extra cell is required on every side to interpolate a contour crossing the bbox.
    return max(0, int(indices[0]) - 1), min(len(coordinate), int(indices[-1]) + 2)


def _geographic_box(lat0: float, lon0: float, bounds_m: dict, margin_m: float = 2000.0):
    from pyproj import Transformer
    from .georaster import aeqd_crs

    east = bounds_m["east"]
    north = bounds_m["north"]
    e = np.array([east[0] - margin_m, east[1] + margin_m] * 2)
    n = np.array([north[0] - margin_m] * 2 + [north[1] + margin_m] * 2)
    lon, lat = Transformer.from_crs(aeqd_crs(lat0, lon0), "EPSG:4326", always_xy=True).transform(e, n)
    # Projection footprints are small.  A jump here means the box crosses the date line; express
    # its longitudes in the same 360-degree branch as the takeoff point.
    lon = lon0 + (np.asarray(lon) - lon0 + 180.0) % 360.0 - 180.0
    return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())


def _read_subset(path: Path, bbox, lon0: float):
    netCDF4 = _netcdf()
    west, south, east, north = bbox
    with netCDF4.Dataset(path, "r") as dataset:
        lat_var = _variable(dataset, ("lat", "latitude", "y"), "latitude")
        lon_var = _variable(dataset, ("lon", "longitude", "x"), "longitude")
        z_var = _variable(dataset, ("elevation", "z", "depth"), "height")
        if lat_var is None or lon_var is None or z_var is None:
            raise ValueError(f"{path.name}: не найдены lon, lat и elevation")
        lat = np.asarray(lat_var[:], dtype=np.float64).squeeze()
        lon_raw = np.asarray(lon_var[:], dtype=np.float64).squeeze()
        if lat.ndim != 1 or lon_raw.ndim != 1 or z_var.ndim != 2:
            raise ValueError(f"{path.name}: ожидается географическая сетка elevation(lat, lon)")
        lon = lon0 + (lon_raw - lon0 + 180.0) % 360.0 - 180.0
        iy = _index_span(lat, south, north)
        ix = _index_span(lon, west, east)
        if iy is None or ix is None:
            return None
        y0, y1 = iy
        x0, x1 = ix
        sy = max(1, int(np.ceil((y1 - y0) / MAX_AXIS_SAMPLES)))
        sx = max(1, int(np.ceil((x1 - x0) / MAX_AXIS_SAMPLES)))
        z = np.ma.asarray(z_var[y0:y1:sy, x0:x1:sx], dtype=np.float32).filled(np.nan)
        return lat[y0:y1:sy], lon[x0:x1:sx], z


def _contour_lines(lat, lon, elevation, depth_m: float, transformer) -> list[list[list[float]]]:
    import cv2

    if elevation.shape[0] < 2 or elevation.shape[1] < 2:
        return []
    mask = np.isfinite(elevation) & (elevation <= -depth_m)
    if not mask.any() or mask.all():
        return []
    found, _ = cv2.findContours((mask.astype(np.uint8) * 255), cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
    lines = []
    for contour in found:
        points = contour[:, 0, :]
        if len(points) < 2:
            continue
        # Half-cell precision is enough for a 15 arc-second source; simplification keeps JSON small.
        points = cv2.approxPolyDP(contour, 0.35, False)[:, 0, :]
        x = np.clip(points[:, 0], 0, len(lon) - 1)
        y = np.clip(points[:, 1], 0, len(lat) - 1)
        east, north = transformer.transform(lon[x], lat[y])
        line = np.round(np.column_stack((east, north)), 1).tolist()
        if len(line) >= 2:
            lines.append(line)
    return lines


def contours(lat0: float, lon0: float, bounds_m: dict, levels=DEFAULT_LEVELS_M) -> dict:
    """Isobaths for one ENU raster.  Reads only the source cells intersecting its padded bbox."""
    from pyproj import Transformer
    from .georaster import aeqd_crs

    levels = parse_levels(DEFAULT_LEVELS_M if levels is None else levels)
    files = _files()
    source = bathymetry_path()
    if not files:
        message = "Укажите файл GEBCO NetCDF" if source is None else "В выбранном пути нет NetCDF-файлов"
        return {"source": None if source is None else str(source), "contours": [], "error": message,
                "levels_m": levels}
    bbox = _geographic_box(float(lat0), float(lon0), bounds_m)
    transformer = Transformer.from_crs("EPSG:4326", aeqd_crs(float(lat0), float(lon0)), always_xy=True)
    result = {depth: [] for depth in levels}
    used, errors = [], []
    for path in files:
        try:
            subset = _read_subset(path, bbox, float(lon0))
            if subset is None:
                continue
            lat, lon, elevation = subset
            used.append(path.name)
            for depth in levels:
                result[depth].extend(_contour_lines(lat, lon, elevation, depth, transformer))
        except Exception as exc:
            errors.append(str(exc))
    layers = [{"depth_m": depth, "color": COLORS[index % len(COLORS)], "lines": result[depth]}
              for index, depth in enumerate(levels)]
    response = {"source": str(source), "files": used, "levels_m": levels, "contours": layers,
                "bbox": {"west": bbox[0], "south": bbox[1], "east": bbox[2], "north": bbox[3]}}
    if errors:
        response["error"] = "; ".join(errors[:3])
    elif not used:
        response["error"] = "Выбранная батиметрия не покрывает район съёмки"
    return response
