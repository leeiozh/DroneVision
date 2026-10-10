"""Land polygons (coastline) for any survey area.

Source: OpenStreetMap land polygons, https://osmdata.openstreetmap.de (``land-polygons-split-4326``) —
the most detailed openly available global coastline; in the Arctic it is traced from Landsat/Sentinel
imagery, an order of magnitude finer than GSHHG or Natural Earth.  ODbL, © OpenStreetMap contributors.

The planet file is ~0.9 GB zipped / ~1.4 GB as a shapefile, so it is never read by the interactive tools.
Instead small regional extracts in the same shapefile format are cut from it on demand and kept in ``maps/``:

    <DroneData>/maps/osm-planet/land_polygons.shp        the planet (or the downloaded zip next to it)
    <DroneData>/maps/coast_osm_<W>_<S>_<E>_<N>.shp       extract covering exactly these whole degrees
    <program>/maps/…                                     extracts shipped with the program

``land_file(bbox)`` gives the extract that covers a box, cutting it when the planet file is available.
An extract may hold no polygons at all: that means "covered, open sea", unlike "no data".
"""
from __future__ import annotations

from pathlib import Path
import re
import struct
import urllib.request
import zipfile

import numpy as np

from . import paths

PLANET_URL = "https://osmdata.openstreetmap.de/download/land-polygons-split-4326.zip"
PLANET_DIR = "osm-planet"
PLANET_SHP = "land_polygons.shp"
SOURCE = "OpenStreetMap land polygons (osmdata.openstreetmap.de), ODbL"
EARTH_RADIUS_M = 6_371_000.0
REFRACTION = 0.13                      # standard terrestrial refraction coefficient
EFFECTIVE_RADIUS_M = EARTH_RADIUS_M / (1 - REFRACTION)
_EXTRACT = re.compile(r"coast_osm_(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)\.shp$")
_COVERAGE: dict[tuple, tuple | None] = {}


def map_dirs() -> list[Path]:
    """Folders searched for extracts: the data set first (writable, shared), then the program."""
    found = []
    for folder in (paths.data_root() / "maps", paths.MAPS_DIR):
        if folder not in found:
            found.append(folder)
    return found


def horizon_distance_m(height_m: float) -> float:
    return float(np.sqrt(2 * max(height_m, 0.0) * EFFECTIVE_RADIUS_M))


def horizon_dip_deg(height_m: float) -> float:
    """Depression of the sea horizon below the horizontal for an eye ``height_m`` above the water."""
    return float(np.rad2deg(np.sqrt(2 * max(height_m, 0.0) / EFFECTIVE_RADIUS_M)))


def box_around(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    dlat = radius_m / 111_000
    dlon = dlat / max(np.cos(np.deg2rad(lat)), 0.05)
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def _coverage(path: Path) -> tuple | None:
    """Box (west, south, east, north) a file answers for: from its name, else its shapefile header."""
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _COVERAGE:
        match = _EXTRACT.search(path.name)
        if match:
            _COVERAGE[key] = tuple(float(v) for v in match.groups())
        else:
            try:
                with path.open("rb") as stream:
                    header = stream.read(100)
                _COVERAGE[key] = struct.unpack_from("<4d", header, 36) if len(header) == 100 else None
            except OSError:
                _COVERAGE[key] = None
    return _COVERAGE[key]


def _candidates() -> list[Path]:
    files = []
    configured = paths._configured("DRONEVISION_COASTLINE", "coastline")
    if configured and configured.is_file():
        files.append(configured)
    for folder in map_dirs():
        try:
            files += sorted(folder.glob("*.shp"))
        except OSError:
            continue
    return files


def planet_file() -> Path | None:
    """The planet shapefile; a downloaded zip is unpacked once (only the .shp is needed)."""
    for folder in map_dirs():
        target = folder / PLANET_DIR / PLANET_SHP
        if target.is_file():
            return target
    for folder in map_dirs():
        archive = folder / PLANET_DIR / "land-polygons-split-4326.zip"
        if archive.is_file():
            target = archive.with_name(PLANET_SHP)
            try:
                with zipfile.ZipFile(archive) as bundle:
                    name = next(n for n in bundle.namelist() if n.endswith("land_polygons.shp"))
                    temporary = target.with_name(target.name + ".part")
                    with bundle.open(name) as source, temporary.open("wb") as sink:
                        while chunk := source.read(8 << 20):
                            sink.write(chunk)
                    temporary.replace(target)
            except (OSError, zipfile.BadZipFile, StopIteration):
                continue          # incomplete download
            return target
    return None


def download_planet(say=print) -> Path:
    """Download the planet land polygons into ``<DroneData>/maps/osm-planet`` (resumable)."""
    folder = next((f for f in map_dirs() if f.parent.is_dir()), paths.MAPS_DIR) / PLANET_DIR
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "land-polygons-split-4326.zip"
    done = archive.stat().st_size if archive.is_file() else 0
    request = urllib.request.Request(PLANET_URL, headers={"User-Agent": "DroneVision/0.1 coastline",
                                                          **({"Range": f"bytes={done}-"} if done else {})})
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as exc:
        if exc.code != 416:           # 416: nothing left to download
            raise
        response = None
    if response is not None:
        with response:
            resumed = response.status == 206
            total = int(response.headers.get("Content-Length") or 0) + (done if resumed else 0)
            with archive.open("ab" if resumed else "wb") as sink:
                got = done if resumed else 0
                step = 0
                while chunk := response.read(4 << 20):
                    sink.write(chunk)
                    got += len(chunk)
                    if got // (50 << 20) != step:
                        step = got // (50 << 20)
                        say(f"  береговая линия OSM: {got / 1e6:.0f} из {total / 1e6:.0f} МБ")
    result = planet_file()
    if result is None:
        raise RuntimeError(f"Архив {archive} скачан не полностью; запустите команду ещё раз")
    return result


def cut_extract(bbox: tuple[float, float, float, float], folder: Path | None = None) -> Path | None:
    """Copy the planet polygons that touch ``bbox`` (grown to whole degrees) into an extract."""
    planet = planet_file()
    if planet is None:
        return None
    west, south, east, north = (int(np.floor(bbox[0])), int(np.floor(bbox[1])),
                                int(np.ceil(bbox[2])), int(np.ceil(bbox[3])))
    records, box = [], [np.inf, np.inf, -np.inf, -np.inf]
    with planet.open("rb") as stream:
        size = stream.seek(0, 2)
        offset = 100
        while offset + 8 <= size:
            stream.seek(offset)
            head = stream.read(44)
            if len(head) < 44:
                break
            _, words = struct.unpack_from(">2i", head, 0)
            if struct.unpack_from("<i", head, 8)[0] == 5:
                xmin, ymin, xmax, ymax = struct.unpack_from("<4d", head, 12)
                if xmax >= west and xmin <= east and ymax >= south and ymin <= north:
                    stream.seek(offset + 8)
                    records.append(stream.read(words * 2))
                    box = [min(box[0], xmin), min(box[1], ymin), max(box[2], xmax), max(box[3], ymax)]
            offset += 8 + words * 2
    if not records:
        box = [west, south, east, north]
    # Extracts go to the program's maps/ first: they travel with it to other computers (``bundle``).
    for target_dir in ([folder] if folder else list(reversed(map_dirs()))):
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"coast_osm_{west}_{south}_{east}_{north}.shp"
            length = 100 + sum(8 + len(r) for r in records)
            header = bytearray(100)
            struct.pack_into(">i", header, 0, 9994)
            struct.pack_into(">i", header, 24, length // 2)
            struct.pack_into("<2i", header, 28, 1000, 5)
            struct.pack_into("<4d", header, 36, *box)
            temporary = target.with_name(target.name + ".part")
            with temporary.open("wb") as sink:
                sink.write(header)
                for number, record in enumerate(records, 1):
                    sink.write(struct.pack(">2i", number, len(record) // 2))
                    sink.write(record)
            temporary.replace(target)
            target.with_suffix(".txt").write_text(
                f"{SOURCE}\nВырезка {west}..{east}° в.д., {south}..{north}° с.ш.; полигонов: {len(records)}\n"
                "© OpenStreetMap contributors\n", encoding="utf-8")
            return target
        except OSError:
            continue
    return None


def land_file(bbox: tuple[float, float, float, float], create: bool = True) -> Path | None:
    """Extract covering ``bbox`` (west, south, east, north); None when the area has no data."""
    west, south, east, north = bbox
    best, best_area = None, None
    for path in _candidates():
        cover = _coverage(path)
        if cover and cover[0] <= west and cover[1] <= south and cover[2] >= east and cover[3] >= north:
            area = (cover[2] - cover[0]) * (cover[3] - cover[1])
            if best is None or area < best_area:
                best, best_area = path, area
    if best is None and create:
        best = cut_extract(bbox)
    return best


def land_file_around(lat: float, lon: float, radius_m: float = 70_000.0, create: bool = True) -> Path | None:
    return land_file(box_around(lat, lon, radius_m), create)


def read_rings(path: Path, bbox: tuple[float, float, float, float]) -> list[np.ndarray]:
    """Polygon rings (lon, lat) of the shapes intersecting ``bbox``."""
    data = path.read_bytes()
    west, south, east, north = bbox
    rings, offset = [], 100
    while offset + 8 <= len(data):
        _, words = struct.unpack_from(">2i", data, offset)
        offset += 8
        end = offset + words * 2
        if end > len(data):
            break
        if struct.unpack_from("<i", data, offset)[0] == 5:
            xmin, ymin, xmax, ymax = struct.unpack_from("<4d", data, offset + 4)
            if xmax >= west and xmin <= east and ymax >= south and ymin <= north:
                parts_count, points_count = struct.unpack_from("<2i", data, offset + 36)
                parts = list(struct.unpack_from(f"<{parts_count}i", data, offset + 44)) + [points_count]
                points = np.frombuffer(data, "<f8", 2 * points_count, offset + 44 + 4 * parts_count).reshape(-1, 2)
                rings += [points[start:stop] for start, stop in zip(parts[:-1], parts[1:])]
        offset = end
    return rings


def status() -> dict:
    planet = planet_file()
    return {"planet": str(planet) if planet else None,
            "extracts": [{"path": str(p), "coverage": _coverage(p)} for p in _candidates()],
            "source": SOURCE}


def prepare_for_catalogue(say=print) -> list[Path]:
    """Cut extracts for every video of the data set that has an SRT position (one per distinct area)."""
    from .catalog import find_srt, interesting_cases
    from .discovery import _srt_points

    made = []
    for case in interesting_cases().get("cases", []):
        try:
            srt = find_srt(Path(case["video_path"]))
            points = _srt_points(srt) if srt else np.empty((0, 3))
        except Exception:
            continue
        fixes = points[(np.abs(points[:, 1]) > 1e-3) & (np.abs(points[:, 2]) > 1e-3)] if len(points) else points
        if not len(fixes):
            continue
        lat, lon = float(np.median(fixes[:, 1])), float(np.median(fixes[:, 2]))
        before = land_file_around(lat, lon, create=False)
        path = before or land_file_around(lat, lon)
        if path and not before:
            made.append(path)
            say(f"  вырезка для {case.get('id')}: {path.name}")
    return made
