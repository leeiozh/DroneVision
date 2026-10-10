"""Cases of the start page: every video of the data folder that has DJI telemetry.

The folder is scanned; files are paired by name and extension wherever they lie inside it:

* video  ``*.MP4 / *.MOV / ...``;
* SRT    the same name or the same DJI number (``DJI_0448 Пена.MP4`` + ``DJI_0448.SRT``).  With several
         candidates the nearest in the folder tree wins, so equal numbers of different trips do not mix;
* logs   ``*FlightRecord*.txt / .json`` anywhere in the folder; the flight is chosen by GPS and time
         (``discovery.match_log``), the date in the file name only narrows the search.

``discovery.match_log`` uses the same index when nothing lies next to the video.  Titles, test
intervals and checked heading corrections of known videos are kept below and applied when the video
is found (matched by DJI number and shooting date).
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import time

from .paths import VIDEO_SUFFIXES, data_root_state


# White Sea 2026: (day, asset, focal_min, focal_max, checked heading correction).
WHITE_SEA_CASES = [
    ("09.07", "DJI_0323", 24, 24, -10.0), ("09.07", "DJI_0328", 24, 48, 0.0),
    ("10.07", "DJI_0337", 24, 48, 0.0), ("10.07", "DJI_0348", 24, 24, 0.0), ("10.07", "DJI_0349", 24, 24, 0.0),
    ("10.07", "DJI_0350", 24, 24, 0.0), ("10.07", "DJI_0351", 24, 24, 0.0), ("10.07", "DJI_0356", 31, 31, 0.0),
    ("10.07", "DJI_0357", 31, 48, 0.0), ("10.07", "DJI_0359", 48, 48, 0.0), ("10.07", "DJI_0360", 24, 48, 0.0),
    ("10.07", "DJI_0362", 48, 48, 0.0), ("10.07", "DJI_0369", 24, 24, 0.0), ("10.07", "DJI_0370", 24, 24, 0.0),
    ("11.07", "DJI_0371", 24, 48, -12.25), ("11.07", "DJI_0390", 24, 24, 0.0), ("11.07", "DJI_0392", 24, 24, 0.0),
    ("14.07", "DJI_0393", 24, 24, 13.6), ("14.07", "DJI_0423", 39, 39, 0.0), ("14.07", "DJI_0424", 24, 24, 0.0),
    ("15.07", "DJI_0439", 24, 24, 0.0), ("15.07", "DJI_0466", 24, 24, 0.0),
]

# (asset, title, description, date, test start, test end, resolution of the batch mosaic).
CASES = [
    ("DJI_0448", "Максимальное покрытие", "Наибольшая GPS-область в наборе: около 33 тыс. м² по bbox траектории.", "2023-07-18", 0, 12, 6.0),
    ("DJI_0204", "Лёд и судно", "Надирная стабильная съёмка: лучший первый тест геометрии и мозаики.", "2023-07-24", 0, 15, 4.0),
    ("DJI_0137", "Поле льдин", "Много контрастных льдин, удобно проверять сшивку соседних кадров.", "2023-07-24", 30, 50, 4.0),
    ("DJI_0331", "Разреженный лёд", "Тёмная вода и отдельные льдины — сложнее, но хорошо видны ошибки привязки.", "2023-07-24", 20, 40, 4.0),
    ("DJI_0402", "Внутренние волны", "Наклонная съёмка длинных полос на поверхности; тест перспективной развёртки.", "2023-07-18", 5, 20, 6.0),
    ("DJI_0455", "Судно в слике", "Судно и неоднородная шероховатость моря, удобные контрольные ориентиры.", "2023-07-18", 5, 20, 5.0),
    ("DJI_0465", "Морская пена", "Почти надир, судно и след — полезный переход к трекингу поверхности.", "2023-07-18", 5, 20, 3.0),
    ("DJI_0460", "Надир: судно и след", "Короткий тематический ролик NadirDrift с выраженной текстурой воды.", "2023-07-18", 3, 18, 3.0),
    ("DJI_0474", "Надир: пенный след", "Контрастный след и меняющееся положение судна, хороший тест временной стабильности.", "2023-07-18", 5, 25, 3.0),
]

_KNOWN = {(asset, date): {"title": title, "description": description, "start": start, "end": end, "resolution": res}
          for asset, title, description, date, start, end, res in CASES}
_KNOWN.update({(asset, f"2026-07-{day[:2]}"): {
    "title": "Белое море · " + (f"{fmin} мм" if fmin == fmax else f"зум {fmin}–{fmax} мм"), "yaw_offset": yaw}
    for day, asset, fmin, fmax, yaw in WHITE_SEA_CASES})

MAX_DEPTH = 7
MAX_CASES = 400
# Results, caches and system folders are not source data.
SKIP_DIRS = {"work", "runs", "projections", "mosaics", "profiles", "motion", "calibration", "dronevision-project",
             "photos", "maps", "srt_invalid", "$recycle.bin", "system volume information", "found.000", "__pycache__"}
_ASSET = re.compile(r"DJI_\d{4}", re.I)
_SRT_END = re.compile(rb"--> (\d\d):(\d\d):(\d\d),(\d{3})")
_SRT_DATE = re.compile(rb"(\d{4}-\d\d-\d\d) \d\d:\d\d:\d\d")
_PATH_DATE = re.compile(r"\d{4}-\d\d-\d\d")
_CACHE: dict[str, tuple[float, list]] = {}
CACHE_SECONDS = 300.0


def _walk(root: Path):
    """(folder, subfolder names, file names) down to MAX_DEPTH, without results and hidden folders."""
    base = len(root.parts)
    for folder, dirs, files in os.walk(root, onerror=lambda error: None):
        depth = len(Path(folder).parts) - base
        dirs[:] = [] if depth >= MAX_DEPTH else sorted(
            d for d in dirs if not d.startswith(".") and d.lower() not in SKIP_DIRS)
        yield Path(folder), files


def _srt_info(path: Path) -> tuple[float, str]:
    """(duration in seconds, shooting date) from the first and last subtitle; (0, "") for an empty file."""
    try:
        size = path.stat().st_size
        if size < 50:
            return 0.0, ""
        with path.open("rb") as stream:
            head = stream.read(600)
            stream.seek(max(0, size - 4096))
            tail = stream.read()
    except OSError:
        return 0.0, ""
    ends = _SRT_END.findall(tail)
    duration = 0.0
    if ends:
        hh, mm, ss, ms = map(int, ends[-1])
        duration = hh * 3600 + mm * 60 + ss + ms / 1000
    date = _SRT_DATE.search(head)
    return duration, date.group(1).decode() if date else ""


_LOG_DATE = re.compile(r"(\d{4})-(\d\d)-(\d\d)")
_INDEX: dict[str, tuple[float, dict]] = {}


def file_index(root: Path, refresh: bool = False) -> dict:
    """Videos, non-empty SRT files and flight logs under ``root`` (cached like the cases)."""
    key = str(root)
    cached = _INDEX.get(key)
    if cached and not refresh and time.monotonic() - cached[0] < CACHE_SECONDS:
        return cached[1]
    videos, srt, logs = [], [], []
    for folder, files in _walk(root):
        for name in files:
            lower = name.lower()
            if name.startswith("."):
                continue
            if lower.endswith(".srt"):
                srt.append(folder / name)
            elif "flightrecord" in lower and lower.endswith((".txt", ".json")):
                logs.append(folder / name)
            elif os.path.splitext(lower)[1] in VIDEO_SUFFIXES:
                videos.append(folder / name)
    by_name: dict[str, list[Path]] = {}
    for path in srt:
        match = _ASSET.search(path.name)
        for name in {path.stem.lower(), match.group(0).lower() if match else path.stem.lower()}:
            by_name.setdefault(name, []).append(path)
    by_asset: dict[str, list[Path]] = {}
    for path in videos:
        by_asset.setdefault(_asset(path).lower(), []).append(path)
    index = {"root": root, "videos": sorted(videos), "srt": by_name, "logs": sorted(logs), "by_asset": by_asset}
    _INDEX[key] = (time.monotonic(), index)
    return index


def _asset(video: Path) -> str:
    match = _ASSET.search(video.name)
    return match.group(0).upper() if match else video.stem


def _closeness(a: Path, b: Path) -> int:
    """Number of leading folders two files share: the larger, the nearer in the tree."""
    shared = 0
    for x, y in zip(a.parent.parts, b.parent.parts):
        if x != y:
            break
        shared += 1
    return shared


def _beside(video: Path, srt: Path) -> bool:
    """The usual places: the folder of the video or around it, or ``telemetry/srt`` of its dataset."""
    around = [video.parent, *list(video.parent.parents)[:5]]
    if srt.parent in around or srt.parent.parent in around:
        return True
    return any(srt.parent == folder / "telemetry" / "srt" for folder in video.parents)


def _fits(video: Path, srt: Path, index: dict) -> bool:
    """A pair found by name far from the video is accepted only when nothing contradicts it:
    no other video with this number lies nearer to the SRT, and the durations agree.
    DJI numbers repeat between trips; without this a 2023 video took the SRT of a 2026 one."""
    from .video import quick_duration

    rivals = index["by_asset"].get(_asset(video).lower(), [])
    if any(_closeness(other, srt) > _closeness(video, srt) for other in rivals if other != video):
        return False
    length, (srt_length, _) = quick_duration(video), _srt_info(srt)
    return length is not None and srt_length > 0 and abs(length - srt_length) <= max(1.0, 0.02 * length)


def paired_srt(video: Path, index: dict) -> Path | None:
    """SRT of a video by name: the same stem first, then the same DJI number; the nearest one wins."""
    stem, asset = video.stem.lower(), _asset(video).lower()
    candidates = []
    for path, exact in [(p, 1) for p in index["srt"].get(stem, [])] + [(p, 0) for p in index["srt"].get(asset, [])]:
        try:
            if path.stat().st_size < 50:        # DJI writes empty SRT files for failed recordings
                continue
        except OSError:
            continue
        candidates.append((_closeness(video, path), exact, path))
    for _, _, path in sorted(candidates, key=lambda item: item[:2], reverse=True):
        if _beside(video, path) or _fits(video, path, index):
            return path
    return None


def logs_for_date(index: dict, date: str) -> list[Path]:
    """Flight logs that can belong to a recording of ``date`` (YYYY-MM-DD): a dated file name must be
    within a day of it (time zones); logs without a date in the name are always kept."""
    if not date:
        return list(index["logs"])
    from datetime import date as Date
    try:
        day = Date.fromisoformat(date)
    except ValueError:
        return list(index["logs"])
    kept = []
    for path in index["logs"]:
        found = _LOG_DATE.search(path.name)
        try:
            near = found is None or abs((Date(*map(int, found.groups())) - day).days) <= 1
        except ValueError:
            near = True
        if near:
            kept.append(path)
    return kept


def _index_for(video: Path) -> dict | None:
    """Index of the current data folder when ``video`` lies inside it."""
    state = data_root_state()
    root = state["path"]
    if not state["available"]:
        return None
    try:
        video.relative_to(root.resolve())
    except ValueError:
        return None
    return file_index(root.resolve())


def find_srt(video: Path) -> Path | None:
    """For ``discovery.match_log``: SRT of ``video`` anywhere in the data folder."""
    index = _index_for(video)
    return paired_srt(video, index) if index else None


def find_logs(video: Path, date: str) -> list[Path]:
    """For ``discovery.match_log``: candidate flight logs of the data folder for a recording date."""
    index = _index_for(video)
    return logs_for_date(index, date) if index else []


def _scan(root: Path, refresh: bool = False) -> list[dict]:
    index = file_index(root, refresh)
    cases = []
    for video in index["videos"]:
        asset = _asset(video)
        srt_path = paired_srt(video, index)
        duration, date = _srt_info(srt_path) if srt_path else (0.0, "")
        relative = video.parent.relative_to(root)
        if not date:
            found = _PATH_DATE.search(str(relative))
            date = found.group(0) if found else ""
        cases.append({"asset": asset, "video": video, "relative": relative, "srt": bool(srt_path) and duration > 0,
                      "srt_path": srt_path, "log": bool(logs_for_date(index, date)), "duration": duration, "date": date,
                      "label": video.stem[len(asset):].strip(" _-") if video.stem.upper().startswith(asset) else ""})
    return cases


def _describe(item: dict, root: Path) -> dict:
    known = _KNOWN.get((item["asset"], item["date"]), {})
    duration = item["duration"]
    available = item["srt"] and item["log"]
    yaw_offset = float(known.get("yaw_offset", 0.0))
    folder = str(item["relative"]).replace("\\", "/")
    if not item["srt"]:
        description = "нет SRT с телеметрией — геопривязка недоступна"
    elif not item["log"]:
        description = "не найдены полётные логи DJIFlightRecord"
    else:
        description = known.get("description") or f"{folder} · SRT · {duration:.1f} с"
        if item["srt_path"].parent != item["video"].parent and not known:
            description += f" · SRT: {str(item['srt_path'].parent.relative_to(root)).replace(chr(92), '/')}"
        if yaw_offset:
            description += f" · курс откалиброван: {yaw_offset:+g}°"
    window = min(8.0, duration / 3) if duration else 0.0
    segments = (f"0-{window:.1f}, {duration / 2 - window / 2:.1f}-{duration / 2 + window / 2:.1f}, "
                f"{duration - window:.1f}-{duration:.1f}") if duration else ""
    start = float(known.get("start", 0.0))
    end = min(float(known.get("end", 15.0)), duration) if duration else float(known.get("end", 15.0))
    return {"id": item["asset"], "title": known.get("title") or item["label"],
            "description": description, "date": item["date"] or "—", "video_path": str(item["video"]),
            "srt": item["srt"], "log": item["log"], "available": available, "known": bool(known),
            "project_dir": str(root / "work" / item["asset"].lower()),
            "start_seconds": start, "end_seconds": end if end > start else start + 10.0, "duration": duration,
            "segments": segments, "resolution_m_per_px": float(known.get("resolution", 1.0)), "frame_interval": 30,
            "yaw_offset_deg": yaw_offset}


def scan_cases(root: Path, refresh: bool = False) -> list[dict]:
    """Cases found under ``root``; the scan of a large external disk is cached for a few minutes."""
    key = str(root)
    cached = _CACHE.get(key)
    if cached and not refresh and time.monotonic() - cached[0] < CACHE_SECONDS:
        return cached[1]
    cases = [_describe(item, root) for item in _scan(root, refresh)]
    # Usable first, then by date and name; a video that cannot be georeferenced stays visible but disabled.
    cases.sort(key=lambda c: (not c["available"], c["date"], c["id"], c["video_path"]))
    _CACHE[key] = (time.monotonic(), cases)
    return cases


def interesting_cases(root: Path | None = None, refresh: bool = False) -> dict:
    """Answer of ``/api/catalog``: the data folder, its state and the cases found in it."""
    state = data_root_state() if root is None else {"path": Path(root), "source": "argument",
                                                    "available": Path(root).is_dir(), "suggestion": None}
    path = state["path"]
    result = {"root": str(path) if path else "", "work_dir": str(path / "work") if path else "",
              "connected": bool(state["available"]), "source": state["source"],
              "suggestion": str(state["suggestion"]) if state["suggestion"] else None, "cases": [], "total": 0}
    if state["available"]:
        cases = scan_cases(path, refresh)
        result.update(cases=cases[:MAX_CASES], total=len(cases), usable=sum(c["available"] for c in cases))
    return result


def choose_data_root(path: str) -> dict:
    """Check a folder picked in the UI, remember it and return its catalogue."""
    from .paths import set_data_root

    chosen = Path(str(path)).expanduser()
    if not chosen.is_dir():
        raise ValueError(f"Папка не найдена: {chosen}")
    cases = scan_cases(chosen.resolve(), refresh=True)
    if not cases:
        raise ValueError("В этой папке не найдено видео (MP4/MOV). Укажите папку, внутри которой — в любых подпапках — "
                         "лежат видео, их SRT с теми же именами и полётные логи DJIFlightRecord_*.txt.")
    set_data_root(chosen)
    return interesting_cases()
