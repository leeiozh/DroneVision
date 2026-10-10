from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import re

import numpy as np

from .telemetry import ensure_json_log


SRT_POINT = re.compile(
    r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+,\d+).*?"
    r"\[latitude : ([\d.-]+)\] \[longtitude : ([\d.-]+)\]", re.S)


def _srt_points(path: Path) -> np.ndarray:
    points = []
    for stamp, lat, lon in SRT_POINT.findall(path.read_text(encoding="utf-8", errors="replace")):
        stamp = stamp.replace(",", ".", 1).rsplit(",", 1)[0]
        value = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
        points.append((value, float(lat), float(lon)))
    if not points:
        raise ValueError(f"В SRT не найдены GPS и время: {path}")
    return np.asarray(points, dtype=np.float64)


def _find_near(video: Path, name: str | None = None, directory: str | None = None) -> list[Path]:
    if directory:
        root = Path(directory).expanduser()
        return list(root.glob(name or "*")) if root.is_dir() else []
    roots = [video.parent, *list(video.parents)[:5]]
    found = []
    for root in roots:
        patterns = [name] if name else ["FlightRecords/*.txt", "logs/*.json", "*.txt", "*.json"]
        for pattern in patterns:
            found.extend(root.glob(pattern))
            found.extend(root.glob(f"*/{pattern}"))
    return sorted(set(p.resolve() for p in found if p.is_file()))


def _dataset_root(video: Path) -> Path | None:
    for parent in video.parents:
        if (parent / "telemetry").is_dir() and (parent / "media").is_dir():
            return parent
    return None


def _stat_key(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns


@lru_cache(maxsize=64)
def _log_track(path: str, size: int, mtime_ns: int) -> np.ndarray:
    """(time, latitude, longitude) of every FlightRecord sample; parsing a log takes ~0.15 s."""
    frames = json.loads(ensure_json_log(path).read_text(encoding="utf-8"))["frames"]
    return np.asarray([(datetime.fromisoformat(f["custom"]["dateTime"].replace("Z", "+00:00")).timestamp(),
                        f["osd"]["latitude"], f["osd"]["longitude"]) for f in frames], dtype=np.float64)


_MATCH_CACHE: dict[tuple, dict] = {}
MAX_CLOCK_OFFSET_S = 3.0     # a larger GPS time shift is not believed (see _hover_sync)
SYNC_PLATEAU_S = 1.0         # shifts fitting the track equally well span more than this -> shift unobservable


def match_log(video_path: str, log_directory: str | None = None) -> dict:
    """Match by the SRT GPS trajectory, then estimate the nominal log time shift.

    Results are cached per video/log file state: viewer, projection and calibration call this repeatedly.
    """
    video = Path(video_path).expanduser().resolve()
    sidecar = sync_sidecar(video)
    key = (_stat_key(video), log_directory, _stat_key(sidecar) if sidecar.is_file() else None)
    if key in _MATCH_CACHE:
        return dict(_MATCH_CACHE[key])
    result = _match_log(video, log_directory)
    if sidecar.is_file():
        # Extra shift measured from image rotation (``check-attitude --save``): GPS matching is weak while hovering.
        extra = float(json.loads(sidecar.read_text(encoding="utf-8")).get("extra_log_shift_s", 0.0))
        result["gps_clock_offset_seconds"] = result["clock_offset_seconds"]
        result["clock_offset_seconds"] = round(result["clock_offset_seconds"] + extra, 3)
        result["image_sync_extra_s"] = extra
    _MATCH_CACHE[key] = result
    return dict(result)


def sync_sidecar(video: Path) -> Path:
    """``<video>.sync.json`` next to the video: per-video log time correction found from the image."""
    return video.with_name(video.stem + ".sync.json")


def _match_log(video: Path, log_directory: str | None) -> dict:
    asset_match = re.search(r"DJI_\d{4}", video.name, re.I)
    asset_id = asset_match.group(0).upper() if asset_match else video.stem
    srt_name = f"{asset_id}.SRT"
    dataset = _dataset_root(video)
    srt_candidates = list((dataset / "telemetry" / "srt").glob(srt_name)) if dataset else []
    if not srt_candidates:
        srt_candidates = _find_near(video, srt_name)
    if not srt_candidates:
        # Nothing next to the video: pair by name anywhere in the data folder (catalog.py).
        from .catalog import find_srt
        far = find_srt(video)
        srt_candidates = [far] if far else []
    if not srt_candidates:
        raise ValueError(f"Не найден {srt_name}; автоматическое пространственное сопоставление невозможно")
    srt = srt_candidates[0]
    srt_points = _srt_points(srt)
    srt_time = float(np.median(srt_points[:, 0]))
    srt_points = srt_points[::max(1, len(srt_points) // 300)]
    if log_directory:
        logs = _find_near(video, directory=log_directory)
    elif dataset:
        logs = list((dataset / "telemetry" / "flight_records").glob("*"))
    else:
        logs = _find_near(video)
    logs = [p for p in logs if p.suffix.lower() in {".txt", ".json"} and "FlightRecord" in p.name]
    if not log_directory:
        # Also every log of the data folder dated within a day of the recording (catalog.py): the logs lying
        # next to the video may belong to other flights, the right one may be kept elsewhere.
        from .catalog import find_logs
        known = {p.resolve() for p in logs}
        logs += [p for p in find_logs(video, datetime.fromtimestamp(srt_time, timezone.utc).strftime("%Y-%m-%d"))
                 if p.resolve() not in known]
    if not logs:
        raise ValueError("Не найдены полётные логи DJIFlightRecord ни рядом с видео, ни в папке данных")
    lat_ref = float(np.median(srt_points[:, 1])); candidates = []
    for source in logs:
        try:
            track = _log_track(*_stat_key(source.resolve()))
            coordinates = track[::max(1, len(track) // 500), 1:]
            valid = np.isfinite(coordinates).all(axis=1) & (np.abs(coordinates[:, 0]) > 1)
            coordinates = coordinates[valid]
            if not len(coordinates):
                continue
            # Distance from each SRT sample to the nearest point of this flight.
            errors = []
            for _, lat, lon in srt_points[::10]:
                d = np.hypot((coordinates[:, 0] - lat) * 111_000,
                             (coordinates[:, 1] - lon) * 111_000 * np.cos(np.deg2rad(lat_ref)))
                errors.append(float(d.min()))
            candidates.append((float(np.median(errors)), source, track))
        except Exception:
            continue
    if not candidates:
        raise ValueError("Не удалось прочитать полётные логи")
    # The same place is often flown on several days: prefer flights whose time fits the recording
    # (camera and log clocks differ by a time zone at most).  With a wrong camera date none fits: use all.
    in_time = [c for c in candidates if c[2][0, 0] - 14 * 3600 <= srt_time <= c[2][-1, 0] + 14 * 3600]
    candidates = in_time or candidates
    spatial_error, source, track = min(candidates, key=lambda item: item[0])
    log_points = track[::max(1, len(track) // 1200)]
    # First infer the whole-hour timezone mismatch, then refine it by comparing
    # the complete GPS trajectories. This avoids false matches while hovering.
    rough_delta = float(np.median(srt_points[:, 0]) - np.median(log_points[:, 0]))
    hours = round(rough_delta / 3600)
    best = (float("inf"), hours * 3600.0)
    scan = []                                   # (offset within the hour, rms of the position error)
    for total_offset in np.arange(hours * 3600 - 15, hours * 3600 + 15.001, 0.05):
        log_times = srt_points[:, 0] - total_offset
        valid = (log_times >= log_points[0, 0]) & (log_times <= log_points[-1, 0])
        if valid.sum() < 10:
            continue
        lat = np.interp(log_times[valid], log_points[:, 0], log_points[:, 1])
        lon = np.interp(log_times[valid], log_points[:, 0], log_points[:, 2])
        errors = np.hypot((lat - srt_points[valid, 1]) * 111_000,
                          (lon - srt_points[valid, 2]) * 111_000 * np.cos(np.deg2rad(lat_ref)))
        score = float(np.median(errors))
        scan.append((total_offset - hours * 3600, float(np.sqrt(np.mean(np.minimum(errors, 50.0) ** 2)))))
        if score < best[0]:
            best = score, total_offset
    total_offset = best[1]
    seconds = float(round(float(total_offset - hours * 3600), 2))
    result = {"log_path": str(source), "srt_path": str(srt), "utc_offset_hours": hours,
              "clock_offset_seconds": seconds, "spatial_error_m": round(spatial_error, 2),
              "confidence": "high" if spatial_error < 5 else "low"}
    result.update(_hover_sync(scan, seconds))
    return result


def _hover_sync(scan: list[tuple[float, float]], seconds: float) -> dict:
    """Accept the GPS time shift only when the track really determines it; otherwise use 0.

    While the drone hovers every shift fits the track equally well and the minimum of the median error is
    noise: DJI_0465 got +9.25 s (true −0.65 s by a gimbal step seen in the image), which put the gimbal
    tilt made before the recording into the first 9 s of the video.  All shifts whose rms position error is
    within 0.1 m of the best one form a plateau.  The shift is trusted when the plateau is at most
    ``SYNC_PLATEAU_S`` wide and the shift itself is at most ``MAX_CLOCK_OFFSET_S`` (camera and log clocks
    differ by < 1.5 s in every flight where the shift is observable); otherwise it is 0.
    A measured correction can still be given in ``<video>.sync.json`` (``check-attitude --save``).
    """
    if not scan:
        return {}
    offsets, rms = np.asarray(scan).T
    centre = int(rms.argmin())
    inside = rms <= np.hypot(rms[centre], 0.1)
    first = last = centre
    while first > 0 and inside[first - 1]:
        first -= 1
    while last < len(offsets) - 1 and inside[last + 1]:
        last += 1
    low, high = round(float(offsets[first]), 2), round(float(offsets[last]), 2)
    hovering = high - low > SYNC_PLATEAU_S
    if not hovering and abs(seconds) <= MAX_CLOCK_OFFSET_S:
        return {"sync_ambiguous": False}
    return {"clock_offset_seconds": 0.0, "sync_ambiguous": True, "gps_best_clock_offset_seconds": seconds,
            "clock_offset_range_seconds": [low, high],
            "sync_note": "висение: сдвиг лога по GPS не определяется, принят 0" if hovering else
                         f"сдвиг по GPS {seconds:+.2f} с больше допустимых {MAX_CLOCK_OFFSET_S:g} с, принят 0"}
