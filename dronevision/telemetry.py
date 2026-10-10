from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import hashlib
import tempfile

import numpy as np

from .paths import run_tool, tool


FIELDS = ("lat", "lon", "hgt", "pit", "rol", "yaw", "pit_2", "rol_2", "yaw_2")


def ensure_json_log(path: str | Path) -> Path:
    """Convert a raw DJI FlightRecord TXT to cached JSON when necessary."""
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() == ".json":
        return source
    if source.suffix.lower() != ".txt":
        raise ValueError("Поддерживаются DJI FlightRecord .txt и декодированный .json")
    stat = source.stat()
    key = hashlib.sha1(f"{source}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:16]
    target = Path(tempfile.gettempdir()) / "dronevision-log-cache" / f"{key}.json"
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    result = run_tool([tool("dji-log"), "-o", str(target), str(source)])
    if result.returncode or not target.is_file():
        raise RuntimeError(f"Не удалось декодировать DJI-лог: {result.stderr.strip()}")
    return target


def _parse_utc(text: str) -> datetime:
    # DJI dateTime ends in Z; keep it timezone-aware to avoid host-dependent timestamps.
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def _regular_clock(stamps: np.ndarray, fly_time: np.ndarray) -> np.ndarray:
    """Replace jittery ``dateTime`` by evenly advancing ``flyTime`` anchored to it.

    The app stamps records with ~70 ms jitter and ~6 % duplicated milliseconds, while flyTime
    advances by exactly 0.1 s per record. Dropping duplicates shifted attitude samples in time.
    """
    clock = stamps.copy()
    if not np.all(np.isfinite(fly_time)):
        return clock
    for segment in np.split(np.arange(len(stamps)), np.flatnonzero(np.diff(fly_time) <= 0) + 1):
        if len(segment) < 10:
            continue
        anchored = fly_time[segment] + np.median(stamps[segment] - fly_time[segment])
        if np.percentile(np.abs(anchored - stamps[segment]), 95) < 0.5:
            clock[segment] = anchored
    return clock if np.all(np.diff(clock) >= 0) else stamps


def _fill_gps_gaps(out: dict[str, np.ndarray], speed: np.ndarray) -> None:
    """Replace lost GPS fixes (DJI writes latitude = longitude = 0 in Atti mode) in place.

    Inside a gap the position is dead-reckoned from osd.xSpeed/ySpeed (north/east, checked against GPS
    on DJI_0337: slope 1.01, r = 0.998) and the closure error to the next fix is spread linearly in time.
    Speeds are often zero in Atti mode, so the closure error can reach hundreds of metres (DJI_0337: 450 m
    over 101 s); it is kept as ``gps_uncertainty_m`` (NaN when the gap has no fix after it).
    """
    lat, lon, time = out["lat"], out["lon"], out["time"]
    bad = ~(np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) > 1e-3) & (np.abs(lon) > 1e-3))
    out["gps_valid"] = (~bad).astype(np.float64)
    out["gps_uncertainty_m"] = np.zeros(len(lat))
    if not bad.any() or bad.all():
        return
    edges = np.flatnonzero(np.diff(np.r_[0, bad.astype(np.int8), 0]))
    for start, stop in zip(edges[::2], edges[1::2]):          # bad samples start..stop-1
        anchor = start - 1 if start > 0 else stop              # a valid neighbour
        forward = start > 0
        index = np.arange(start, stop)
        scale_north = 111_320.0
        scale_east = 111_320.0 * np.cos(np.deg2rad(lat[anchor]))
        dt = np.diff(time[anchor:stop] if forward else time[start:stop + 1])
        if forward:
            north = np.cumsum(speed[start - 1:stop - 1, 0] * dt)
            east = np.cumsum(speed[start - 1:stop - 1, 1] * dt)
        else:   # gap at the beginning: integrate backwards from the first fix
            north = -np.cumsum((speed[start:stop, 0] * dt)[::-1])[::-1]
            east = -np.cumsum((speed[start:stop, 1] * dt)[::-1])[::-1]
        uncertainty = np.full(len(index), np.nan)
        if forward and stop < len(lat):
            span = time[stop] - time[start - 1]
            closure_n = (lat[stop] - lat[anchor]) * scale_north - (north[-1] + speed[stop - 1, 0] * (time[stop] - time[stop - 1]))
            closure_e = (lon[stop] - lon[anchor]) * scale_east - (east[-1] + speed[stop - 1, 1] * (time[stop] - time[stop - 1]))
            fraction = (time[index] - time[start - 1]) / max(span, 1e-6)
            north = north + closure_n * fraction
            east = east + closure_e * fraction
            uncertainty = np.hypot(closure_n, closure_e) * 4 * fraction * (1 - fraction)
        lat[index] = lat[anchor] + north / scale_north
        lon[index] = lon[anchor] + east / scale_east
        out["gps_uncertainty_m"][index] = uncertainty


@lru_cache(maxsize=8)
def _load_log(path: str, size: int, mtime_ns: int) -> dict[str, np.ndarray]:
    with ensure_json_log(path).open(encoding="utf-8") as stream:
        frames = json.load(stream)["frames"]
    if len(frames) < 2:
        raise ValueError("DJI-лог содержит меньше двух кадров")
    out = {key: np.empty(len(frames), dtype=np.float64) for key in ("time",) + FIELDS}
    fly_time = np.empty(len(frames), dtype=np.float64)
    speed = np.zeros((len(frames), 2), dtype=np.float64)     # osd.xSpeed (north), osd.ySpeed (east), m/s
    for i, frame in enumerate(frames):
        out["time"][i] = _parse_utc(frame["custom"]["dateTime"]).timestamp()
        osd, gimbal = frame["osd"], frame["gimbal"]
        fly_time[i] = np.nan if osd.get("flyTime") is None else osd["flyTime"]
        out["lat"][i], out["lon"][i], out["hgt"][i] = osd["latitude"], osd["longitude"], osd["height"]
        out["pit"][i], out["rol"][i], out["yaw"][i] = osd["pitch"], osd["roll"], osd["yaw"]
        out["pit_2"][i], out["rol_2"][i], out["yaw_2"][i] = gimbal["pitch"], gimbal["roll"], gimbal["yaw"]
        speed[i] = osd.get("xSpeed") or 0.0, osd.get("ySpeed") or 0.0
    if np.any(np.diff(out["time"]) < 0):
        raise ValueError("Временная шкала DJI-лога не возрастает")
    out["time"] = _regular_clock(out["time"], fly_time)
    _fill_gps_gaps(out, speed)
    # Without flyTime the raw stamps may still contain several samples with the same millisecond.
    _, unique_indices = np.unique(out["time"], return_index=True)
    if len(unique_indices) != len(out["time"]):
        out = {key: values[unique_indices] for key, values in out.items()}
    return out


def read_dji_log(path: str | Path, utc_offset_hours: float = 0.0,
                 clock_offset_seconds: float = 0.0) -> dict[str, np.ndarray]:
    source = Path(path).expanduser().resolve()
    stat = source.stat()
    cached = _load_log(str(source), stat.st_size, stat.st_mtime_ns)
    out = {key: values.copy() for key, values in cached.items()}
    out["time"] += utc_offset_hours * 3600 + clock_offset_seconds
    return out


def interpolate(log: dict[str, np.ndarray], times: np.ndarray) -> dict[str, np.ndarray]:
    if times[0] < log["time"][0] or times[-1] > log["time"][-1]:
        raise ValueError("Выбранный фрагмент видео не полностью покрывается DJI-логом")
    result = {}
    for key, values in log.items():
        if key == "time":
            continue
        if key in {"pit", "rol", "yaw", "pit_2", "rol_2", "yaw_2"}:
            radians = np.unwrap(np.deg2rad(values))
            result[key] = np.rad2deg(np.interp(times, log["time"], radians))
        else:
            result[key] = np.interp(times, log["time"], values)
    return result
