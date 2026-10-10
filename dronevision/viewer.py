from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
from datetime import datetime, timezone

import numpy as np

from .discovery import match_log
from .paths import cache_dir, is_video, run_tool, tool
from .geometry import CAMERA_PROFILES, camera_profile, focal_length_px, horizontal_fov_deg
from .telemetry import interpolate, read_dji_log
from .video import probe


SRT_ROW = re.compile(
    r"(\d+)\s*\n(\d\d):(\d\d):(\d\d),(\d{3})\s+-->.*?"
    r"(.*?)"
    r"\[latitude : ([\d.-]+)\]\s*\[longtitude : ([\d.-]+)\]\s*\[altitude:\s*([\d.-]+)\]",
    re.S,
)
FOCAL_LENGTH = re.compile(r"\[focal_len\s*:\s*([\d.-]+)\]")
SRT_DATE = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d+),(\d+)")


def safe_media_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    # The server listens on 127.0.0.1 only, so any local video may be opened, not just DroneData.
    if not is_video(path):
        raise ValueError(f"Видеофайл не найден: {path}")
    return path


def _srt_series(path: Path):
    rows = []
    for _, hh, mm, ss, ms, metadata, lat, lon, alt in SRT_ROW.findall(path.read_text(encoding="utf-8", errors="replace")):
        time_s = int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000
        focal = FOCAL_LENGTH.search(metadata)
        rows.append((time_s, float(lat), float(lon), float(alt), float(focal.group(1)) if focal else 240.0))
    if not rows:
        raise ValueError("В SRT не найдена телеметрия")
    return np.asarray(rows, dtype=np.float64)


def _srt_start_epoch(path: Path) -> float:
    match = SRT_DATE.search(path.read_text(encoding="utf-8", errors="replace"))
    if not match:
        raise ValueError("В SRT не найдено абсолютное время первого кадра")
    value = datetime.strptime(f"{match.group(1)}.{match.group(2)}{match.group(3)}", "%Y-%m-%d %H:%M:%S.%f")
    return value.replace(tzinfo=timezone.utc).timestamp()


def viewer_data(video_path: str, max_points: int = 2400) -> dict:
    video = safe_media_path(video_path)
    meta = probe(video)
    matched = match_log(str(video))
    srt = _srt_series(Path(matched["srt_path"]))
    count = min(max_points, max(2, round(meta["duration"] * 10)))
    timeline = np.linspace(0, meta["duration"], count)
    log = read_dji_log(matched["log_path"], matched["utc_offset_hours"], matched["clock_offset_seconds"])
    srt_start_epoch = _srt_start_epoch(Path(matched["srt_path"]))
    epoch = srt_start_epoch + timeline
    values = interpolate(log, epoch)
    # SRT writes 0/0 while GPS is lost; take the track from the log there (gaps are dead-reckoned).
    srt_fix = (np.abs(srt[:, 1]) > 1e-3) & (np.abs(srt[:, 2]) > 1e-3)
    latitude = np.interp(timeline, srt[srt_fix, 0], srt[srt_fix, 1]) if srt_fix.any() else values["lat"]
    longitude = np.interp(timeline, srt[srt_fix, 0], srt[srt_fix, 2]) if srt_fix.any() else values["lon"]
    lost = np.interp(timeline, srt[:, 0], (~srt_fix).astype(float)) > 0
    latitude = np.where(lost, values["lat"], latitude)
    longitude = np.where(lost, values["lon"], longitude)
    north = (latitude - latitude[0]) * 111_000
    east = (longitude - longitude[0]) * 111_000 * np.cos(np.deg2rad(float(latitude.mean())))
    focal_length_mm = np.interp(timeline, srt[:, 0], srt[:, 4]) / 10.0
    # Zoom relative to the camera's base focal length: a video may never be recorded without zoom.
    profile = camera_profile(srt[:, 4] / 10.0)
    base_focal_mm = CAMERA_PROFILES[profile]["base_focal_mm"]
    hfov = [horizontal_fov_deg(meta["width"], focal_length_px(meta["width"], f, profile)) for f in focal_length_mm]
    gimbal_relative_yaw = (values["yaw_2"] - values["yaw"] + 180.0) % 360.0 - 180.0
    return {
        "video": meta,
        "video_path": str(video),
        "log_path": matched["log_path"],
        "srt_path": matched["srt_path"],
        "camera": {"profile": profile, "label": CAMERA_PROFILES[profile]["label"], "base_focal_mm": base_focal_mm},
        # Constant gimbal pitch/roll corrections of this video (<video>.pose.json, see pose.pose_offsets).
        "pose_offsets": _pose_offsets(video),
        "sync": {"hours": matched["utc_offset_hours"], "seconds": matched["clock_offset_seconds"],
                 "gps_error_m": matched["spatial_error_m"],
                 # Hovering: the GPS track does not fix the shift (see discovery._hover_sync).
                 "ambiguous": bool(matched.get("sync_ambiguous")),
                 "range_seconds": matched.get("clock_offset_range_seconds"),
                 "note": matched.get("sync_note"),
                 "image_extra_seconds": matched.get("image_sync_extra_s"),
                 "srt_start_correction_seconds": round(srt_start_epoch - meta["creation_time"], 3)},
        "series": {
            "time": timeline.round(3).tolist(),
            "latitude": latitude.tolist(), "longitude": longitude.tolist(),
            "position_east": east.tolist(), "position_north": north.tolist(),
            "altitude_srt": np.interp(timeline, srt[:, 0], srt[:, 3]).tolist(),
            "focal_length_mm": focal_length_mm.tolist(),
            "zoom": (focal_length_mm / base_focal_mm).tolist(), "horizontal_fov_deg": hfov,
            "height": values["hgt"].tolist(),
            "drone_yaw": values["yaw"].tolist(), "drone_pitch": values["pit"].tolist(),
            "drone_roll": values["rol"].tolist(), "gimbal_yaw": values["yaw_2"].tolist(),
            "gimbal_relative_yaw": gimbal_relative_yaw.tolist(),
            "gimbal_pitch": values["pit_2"].tolist(), "gimbal_roll": values["rol_2"].tolist(),
        },
    }


def proxy_path(video_path: str, height: int = 720) -> Path:
    source = safe_media_path(video_path)
    stat = source.stat()
    key = hashlib.sha1(f"{source}:{stat.st_size}:{stat.st_mtime_ns}:{height}".encode()).hexdigest()[:20]
    return cache_dir("dronevision-preview-cache") / f"{key}_{height}p.mp4"


def ensure_proxy(video_path: str, height: int = 720) -> Path:
    source = safe_media_path(video_path)
    target = proxy_path(str(source), height)
    if target.is_file() and target.stat().st_size > 1024:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".part.mp4")
    command = [tool("ffmpeg"), "-y", "-v", "error", "-i", str(source), "-map", "0:v:0", "-vf", f"scale=-2:{height}",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "27", "-g", "30", "-keyint_min", "30",
               "-an", "-movflags", "+faststart", str(temporary)]
    result = run_tool(command)
    if result.returncode:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Не удалось создать proxy: {result.stderr.strip()}")
    os.replace(temporary, target)
    return target


def _pose_offsets(video: Path) -> dict:
    from .pose import pose_offsets
    return pose_offsets(video)
