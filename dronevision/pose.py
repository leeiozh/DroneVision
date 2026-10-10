"""Per-frame camera pose from SRT and FlightRecord, shared by projection, mosaic and calibration."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .discovery import match_log
from .geometry import CAMERA_PROFILES, camera_profile, focal_length_px, horizontal_fov_deg
from .telemetry import interpolate, read_dji_log
from .video import probe
from .viewer import _srt_series, _srt_start_epoch


@dataclass(frozen=True)
class VideoContext:
    video: Path
    meta: dict
    srt_path: Path | None
    srt: np.ndarray | None          # time_s, lat, lon, altitude, focal_len (mm x 10)
    start_epoch: float              # clock of the first frame, same clock as the corrected log
    log_path: str
    utc_offset_hours: float
    clock_offset_seconds: float
    log: dict
    profile: str
    takeoff: tuple[float, float]


_CONTEXTS: dict[tuple, VideoContext] = {}


def load_context(video_path: str | Path, log_path: str | None = None, utc_offset_hours: float | None = None,
                 clock_offset_seconds: float | None = None) -> VideoContext:
    """Video metadata, matched SRT/log and the takeoff reference; cached per file state."""
    video = Path(video_path).expanduser().resolve()
    stat = video.stat()
    key = (str(video), stat.st_size, stat.st_mtime_ns, log_path, utc_offset_hours, clock_offset_seconds)
    if key in _CONTEXTS:
        return _CONTEXTS[key]
    meta = probe(video)
    try:
        matched = match_log(str(video))
    except ValueError:
        if not log_path:
            raise
        matched = {}
    srt_path = Path(matched["srt_path"]) if matched.get("srt_path") else None
    log_path = log_path or matched["log_path"]
    hours = matched.get("utc_offset_hours", 0) if utc_offset_hours is None else utc_offset_hours
    seconds = matched.get("clock_offset_seconds", 0.0) if clock_offset_seconds is None else clock_offset_seconds
    log = read_dji_log(log_path, hours, seconds)
    if srt_path is not None:
        srt = _srt_series(srt_path)
        # SRT holds the frame clock with milliseconds; MP4 creation_time is truncated to whole seconds.
        start_epoch = _srt_start_epoch(srt_path)
        profile = camera_profile(srt[:, 4] / 10)
    else:
        srt, start_epoch, profile = None, meta["creation_time"], "mavic2_zoom"
    valid = np.isfinite(log["lat"]) & np.isfinite(log["lon"]) & (np.abs(log["lat"]) > 1)
    candidates = np.flatnonzero(valid & (log["time"] <= start_epoch))
    if not len(candidates):
        candidates = np.flatnonzero(valid)
    if not len(candidates):
        raise ValueError("В DJI-логе нет валидных GPS-координат")
    on_ground = candidates[np.abs(log["hgt"][candidates]) <= 1.0]
    takeoff_index = int(on_ground[-1] if len(on_ground) else candidates[0])
    context = VideoContext(video, meta, srt_path, srt, start_epoch, str(log_path), float(hours), float(seconds), log,
                           profile, (float(log["lat"][takeoff_index]), float(log["lon"][takeoff_index])))
    if len(_CONTEXTS) > 16:
        _CONTEXTS.clear()
    _CONTEXTS[key] = context
    return context


def enu_offsets(geod, lat0: float, lon0: float, lat, lon) -> np.ndarray:
    lat = np.atleast_1d(np.asarray(lat, dtype=np.float64)); lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
    azimuth, _, distance = geod.inv(np.full(lat.shape, lon0), np.full(lat.shape, lat0), lon, lat)
    angle = np.deg2rad(azimuth)
    return np.stack([distance * np.sin(angle), distance * np.cos(angle)], axis=-1)


def pose_sidecar(video: Path) -> Path:
    """``<video>.pose.json`` next to the video: constant gimbal pitch/roll corrections found from the image."""
    return Path(video).with_name(Path(video).stem + ".pose.json")


def pose_offsets(video: Path) -> dict:
    """Pitch and roll offsets (degrees) added to the logged gimbal angles of this video.

    The absolute gimbal pitch of the log can be off by a degree or more (DJI_0531: -5.7° logged, -7.0° by the
    sea horizon), which changes every ground range.  The offsets are measured by ``horizon`` / ``coastview``
    and stored per video; the heading offset stays a parameter of the request, as before.
    """
    try:
        raw = json.loads(pose_sidecar(video).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {"pitch_offset_deg": 0.0, "roll_offset_deg": 0.0}
    result = {key: float(raw.get(key) or 0.0) for key in ("pitch_offset_deg", "roll_offset_deg")}
    result.update({key: raw[key] for key in ("method", "measured_at_s", "yaw_offset_deg", "note") if key in raw})
    return result


def save_pose_offsets(video: Path, pitch_offset_deg: float, roll_offset_deg: float, **extra) -> dict:
    data = {"pitch_offset_deg": round(float(pitch_offset_deg), 4), "roll_offset_deg": round(float(roll_offset_deg), 4),
            **extra}
    sidecar = pose_sidecar(Path(video).expanduser().resolve())
    if not data["pitch_offset_deg"] and not data["roll_offset_deg"]:
        sidecar.unlink(missing_ok=True)
    else:
        sidecar.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": str(sidecar), **data}


def frame_poses(context: VideoContext, times_s, geod, yaw_offset_deg: float = 0.0, takeoff_height_m: float = 0.0,
                base_horizontal_fov_deg: float = 0.0) -> dict[str, np.ndarray]:
    """Camera pose for video times (seconds from the first frame).

    ``pitch`` and ``roll`` include the per-video offsets of ``pose_offsets``; the logged values stay in
    ``log_pitch`` / ``log_roll``.
    """
    offsets = pose_offsets(context.video)
    times_s = np.atleast_1d(np.asarray(times_s, dtype=np.float64))
    telemetry = interpolate(context.log, context.start_epoch + times_s)
    if context.srt is not None:
        focal_mm = np.interp(times_s, context.srt[:, 0], context.srt[:, 4]) / 10
    else:
        focal_mm = np.full(times_s.shape, CAMERA_PROFILES[context.profile]["base_focal_mm"])
    width = context.meta["width"]
    focal_px = np.array([focal_length_px(width, f, context.profile, base_horizontal_fov_deg) for f in focal_mm])
    altitude = telemetry["hgt"] + takeoff_height_m
    if np.any(altitude <= 0):
        raise ValueError("Высота камеры над водой должна быть положительной")
    return {
        "time_s": times_s, "epoch": context.start_epoch + times_s,
        "lat": telemetry["lat"], "lon": telemetry["lon"],
        "offset": enu_offsets(geod, *context.takeoff, telemetry["lat"], telemetry["lon"]),
        "altitude": altitude, "flight_altitude": telemetry["hgt"],
        # Aircraft heading: gimbal.yaw uses an arbitrary reference on Mavic 2 (see geometry.py).
        "yaw": (telemetry["yaw"] + yaw_offset_deg) % 360, "log_yaw": telemetry["yaw"] % 360,
        "gimbal_yaw": telemetry["yaw_2"] % 360,
        "pitch": telemetry["pit_2"] + offsets["pitch_offset_deg"], "roll": telemetry["rol_2"] + offsets["roll_offset_deg"],
        "log_pitch": telemetry["pit_2"], "log_roll": telemetry["rol_2"],
        "body_pitch": telemetry["pit"], "body_roll": telemetry["rol"],
        "focal_mm": focal_mm, "zoom": focal_mm / CAMERA_PROFILES[context.profile]["base_focal_mm"],
        "focal_px": focal_px, "horizontal_fov_deg": np.array([horizontal_fov_deg(width, f) for f in focal_px]),
        # GPS lost (zeros in the log): position is dead-reckoned, see telemetry._fill_gps_gaps.
        "gps_valid": telemetry.get("gps_valid", np.ones(times_s.shape)) > 0.999,
        "gps_uncertainty_m": telemetry.get("gps_uncertainty_m", np.zeros(times_s.shape)),
    }
