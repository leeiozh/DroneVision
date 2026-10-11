"""Names of the saved results, the same for every tool:

    <flight start>_<video name>_<video start>_<first time>s_<last time>s_<type>[_<n>].<ext>

* flight start — the first record of the matched FlightRecord (takeoff), UTC, ``YYYYMMDDTHHMMSS``;
* video start — the clock of the first frame (SRT, else MP4 creation_time), UTC, ``YYYYMMDDTHHMMSS``;
* first / last time — seconds from the start of the video, to tenths (``0006.6s``): the first and the last frame, or the start of the
  first interval and the end of the last one (equal for a single frame);
* type — ``projection``, ``compare1`` / ``compare2`` (two frames or two averaged intervals on one canvas),
  ``mosaic``, ``average``, ``enhanced``, ``currents``, ``motion``; transects add ``lines`` / ``contours`` and the
  object name to the name of their image;
* ``_<n>`` (2, 3, ...) only when a result with this name already exists: nothing is overwritten.

Heading correction and log time shift are not part of the name: they are written to the metadata
(``*.geo.json``, result JSON, TXT headers), see ``metadata``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def flight_start_utc(context) -> datetime:
    """First record of the FlightRecord in its own (UTC) clock; ``context.log`` is shifted to the camera clock."""
    start = float(context.log["time"][0]) - context.utc_offset_hours * 3600 - context.clock_offset_seconds
    return datetime.fromtimestamp(start, timezone.utc)


def video_start_utc(context) -> datetime:
    """First frame of the video in UTC; ``context.start_epoch`` is the camera's local clock written as an epoch."""
    return datetime.fromtimestamp(context.start_epoch - context.utc_offset_hours * 3600, timezone.utc)


def video_prefix(context) -> str:
    """``<flight start>_<video name>_<video start>`` (both UTC): the beginning of every saved file name."""
    return (f"{flight_start_utc(context):%Y%m%dT%H%M%S}_{Path(context.video).stem}"
            f"_{video_start_utc(context):%Y%m%dT%H%M%S}")


def base_name(context, first_s: float, last_s: float) -> str:
    return f"{video_prefix(context)}_{float(first_s):06.1f}s_{float(last_s):06.1f}s"


def output_stem(folder: Path, context, first_s: float, last_s: float, kind: str,
                suffixes: tuple[str, ...] = (".png",)) -> str:
    """Stem for a new result in ``folder``, free for every suffix in ``suffixes``."""
    stem = candidate = f"{base_name(context, first_s, last_s)}_{kind}"
    number = 1
    while any((Path(folder) / f"{candidate}{suffix}").exists() for suffix in suffixes):
        number += 1
        candidate = f"{stem}_{number}"
    return candidate


def metadata(context, yaw_offset_deg: float | None = None) -> dict:
    """What used to be in the file names, plus the log synchronisation, for sidecars and result files."""
    result = {"video_start_utc": video_start_utc(context).isoformat(timespec="seconds"),
              "flight_start_utc": flight_start_utc(context).isoformat(timespec="seconds"),
              "flight_log": Path(context.log_path).name,
              "log_sync": {"utc_offset_hours": context.utc_offset_hours,
                           "clock_offset_seconds": context.clock_offset_seconds}}
    if yaw_offset_deg is not None:
        result["yaw_offset_deg"] = float(yaw_offset_deg)
    return result
