from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from .paths import run_tool, tool


def probe(path: str | Path) -> dict:
    command = [tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration:format_tags=creation_time",
               "-of", "json", str(path)]
    result = run_tool(command)
    if result.returncode:
        raise ValueError(f"ffprobe не смог прочитать видео: {result.stderr.strip()}")
    raw = json.loads(result.stdout)
    stream = raw["streams"][0]
    numerator, denominator = map(float, stream["avg_frame_rate"].split("/"))
    creation = raw.get("format", {}).get("tags", {}).get("creation_time")
    if not creation:
        raise ValueError("В метаданных видео отсутствует creation_time")
    created = datetime.fromisoformat(creation.replace("Z", "+00:00")).astimezone(timezone.utc)
    return {
        "width": int(stream["width"]), "height": int(stream["height"]),
        "fps": numerator / denominator, "frames": int(stream.get("nb_frames") or 0),
        "duration": float(stream["duration"]), "creation_time": created.timestamp(),
        "creation_time_iso": created.isoformat(),
    }


def quick_duration(path: str | Path) -> float | None:
    """Duration of an MP4/MOV read from its ``mvhd`` box without ffprobe (a few seeks); None if unknown.

    Used to check a video/SRT pair found by name only, for every video of a folder scan.
    """
    import struct

    def boxes(stream, start: int, end: int):
        position = start
        while position + 8 <= end:
            stream.seek(position)
            header = stream.read(8)
            if len(header) < 8:
                return
            size, kind = struct.unpack(">I4s", header)
            body = position + 8
            if size == 1:
                size = struct.unpack(">Q", stream.read(8))[0]
                body += 8
            elif size == 0:
                size = end - position
            if size < 8:
                return
            yield kind, body, position + size
            position += size

    try:
        path = Path(path)
        total = path.stat().st_size
        with path.open("rb") as stream:
            for kind, body, stop in boxes(stream, 0, total):
                if kind != b"moov":
                    continue
                for inner, inner_body, _ in boxes(stream, body, stop):
                    if inner != b"mvhd":
                        continue
                    stream.seek(inner_body)
                    data = stream.read(32)
                    if data[0] == 1:
                        timescale, duration = struct.unpack(">IQ", data[20:32])
                    else:
                        timescale, duration = struct.unpack(">II", data[12:20])
                    return duration / timescale if timescale else None
    except (OSError, struct.error, IndexError):
        return None
    return None
