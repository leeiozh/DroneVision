from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from .paths import is_photo
from .paths import run_tool, tool


def _gps_degrees(values, reference: str) -> float:
    degrees, minutes, seconds = (float(value) for value in values)
    result = degrees + minutes / 60 + seconds / 3600
    return -result if reference.upper() in {"S", "W"} else result


def photo_probe(path: str | Path) -> dict:
    """Image dimensions and DJI EXIF needed to place a still photograph in its flight log."""
    try:
        from PIL import ExifTags, Image
    except ImportError as exc:
        raise RuntimeError("Для фотографий нужна библиотека Pillow; перезапустите установщик DroneVision") from exc
    source = Path(path)
    with Image.open(source) as image:
        width, height = image.size
        exif = image.getexif()
        details = exif.get_ifd(ExifTags.IFD.Exif)
        gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
    created_text = details.get(36867) or details.get(36868) or exif.get(306)
    if not created_text:
        raise ValueError("В EXIF фотографии отсутствует время съёмки")
    created = datetime.strptime(str(created_text), "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)
    try:
        latitude = _gps_degrees(gps[2], gps[1])
        longitude = _gps_degrees(gps[4], gps[3])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("В EXIF фотографии отсутствуют GPS-координаты") from exc
    focal_35mm = float(details.get(41989) or 28.0)
    return {
        "width": int(width), "height": int(height), "fps": 1.0, "frames": 1, "duration": 1.0,
        # EXIF has local camera time without a zone.  Like SRT, it is kept on a UTC-shaped timeline;
        # match_log determines the whole-hour difference from the actual UTC FlightRecord.
        "creation_time": created.timestamp(), "creation_time_iso": created.isoformat(),
        "codec": "jpeg", "profile": None, "pixel_format": "yuvj420p", "browser_playable": False,
        "is_photo": True, "latitude": latitude, "longitude": longitude,
        "focal_length_35mm": focal_35mm,
        "camera_make": str(exif.get(271) or ""), "camera_model": str(exif.get(272) or ""),
    }


def probe(path: str | Path) -> dict:
    if is_photo(Path(path)):
        return photo_probe(path)
    command = [tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration,codec_name,profile,pix_fmt:format_tags=creation_time",
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
    codec = str(stream.get("codec_name") or "").lower()
    pixel_format = str(stream.get("pix_fmt") or "").lower()
    # The Chromium bundled with the desktop app reliably plays the common web codecs, but not
    # DJI's HEVC/H.265 (and not professional camera codecs such as ProRes).  Tell the UI before it
    # tries the source so it can create a cached H.264 preview without first showing a decoder error.
    browser_playable = codec in {"h264", "vp8", "vp9", "av1"}
    if codec == "h264" and pixel_format and pixel_format not in {"yuv420p", "yuvj420p"}:
        browser_playable = False
    return {
        "width": int(stream["width"]), "height": int(stream["height"]),
        "fps": numerator / denominator, "frames": int(stream.get("nb_frames") or 0),
        "duration": float(stream["duration"]), "creation_time": created.timestamp(),
        "creation_time_iso": created.isoformat(),
        "codec": codec, "profile": stream.get("profile"), "pixel_format": pixel_format,
        "browser_playable": browser_playable, "is_photo": False,
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
