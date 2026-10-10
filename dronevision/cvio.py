"""OpenCV file access that works with non-ASCII paths on Windows.

``cv2.imread``/``cv2.imwrite`` open files through the ANSI code page there and fail silently on
``C:\\Users\\Иван\\...`` or ``DJI_0402 ВВ.MP4``; Python's own file access has no such limit.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def imread(path, flags: int = 1):       # 1 = cv2.IMREAD_COLOR; cv2 is imported lazily as elsewhere
    import cv2
    try:
        data = np.fromfile(os.fspath(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, flags) if data.size else None


def imwrite(path, image, params=()) -> bool:
    import cv2
    path = Path(path)
    ok, encoded = cv2.imencode(path.suffix, image, list(params))
    if not ok:
        return False
    try:
        encoded.tofile(os.fspath(path))
    except OSError:
        return False
    return True


def _short_path(path: str) -> str:
    """8.3 alias of a Windows path (pure ASCII) or the path itself when there is none."""
    try:
        import ctypes
        buffer = ctypes.create_unicode_buffer(32768)
        length = ctypes.windll.kernel32.GetShortPathNameW(path, buffer, len(buffer))
        return buffer.value if 0 < length < len(buffer) else path
    except Exception:
        return path


def open_video(path):
    import cv2
    text = os.fspath(path)
    capture = cv2.VideoCapture(text)
    if not capture.isOpened() and os.name == "nt" and not text.isascii():
        capture = cv2.VideoCapture(_short_path(text))
    return capture
