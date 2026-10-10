"""Machine-dependent locations: data set, coastline, caches and external programs.

Nothing here is tied to one computer.  Lookup order for every item: environment variable,
``dronevision_settings.json`` next to the package, then automatic search.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import string
import subprocess
import sys
import tempfile


APP_DIR = Path(__file__).resolve().parent.parent
BIN_DIR = APP_DIR / "bin"                      # ffmpeg.exe, ffprobe.exe, dji-log.exe on Windows
MAPS_DIR = APP_DIR / "maps"
SETTINGS_FILE = APP_DIR / "dronevision_settings.json"
# Written by the web UI («Указать папку с данными…»); per user, so the program folder may be read-only.
USER_SETTINGS_FILE = Path.home() / ".dronevision" / "settings.json"
COASTLINE_NAME = "coastline_33_34E_66_67N_land_polygons.shp"
IS_WINDOWS = os.name == "nt"
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def settings() -> dict:
    """Optional ``dronevision_settings.json``: {"data_root": "D:/DroneData", "coastline": "..."}.

    Backslashes are taken literally, so a path pasted from Explorer (``D:\\DroneData``) is accepted
    as it is; a BOM written by Notepad is ignored.
    """
    return _read_settings(SETTINGS_FILE)


def _read_settings(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return {}
    try:
        raw = json.loads(text)
    except ValueError as exc:
        # People often paste paths from Explorer into JSON literally: ``D:\DroneData`` or
        # ``\\server\share``.  Those are not valid JSON escapes.  Doubling every slash is the
        # correct repair for that raw form, including the two leading slashes of an UNC path.
        try:
            raw = json.loads(text.replace("\\", "\\\\"))
        except ValueError:
            raise ValueError(f"Ошибка в {path}: {exc}") from exc
    else:
        # Some raw Windows paths happen to be valid JSON because a folder starts with t/n/r/b/f;
        # json.loads then silently turns that part into a control character.  Retry as an Explorer
        # path only for the path-valued settings.  Properly escaped JSON never contains controls.
        if isinstance(raw, dict) and any(isinstance(raw.get(key), str) and any(ord(c) < 32 for c in raw[key])
                                         for key in ("data_root", "coastline")):
            try:
                raw = json.loads(text.replace("\\", "\\\\"))
            except ValueError:
                pass
    return raw if isinstance(raw, dict) else {}


def user_settings() -> dict:
    try:
        return _read_settings(USER_SETTINGS_FILE)
    except ValueError:
        return {}       # a damaged file written by the program is not worth stopping for


def _configured(env: str, key: str) -> Path | None:
    """Explicit choice: environment, then the folder picked in the UI, then the settings file."""
    return _configured_with_source(env, key)[0]


def _configured_with_source(env: str, key: str) -> tuple[Path | None, str]:
    for source, value in (("env", os.environ.get(env)), ("user", user_settings().get(key)),
                          ("file", settings().get(key))):
        if value:
            return Path(str(value)).expanduser(), source
    return None, ""


def _data_root_candidates() -> list[Path]:
    found = [APP_DIR.parent / "DroneData", Path.home() / "DroneData"]
    if IS_WINDOWS:
        found += [Path(f"{letter}:\\DroneData") for letter in string.ascii_uppercase[2:]]
    else:
        found.append(Path("/media/leeiozh/android/DroneData"))
        for base in (Path("/media"), Path("/run/media"), Path("/mnt"), Path("/Volumes")):
            found += sorted(base.glob("*/DroneData")) + sorted(base.glob("*/*/DroneData"))
    return found


def _detected_data_root() -> Path | None:
    for candidate in _data_root_candidates():
        try:
            if candidate.is_dir():
                return candidate
        except OSError:
            continue
    return None


def data_root_state() -> dict:
    """Where the data folder comes from and whether it can be used now.

    ``source``: "env", "user" (picked in the UI), "file" (settings file), "auto" (a folder named
    DroneData found in a usual place) or "none".  An explicitly chosen folder that is not available
    is never replaced silently: ``suggestion`` only offers what the automatic search found.
    """
    configured, source = _configured_with_source("DRONEVISION_DATA", "data_root")
    detected = _detected_data_root()
    if configured is not None:
        try:
            available = configured.is_dir()
        except OSError:
            available = False
        return {"path": configured, "source": source, "available": available,
                "suggestion": detected if not available and detected != configured else None}
    if detected is not None:
        return {"path": detected, "source": "auto", "available": True, "suggestion": None}
    return {"path": None, "source": "none", "available": False, "suggestion": None}


def data_root() -> Path:
    """Folder with the videos and telemetry (cases, ``maps/``); it does not have to exist."""
    state = data_root_state()
    return state["path"] or Path.home() / "DroneData"


def set_data_root(path: str | Path) -> Path:
    """Remember the folder picked in the UI (``~/.dronevision/settings.json``)."""
    chosen = Path(str(path)).expanduser().resolve()
    if not chosen.is_dir():
        raise ValueError(f"Папка не найдена: {chosen}")
    stored = user_settings()
    stored["data_root"] = str(chosen)
    USER_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = USER_SETTINGS_FILE.with_name(USER_SETTINGS_FILE.name + ".part")
    temporary.write_text(json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, USER_SETTINGS_FILE)
    return chosen


def coastline_path() -> Path:
    """OSM land polygons used for «Суша», «Берег» and the heading calibration."""
    configured = _configured("DRONEVISION_COASTLINE", "coastline")
    if configured:
        return configured
    in_data = data_root() / "maps" / COASTLINE_NAME
    bundled = MAPS_DIR / COASTLINE_NAME
    return in_data if in_data.is_file() or not bundled.is_file() else bundled


def cache_dir(name: str) -> Path:
    return Path(tempfile.gettempdir()) / name


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES and path.is_file()


def find_tool(name: str) -> str | None:
    """Path of an external program: ``bin/`` of the application, its root, then PATH."""
    file_name = f"{name}.exe" if IS_WINDOWS else name
    for folder in (BIN_DIR, APP_DIR):
        candidate = folder / file_name
        if candidate.is_file() and (IS_WINDOWS or os.access(candidate, os.X_OK)):
            return str(candidate)
    return shutil.which(name)


def tool(name: str) -> str:
    found = find_tool(name)
    if not found:
        hint = ("запустите DroneVision.bat ещё раз или выполните «python -m dronevision setup»" if IS_WINDOWS
                else f"установите {name} (для ffmpeg/ffprobe: sudo apt install ffmpeg)")
        raise RuntimeError(f"Не найдена программа {name}: {hint}")
    return found


def run_tool(command: list[str]) -> subprocess.CompletedProcess:
    """Run a helper program; its output is decoded as UTF-8 whatever the system code page is."""
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")


def utf8_console() -> None:
    """Russian messages and symbols like ° must not crash a cp866/cp1251 console."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
