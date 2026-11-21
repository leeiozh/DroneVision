"""Work session of the web UI: opened video, marked frames, settings, drawn shapes and shown results.

The browser sends its state (``web_static/session.js``); here it is only stored and checked.
The last session lives in the user's profile and is restored at the next start; the same JSON can
be written to any file («Сохранить в файл…») and opened later.
"""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import shutil


FORMAT = "dronevision-session"
VERSION = 1
SESSION_DIR = Path.home() / ".dronevision"
LAST_SESSION = SESSION_DIR / "last_session.json"
# The last session of the previously opened video: a safety net when another video replaces it.
PREVIOUS_SESSION = SESSION_DIR / "previous_session.json"
FILE_SUFFIX = ".dvsession.json"


def _read(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except ValueError as exc:
        raise ValueError(f"Файл сессии повреждён: {path.name}") from exc
    if not isinstance(raw, dict) or raw.get("format") != FORMAT:
        raise ValueError(f"Это не файл сессии DroneVision: {path.name}")
    return raw


def _video(session: dict) -> str:
    return str((session.get("form") or {}).get("video_path") or "")


def _relocate_session(session: dict) -> dict:
    """Rebase a session when its data folder was moved to the configured root.

    Browser tabs can outlive a disk remount and autosave their old absolute path.  If that video
    no longer exists but the same file is present directly in the currently selected data root,
    move every path below the old video folder together.  An explicit, still valid path is never
    changed.
    """
    video_text = _video(session)
    if not video_text:
        return session
    old_video = Path(video_text).expanduser()
    try:
        if old_video.is_file():
            return session
        from .paths import data_root
        new_root = data_root().expanduser().resolve()
        new_video = new_root / old_video.name
        if not new_video.is_file():
            return session
    except OSError:
        return session
    old_root = str(old_video.parent)
    new_root_text = str(new_root)

    def rebase(value):
        if isinstance(value, str) and (value == old_root or value.startswith(old_root + os.sep)):
            return new_root_text + value[len(old_root):]
        if isinstance(value, list):
            return [rebase(item) for item in value]
        if isinstance(value, dict):
            return {key: rebase(item) for key, item in value.items()}
        return value

    return rebase(session)


def save_session(session: dict, path: str | None = None, keep_results: bool = False) -> dict:
    """Write the session to ``path`` or, without it, to the last-session file.

    ``keep_results`` (autosave): the browser left out the bulky ``results`` because they did not
    change; they are taken from the stored last session of the same video.
    """
    if not isinstance(session, dict):
        raise ValueError("Пустая сессия")
    session = _relocate_session(session)
    session = {**session, "format": FORMAT, "version": VERSION,
               "saved_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    if path:
        target = Path(path).expanduser()
        if target.suffix.lower() != ".json":
            target = target.with_name(target.name + FILE_SUFFIX)
        target = target.resolve()
    else:
        target = LAST_SESSION
        previous = None
        if target.is_file():
            try:
                # Compare canonical paths: an old browser tab may still autosave the location
                # used before a removable disk was mounted under a different name.
                previous = _relocate_session(_read(target))
            except (OSError, ValueError):
                previous = None
        if previous is not None:
            if _video(previous) != _video(session):
                shutil.copyfile(target, PREVIOUS_SESSION)
            elif keep_results and "results" not in session:
                session["results"] = previous.get("results") or {}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    temporary.write_text(json.dumps(session, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, target)
    return {"path": str(target), "saved_at": session["saved_at"]}


def _raster_ready(path: str | None) -> bool:
    """The PNG and its sidecar still exist, so ``/api/raster`` can show the result again."""
    if not path:
        return False
    image = Path(path)
    return image.is_file() and image.with_suffix(".geo.json").is_file()


def load_session(path: str | None = None) -> dict:
    """Stored session plus what is no longer on disk; ``{"session": None}`` when there is none."""
    source = Path(path).expanduser().resolve() if path else LAST_SESSION
    if not source.is_file():
        if path:
            raise ValueError(f"Файл не найден: {source}")
        return {"session": None}
    session = _relocate_session(_read(source))
    results = dict(session.get("results") or {})
    missing = []
    projection = results.get("projection")
    if projection:
        paths = [v.get("path") for v in (projection.get("result") or {}).get("variants") or []]
        if not paths or not all(_raster_ready(p) for p in paths):
            missing += [p for p in paths if p]
            results.pop("projection")
    mosaic = results.get("mosaic")
    if mosaic:
        result = mosaic.get("result") or {}
        # A comparison of two averaged intervals keeps its two rasters in "variants".
        paths = [v.get("path") for v in result.get("variants") or []] or [result.get("path")]
        if not all(_raster_ready(p) for p in paths):
            missing += [p for p in paths if p] or ["мозаика"]
            results.pop("mosaic")
    waves = results.get("waves")
    if waves and not Path(str(waves.get("path") or "")).is_file():
        missing.append(waves.get("path") or "течения")
        results.pop("waves")
    ice = results.get("ice")
    if ice and not Path(str(ice.get("path") or "")).is_file():
        missing.append(ice.get("path") or "дрейф льдин")
        results.pop("ice")
    session["results"] = results
    video = _video(session)
    return {"session": session, "path": str(source), "saved_at": session.get("saved_at"),
            "video_exists": bool(video) and Path(video).is_file(), "missing_results": missing}


def clear_last_session() -> dict:
    """«Новая сессия»: the stored one is kept as the previous session, not deleted."""
    if LAST_SESSION.is_file():
        os.replace(LAST_SESSION, PREVIOUS_SESSION)
    return {"cleared": True}
