"""One-time preparation of a computer: external programs and a self-check.

``python -m dronevision setup`` downloads what is missing into ``bin/`` (Windows) and reports what
works.  ``--target windows`` fills ``bin/`` from any system, to hand a complete folder to colleagues.
"""
from __future__ import annotations

import io
from pathlib import Path
import shutil
import urllib.request
import zipfile

from . import paths


DJI_LOG_URL = ("https://github.com/lvauvillier/dji-log-parser/releases/download/v0.5.3/"
               "dji-log-x86_64-pc-windows-msvc.zip")     # same version as the bundled Linux binary
FFMPEG_URLS = (
    "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
)


def _download(url: str, say) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "DroneVision setup"})
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or 0)
        chunks, done, shown = [], 0, -1
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
            done += len(chunk)
            step = done * 10 // total if total else done >> 24
            if step != shown:
                shown = step
                say(f"    {done / 1e6:6.0f} МБ" + (f" из {total / 1e6:.0f}" if total else ""))
    return b"".join(chunks)


def _extract(archive: bytes, names: set[str]) -> list[str]:
    """Copy the listed .exe files from a zip (whatever folder they are in) to ``bin/``."""
    paths.BIN_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        for member in bundle.namelist():
            name = member.rsplit("/", 1)[-1]
            if name in names and name not in written:
                with bundle.open(member) as source, (paths.BIN_DIR / name).open("wb") as target:
                    shutil.copyfileobj(source, target)
                written.append(name)
    missing = names - set(written)
    if missing:
        raise RuntimeError(f"в архиве нет {', '.join(sorted(missing))}")
    return written


def install_windows_tools(say=print, force: bool = False) -> list[str]:
    """Download dji-log.exe, ffmpeg.exe and ffprobe.exe into ``bin/``; returns the problems found."""
    problems = []
    present = lambda name: (paths.BIN_DIR / name).is_file() and not force
    if not present("dji-log.exe"):
        say("Скачиваю dji-log (декодер полётных логов DJI, 1.4 МБ)…")
        try:
            _extract(_download(DJI_LOG_URL, say), {"dji-log.exe"})
        except Exception as exc:
            problems.append(f"dji-log.exe: {exc}. Скачайте вручную {DJI_LOG_URL} и положите dji-log.exe в {paths.BIN_DIR}")
    if not (present("ffmpeg.exe") and present("ffprobe.exe")):
        say("Скачиваю ffmpeg и ffprobe (чтение видео, около 100 МБ, один раз)…")
        error = None
        for url in FFMPEG_URLS:
            try:
                _extract(_download(url, say), {"ffmpeg.exe", "ffprobe.exe"})
                error = None
                break
            except Exception as exc:
                error = exc
        if error is not None:
            problems.append(f"ffmpeg: {error}. Скачайте вручную {FFMPEG_URLS[0]} и положите ffmpeg.exe и ffprobe.exe "
                            f"из папки bin архива в {paths.BIN_DIR}")
    return problems


def check(say=print) -> list[str]:
    """Report the state of this computer; returns the problems that block work."""
    problems = []
    for module, package in (("numpy", "numpy"), ("cv2", "opencv-python"), ("pyproj", "pyproj")):
        try:
            version = __import__(module).__version__
            say(f"  [ok] {package} {version}")
        except Exception as exc:
            problems.append(f"не установлен пакет {package} ({exc})")
    for name in ("ffmpeg", "ffprobe", "dji-log"):
        found = paths.find_tool(name)
        if not found:
            problems.append(f"не найдена программа {name}")
            continue
        try:
            result = paths.run_tool([found, "--version" if name == "dji-log" else "-version"])
            line = (result.stdout or result.stderr).strip().splitlines()[0][:60] if result.returncode == 0 else ""
        except OSError as exc:
            line, result = "", exc
        if line:
            say(f"  [ok] {name}: {line}")
        else:
            problems.append(f"программа {found} не запускается")
    try:
        import tkinter  # noqa: F401
        dialog = True
    except Exception:
        dialog = bool(shutil.which("zenity"))
    if dialog:
        say("  [ok] окно выбора файлов")
    else:
        problems.append("нет tkinter: переустановите Python с python.org с галочкой «tcl/tk and IDLE»")
    from . import coastdata
    root = paths.data_root()
    coast = coastdata.status()
    areas = [Path(item["path"]).name for item in coast["extracts"]]
    say(f"  [{'ok' if areas else '--'}] береговая линия (районов: {len(areas)}): {', '.join(areas[:6]) or 'нет'}"
        + ("" if areas else " — «Суша», «Берег» и выравнивание по берегу недоступны"))
    say(f"  [{'ok' if coast['planet'] else '--'}] планетарные полигоны суши OSM: {coast['planet'] or 'не скачаны'}"
        + ("" if coast["planet"] else " (для новых районов: python -m dronevision coastline)"))
    say(f"  [{'ok' if root.is_dir() else '--'}] набор DroneData: {root}"
        + ("" if root.is_dir() else f" (не найден; это не мешает работе со своими видео. Путь задаётся в {paths.SETTINGS_FILE.name})"))
    return problems


def run(target: str | None = None, force: bool = False, say=print) -> int:
    problems = []
    if target == "windows" or (target is None and paths.IS_WINDOWS):
        problems += install_windows_tools(say, force)
    if target is None:
        say("Проверка DroneVision:")
        problems += check(say)
    for problem in problems:
        say(f"  [!!] {problem}")
    say("Готово: всё на месте." if not problems else "Есть проблемы, см. строки [!!] выше.")
    return 1 if problems else 0


BUNDLE_ITEMS = ("dronevision", "bin", "maps", "docs", "DroneVision.bat", "DroneVision-check.bat", "README_WINDOWS.txt",
                "README.md", "pyproject.toml", "requirements-app.txt", "dronevision_settings.example.json")


def bundle(output: str | None = None, say=print) -> Path:
    """Zip everything a Windows computer needs (code, helper programs, coastline) into one archive."""
    problems = install_windows_tools(say)
    if problems:
        raise RuntimeError("; ".join(problems))
    target = Path(output).expanduser().resolve() if output else paths.APP_DIR / "dist" / "DroneVision-windows.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for item in BUNDLE_ITEMS:
            source = paths.APP_DIR / item
            files = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())
            for file in files:
                # The planet land polygons (~1.4 GB) stay behind: the regional extracts in maps/ are enough.
                if "__pycache__" in file.parts or file.suffix == ".pyc" or "osm-planet" in file.parts:
                    continue
                archive.write(file, Path("DroneVision") / file.relative_to(paths.APP_DIR))
                count += 1
    say(f"{target}: {count} файлов, {target.stat().st_size / 1e6:.0f} МБ")
    return target
