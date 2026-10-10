"""Fill a fresh Python environment with what DroneVision needs, reusing what is already installed.

Run as a plain file by ``DroneVision.bat`` (``python dronevision\\bootstrap.py``) inside the new ``.venv``; only the
standard library is used here.  When the environment was created from a Python that already has some of the
libraries (``--system-site-packages``), the suitable ones are reused and only the rest is downloaded.  New
packages always go into ``.venv``: the user's own Python installation is never changed.  Only ready-made wheels
are accepted (``--only-binary``), so nothing is compiled.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import re
import subprocess
import sys

# import name, pip requirement, minimal version
NEEDED = (("numpy", "numpy>=1.26", (1, 26)), ("cv2", "opencv-python>=4.9", (4, 9)), ("pyproj", "pyproj>=3.6", (3, 6)))
PINNED = Path(__file__).resolve().parent.parent / "requirements-app.txt"


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text)[:3])


def _probe(module: str, minimum: tuple[int, ...]) -> tuple[str, str, str]:
    """('ok' | 'old' | 'missing', version, folder) of one library, checked in a separate process."""
    code = f"import {module} as m; print(m.__version__); print(m.__file__)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        return "missing", "", ""
    version, location = (result.stdout.strip().splitlines() + ["", ""])[:2]
    return ("ok" if _version(version) >= minimum else "old"), version, str(Path(location).parent)


def _together() -> bool:
    """All libraries in one process: a numpy that does not match the build of cv2 or pyproj fails only here."""
    code = ("import numpy, cv2, pyproj; "
            "cv2.resize(numpy.zeros((4, 4, 3), numpy.uint8), (2, 2)); pyproj.Geod(ellps='WGS84').inv(0, 0, 1, 1)")
    return subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0


def plan() -> tuple[list[str], list[str]]:
    """(lines describing every library as it is now, pip requirements still to install)."""
    own = Path(sys.prefix).resolve()
    lines, wanted = [], []
    for module, requirement, minimum in NEEDED:
        state, version, location = _probe(module, minimum)
        if state == "ok":
            inside = own in Path(location).resolve().parents
            lines.append(f"  [ok] {module} {version} — {'в окружении DroneVision' if inside else 'из Python, установленного на компьютере'}")
        else:
            lines.append(f"  [--] {module}: " + ("нет" if state == "missing" else f"{version} слишком старый, для DroneVision нужен новее"))
            wanted.append(requirement)
    return lines, wanted


def _pip_install(arguments: list[str]) -> bool:
    """pip with its progress on screen; complaints about the user's other libraries are replaced by a calm note."""
    command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
               "--only-binary=:all:", *arguments]
    result = subprocess.run(command, stderr=subprocess.PIPE, text=True, errors="replace")
    errors = result.stderr or ""
    marker = "ERROR: pip's dependency resolver"
    if result.returncode == 0 and marker in errors:
        # A newer numpy was put into .venv next to e.g. an old scipy of the user's Python: DroneVision does not
        # use those libraries, and the user's Python still sees its own numpy.
        errors = errors[:errors.index(marker)]
        print("  Замечание: в .venv поставлены более новые версии части библиотек, чем в вашем Python.\n"
              "  Ваш Python и его библиотеки не изменены и работают как раньше.")
    if errors.strip():
        print(errors.rstrip(), file=sys.stderr)
    return result.returncode == 0


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    dry = "--dry-run" in sys.argv
    if sys.prefix == sys.base_prefix and not dry:
        print("  [!!] bootstrap.py запускается только внутри .venv: чужой Python не изменяется")
        return 2
    lines, wanted = plan()
    if wanted:
        if len(wanted) < len(NEEDED):
            print("  Часть библиотек уже есть на компьютере:")
            print("\n".join(lines))
        if len(wanted) == len(NEEDED) and sys.version_info[:2] == (3, 12) and PINNED.is_file():
            # Nothing to reuse on the reference Python: take the exact versions the program was checked with.
            arguments = ["-r", str(PINNED)]
        else:
            arguments = wanted
        print(f"  Скачиваю в .venv: {', '.join(wanted)}")
        if dry:
            return 0
        if not _pip_install(arguments):
            print("  [!!] pip не смог установить пакеты (нет сети или для этого Python нет готовых сборок)")
            return 1
        lines, wanted = plan()   # what is actually used now: pip may have replaced a reused library by a newer one
    print("\n".join(lines))
    if dry:
        return 0
    if wanted or not _together():
        print("  [!!] библиотеки не работают вместе (несовместимые версии)")
        return 1
    print("  Библиотеки на месте.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
