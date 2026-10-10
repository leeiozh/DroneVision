"""Native "open file / choose folder / save as" dialog for the web UI.

Windows and macOS use tkinter (part of the standard Python installer); Linux prefers zenity.
The tkinter window runs in a separate process: Tk must own the main thread, the web server does not.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys


FILTERS = {
    "video": ("Видео", ("*.MP4", "*.mp4", "*.MOV", "*.mov")),
    "log": ("DJI-логи", ("*.txt", "*.json")),
    "session": ("Сессии DroneVision", ("*.dvsession.json", "*.json")),
}


def _start_dir(start: str | None) -> Path:
    path = Path(start).expanduser() if start else Path.home()
    try:
        if path.is_file():
            path = path.parent
        while not path.is_dir() and path != path.parent:
            path = path.parent
        return path if path.is_dir() else Path.home()
    except OSError:
        return Path.home()


def _zenity(kind: str, filter_name: str | None, start: Path, name: str) -> str | None:
    command = ["zenity", "--file-selection", f"--filename={start}/{name if kind == 'save' else ''}"]
    if kind == "directory":
        command.append("--directory")
    elif kind == "save":
        command += ["--save", "--confirm-overwrite"]
    if kind != "directory" and filter_name in FILTERS:
        label, patterns = FILTERS[filter_name]
        command.append(f"--file-filter={label} | {' '.join(patterns)}")
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return result.stdout.strip() or None if result.returncode == 0 else None


def _tk(kind: str, filter_name: str | None, start: Path, name: str) -> str | None:
    result = subprocess.run([sys.executable, "-m", "dronevision.dialog", kind, filter_name or "", str(start), name],
                            capture_output=True, cwd=str(Path(__file__).resolve().parent.parent))
    if result.returncode:
        raise RuntimeError("Не удалось открыть окно выбора файла: "
                           + result.stderr.decode("utf-8", "replace").strip()[-300:])
    return result.stdout.decode("utf-8", "replace").strip() or None


def choose_path(kind: str = "file", filter_name: str | None = None, start: str | None = None,
                name: str | None = None) -> str | None:
    """Selected path in the native form of the system (``C:\\...`` on Windows) or None when cancelled.

    ``kind``: "file" (open), "directory" or "save" (``name`` = suggested file name).
    """
    folder = _start_dir(start)
    # A suggested name is a bare file name: no folders, no characters Windows forbids.
    name = "".join("_" if c in '\\/:*?"<>|' or ord(c) < 32 else c for c in (name or "")).strip()
    backend = _zenity if sys.platform.startswith("linux") and shutil.which("zenity") else _tk
    chosen = backend(kind, filter_name, folder, name)
    return str(Path(chosen)) if chosen else None


def _main(kind: str, filter_name: str, start: str, name: str = "") -> None:
    import tkinter
    from tkinter import filedialog

    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)      # otherwise the dialog opens behind the browser
    root.update()
    types = [(FILTERS[filter_name][0], " ".join(FILTERS[filter_name][1]))] if filter_name in FILTERS else []
    types.append(("Все файлы", "*.*"))
    if kind == "directory":
        chosen = filedialog.askdirectory(parent=root, initialdir=start, title="DroneVision: выберите папку")
    elif kind == "save":
        chosen = filedialog.asksaveasfilename(parent=root, initialdir=start, initialfile=name, filetypes=types,
                                              title="DroneVision: сохранить как")
    else:
        chosen = filedialog.askopenfilename(parent=root, initialdir=start, title="DroneVision: выберите файл",
                                            filetypes=types)
    root.destroy()
    sys.stdout.buffer.write((chosen or "").encode("utf-8"))


if __name__ == "__main__":
    _main(*sys.argv[1:5])
