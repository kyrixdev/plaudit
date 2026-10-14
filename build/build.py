"""Builds Plaudit.exe and the Plaudit-Setup installer.

    .venv\\Scripts\\python build\\build.py

Output: build\\dist\\Plaudit\\ (the app) and build\\dist\\Plaudit-Setup-<version>.exe.
Needs the dev requirements: pip install -r requirements-dev.txt
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
DIST = BUILD / "dist"
WORK = BUILD / "work"
ICON = ROOT / "assets" / "plaudit.ico"

sys.path.insert(0, str(ROOT))
from plaudit import __version__  # noqa: E402

# Keep the installer small: it only needs the installer screens, not the audio engine.
SETUP_EXCLUDES = ["numpy", "sounddevice", "_sounddevice_data", "PIL", "pystray", "pytest"]


def version_file(name: str, description: str) -> Path:
    numbers = [int(part) for part in __version__.split(".")] + [0]
    tuple_text = ", ".join(str(n) for n in numbers[:4])
    path = WORK / f"{name}-version.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers=({tuple_text}), prodvers=({tuple_text}), mask=0x3f, flags=0x0, OS=0x40004,
                    fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', 'Oussama Kefi'),
      StringStruct('FileDescription', '{description}'),
      StringStruct('FileVersion', '{__version__}'),
      StringStruct('InternalName', '{name}'),
      StringStruct('LegalCopyright', 'Copyright (c) 2026 Oussama Kefi. MIT licence.'),
      StringStruct('OriginalFilename', '{name}.exe'),
      StringStruct('ProductName', 'Plaudit'),
      StringStruct('ProductVersion', '{__version__}')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])]
)
""", encoding="utf-8")
    return path


def pyinstaller(*args: str) -> None:
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--windowed", "--icon", str(ICON),
               "--distpath", str(DIST), "--workpath", str(WORK), "--specpath", str(WORK), *args]
    print(">", " ".join(command[3:]), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def data(source: Path, target: str) -> list[str]:
    return ["--add-data", f"{source};{target}"]


def stop_built_copies() -> None:
    """A Plaudit.exe still running from build/dist locks its files; stop only those copies."""
    script = (f"Get-Process Plaudit -ErrorAction SilentlyContinue | "
              f"Where-Object {{ $_.Path -like '{DIST}*' }} | Stop-Process -Force")
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=False)


def main() -> None:
    stop_built_copies()
    subprocess.run([sys.executable, str(BUILD / "make_icon.py")], check=True)
    shutil.rmtree(DIST / "Plaudit", ignore_errors=True)

    # 1. The app: a folder with Plaudit.exe (starts faster and trips antivirus less than one big file).
    pyinstaller("--name", "Plaudit", "--version-file", str(version_file("Plaudit", "Plaudit")),
                "--hidden-import", "pystray._win32",
                *data(ROOT / "plaudit" / "gui" / "web", "plaudit/gui/web"),
                *data(ROOT / "plaudit" / "installer" / "web", "plaudit/installer/web"),
                str(BUILD / "entry_app.py"))
    app = DIST / "Plaudit"
    if "--app-only" in sys.argv:
        return

    # 2. The payload: the app folder, zipped.
    payload = WORK / "payload.zip"
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for file in sorted(app.rglob("*")):
            if file.is_file():
                archive.write(file, file.relative_to(app))
    print(f"payload: {payload.stat().st_size / 1e6:.1f} MB", flush=True)

    # 3. The installer: one .exe carrying the payload.
    setup_name = f"Plaudit-Setup-{__version__}"
    excludes = [arg for module in SETUP_EXCLUDES for arg in ("--exclude-module", module)]
    pyinstaller("--onefile", "--name", setup_name,
                "--version-file", str(version_file(setup_name, "Plaudit Setup")),
                *excludes,
                *data(payload, "."),
                *data(ROOT / "plaudit" / "installer" / "web", "plaudit/installer/web"),
                str(BUILD / "entry_setup.py"))
    setup = DIST / f"{setup_name}.exe"
    print(f"\nApp:       {app / 'Plaudit.exe'}")
    print(f"Installer: {setup}  ({setup.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
