"""Windows plumbing for installing and uninstalling Plaudit (per user, no admin rights).

Installs to %LOCALAPPDATA%\\Programs\\Plaudit, adds Start menu (and optionally desktop)
shortcuts, and registers in Settings > Apps > Installed apps.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import tempfile
import time
import winreg
import zipfile
from pathlib import Path
from typing import Callable

from .. import REPO_URL

APP_NAME = "Plaudit"
EXE_NAME = "Plaudit.exe"
PUBLISHER = "Oussama Kefi"
UNINSTALL_KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\Plaudit"
CREATE_NO_WINDOW = 0x08000000


class InstallError(RuntimeError):
    pass


# -- locations -----------------------------------------------------------------------------------

def default_install_dir() -> Path:
    return Path(os.environ["LOCALAPPDATA"]) / "Programs" / APP_NAME


def _known_folder(csidl: int) -> Path:
    buffer = ctypes.create_unicode_buffer(260)
    ctypes.windll.shell32.SHGetFolderPathW(None, csidl, None, 0, buffer)
    return Path(buffer.value)


def start_menu_shortcut() -> Path:
    return _known_folder(0x02) / f"{APP_NAME}.lnk"  # CSIDL_PROGRAMS


def desktop_shortcut() -> Path:
    return _known_folder(0x10) / f"{APP_NAME}.lnk"  # CSIDL_DESKTOPDIRECTORY (follows OneDrive)


def looks_like_install_dir(path: Path) -> bool:
    """Only ever delete a folder named Plaudit that holds Plaudit.exe."""
    return path.name == APP_NAME and (path / EXE_NAME).is_file()


# -- processes ------------------------------------------------------------------------------------

def stop_running(exclude_pid: int | None = None) -> None:
    command = ["taskkill", "/F", "/T", "/FI", f"IMAGENAME eq {EXE_NAME}"]
    if exclude_pid:
        command += ["/FI", f"PID ne {exclude_pid}"]
    subprocess.run(command, capture_output=True, creationflags=CREATE_NO_WINDOW)
    time.sleep(0.5)  # let Windows release the files


def remove_after_exit(path: Path, pid: int) -> None:
    """Deletes `path` once process `pid` (the running uninstaller) has exited."""
    script = Path(tempfile.gettempdir()) / f"plaudit-cleanup-{pid}.cmd"
    script.write_text(
        "@echo off\r\n"
        ":wait\r\n"
        f'tasklist /FI "PID eq {pid}" 2>nul | find "{pid}" >nul && (ping -n 2 127.0.0.1 >nul & goto wait)\r\n'
        f'rmdir /s /q "{path}"\r\n'
        'del "%~f0"\r\n',
        encoding="ascii",
    )
    subprocess.Popen(["cmd", "/c", str(script)], creationflags=CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
                     close_fds=True)


# -- files ----------------------------------------------------------------------------------------

def extract(payload: Path, destination: Path, progress: Callable[[float], None]) -> None:
    with zipfile.ZipFile(payload) as archive:
        members = archive.infolist()
        total = sum(m.file_size for m in members) or 1
        done = 0
        for member in members:
            archive.extract(member, destination)
            done += member.file_size
            progress(done / total)


def remove_previous(destination: Path) -> None:
    if not destination.exists():
        return
    if not looks_like_install_dir(destination) and any(destination.iterdir()):
        raise InstallError(f"{destination} exists and doesn't look like a Plaudit install; not touching it.")
    for attempt in range(10):
        try:
            shutil.rmtree(destination)
            return
        except OSError:
            time.sleep(0.5)  # a just-closed Plaudit may still hold files
    raise InstallError("The previous version is still in use. Quit Plaudit from its tray icon and try again.")


def folder_size_kb(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) // 1024


# -- shortcuts ------------------------------------------------------------------------------------

def create_shortcut(link: Path, target: Path, description: str) -> None:
    """Creates a .lnk with PowerShell's WScript.Shell; paths travel in environment variables."""
    script = ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:PL_LINK); "
              "$s.TargetPath = $env:PL_TARGET; $s.WorkingDirectory = $env:PL_DIR; "
              "$s.IconLocation = $env:PL_TARGET + ',0'; $s.Description = $env:PL_DESC; $s.Save()")
    env = dict(os.environ, PL_LINK=str(link), PL_TARGET=str(target), PL_DIR=str(target.parent), PL_DESC=description)
    link.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script], env=env,
                            capture_output=True, text=True, creationflags=CREATE_NO_WINDOW)
    if result.returncode != 0 or not link.exists():
        raise InstallError(f"Couldn't create the shortcut {link.name}: {result.stderr.strip()[:200]}")


def remove_shortcuts() -> None:
    for link in (start_menu_shortcut(), desktop_shortcut()):
        try:
            link.unlink()
        except FileNotFoundError:
            pass


# -- Installed apps entry ---------------------------------------------------------------------

def register(install_dir: Path, version: str) -> None:
    exe = install_dir / EXE_NAME
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY) as key:
        for name, value in {
            "DisplayName": APP_NAME,
            "DisplayVersion": version,
            "Publisher": PUBLISHER,
            "DisplayIcon": str(exe),
            "InstallLocation": str(install_dir),
            "UninstallString": f'"{exe}" --uninstall',
            "URLInfoAbout": REPO_URL,
            "HelpLink": f"{REPO_URL}/issues",
        }.items():
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)
        for name, value in {"NoModify": 1, "NoRepair": 1, "EstimatedSize": folder_size_kb(install_dir)}.items():
            winreg.SetValueEx(key, name, 0, winreg.REG_DWORD, value)


def unregister() -> None:
    try:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY)
    except FileNotFoundError:
        pass


def installed_version() -> str | None:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY) as key:
            return winreg.QueryValueEx(key, "DisplayVersion")[0]
    except OSError:
        return None
