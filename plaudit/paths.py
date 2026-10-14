"""Where Plaudit keeps its files and how it starts itself.

From source, everything stays in the project folder, as before. When installed
(the frozen Plaudit.exe), user data lives in %APPDATA%\\Plaudit, because the
install folder is replaced on upgrade and removed on uninstall.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(sys.executable).resolve().parent if FROZEN else PROJECT_ROOT
DATA_DIR = Path(os.environ.get("APPDATA") or Path.home()) / "Plaudit" if FROZEN else PROJECT_ROOT

SCRIPTS = {"tray": "plaudit_tray.pyw", "settings": "plaudit_settings.pyw"}


def launch_args(mode: str) -> list[str]:
    """The command that starts Plaudit in `mode` ("tray" or "settings") without a console window."""
    if FROZEN:
        return [sys.executable, f"--{mode}"]
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    return [str(pythonw), str(PROJECT_ROOT / SCRIPTS[mode])]


def launch(mode: str) -> None:
    subprocess.Popen(launch_args(mode), cwd=APP_DIR)


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR
