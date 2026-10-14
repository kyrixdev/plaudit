"""'Start with Windows': a per-user entry in the registry's Run key (no admin rights needed)."""

from __future__ import annotations

import subprocess
import winreg

from .paths import launch_args

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE_NAME = "Plaudit"


def launch_command() -> str:
    """The tray app, started without a console window."""
    return subprocess.list2cmdline(launch_args("tray"))


def is_enabled(key_path: str = RUN_KEY, name: str = VALUE_NAME) -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            value, _ = winreg.QueryValueEx(key, name)
            return bool(value)
    except OSError:
        return False


def enable(key_path: str = RUN_KEY, name: str = VALUE_NAME, command: str | None = None) -> None:
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
        winreg.SetValueEx(key, name, 0, winreg.REG_SZ, command or launch_command())


def disable(key_path: str = RUN_KEY, name: str = VALUE_NAME) -> None:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, name)
    except FileNotFoundError:
        pass
