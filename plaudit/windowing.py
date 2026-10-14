"""Keeps a single settings window: opening Plaudit again (tray icon, Start menu) restores
the existing window, even when it's minimised, instead of starting a second copy."""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes

from .paths import DATA_DIR, ensure_data_dir, launch

WINDOW_TITLE = "Plaudit"
MUTEX_NAME = "Local\\PlauditSettings"
PID_FILE = DATA_DIR / "settings.pid"

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
user32.EnumWindows.argtypes = [EnumWindowsProc, wintypes.LPARAM]
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsIconic.argtypes = [wintypes.HWND]
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
kernel32.CreateMutexW.restype = wintypes.HANDLE
SW_SHOW, SW_RESTORE = 5, 9


def claim_single_instance() -> bool:
    """Called by the settings window at start. False if another one is already open.
    The mutex lives as long as this process."""
    claim_single_instance.handle = kernel32.CreateMutexW(None, False, MUTEX_NAME)
    if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        return False
    ensure_data_dir()
    PID_FILE.write_text(str(os.getpid()), encoding="ascii")
    return True


def release_single_instance() -> None:
    try:
        if PID_FILE.read_text(encoding="ascii").strip() == str(os.getpid()):
            PID_FILE.unlink()
    except (OSError, ValueError):
        pass


def find_settings_window() -> int | None:
    """The top-level 'Plaudit' window of the process recorded in settings.pid."""
    try:
        pid = int(PID_FILE.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None
    found: list[int] = []

    def visit(hwnd, _):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid:
            title = ctypes.create_unicode_buffer(64)
            user32.GetWindowTextW(hwnd, title, 64)
            if title.value == WINDOW_TITLE:
                found.append(hwnd)
                return False
        return True

    user32.EnumWindows(EnumWindowsProc(visit), 0)
    return found[0] if found else None


def activate_settings_window() -> bool:
    hwnd = find_settings_window()
    if hwnd is None:
        return False
    user32.ShowWindow(hwnd, SW_RESTORE if user32.IsIconic(hwnd) or not user32.IsWindowVisible(hwnd) else SW_SHOW)
    user32.SetForegroundWindow(hwnd)
    return True


def show_settings() -> None:
    """Brings the open settings window to the front, or opens one."""
    if not activate_settings_window():
        launch("settings")
