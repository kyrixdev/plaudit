"""The Python side of the settings window. Every public method is callable from JavaScript
as window.pywebview.api.<name>(...) and returns plain JSON-able data."""

from __future__ import annotations

import csv
import ctypes
import os
from pathlib import Path

import webbrowser

from .. import AUTHOR_URL, REPO_URL, startup
from ..actions import KINDS, MEDIA_KEYS, SOUNDS, ActionDispatcher, ActionError, ZoneAction, plan
from ..livelog import LOGS_DIR
from ..paths import launch
from ..patterns import COUNTS, MAX_GAP_S, SETTINGS_PATH, PatternSettings
from ..tray import MUTEX_NAME, STATUS_PATH, load_tray_settings, update_tray_settings

MIN_GAP_S, MAX_GAP_LIMIT_S = 0.4, 1.2

# Order and grouping of the action menu, like a macOS pop-up menu with separators.
KIND_GROUPS = [
    ["none"],
    ["media", "hotkey"],
    ["open", "url", "screenshot"],
    ["sound", "speak_text", "copy_text"],
    ["command"],
]
CHOICE_LABELS = {
    "media": {key: label for key, (_, label) in MEDIA_KEYS.items()},
    "screenshot": {"full": "Whole screen to clipboard", "selection": "Select an area (Snipping overlay)"},
    "sound": {key: key.capitalize() for key in SOUNDS},
}
PLACEHOLDERS = {
    "hotkey": "win+d",
    "open": "notepad, a file path, or spotify:",
    "url": "example.com",
    "command": "Start-Process calc",
    "copy_text": "Text to copy",
    "speak_text": "Text to say",
}


def tray_running() -> bool:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenMutexW.restype = ctypes.c_void_p
    handle = kernel32.OpenMutexW(0x00100000, False, MUTEX_NAME)  # SYNCHRONIZE
    if handle:
        kernel32.CloseHandle(ctypes.c_void_p(handle))
        return True
    return False


class SettingsApi:
    """Only public methods may be public: pywebview exposes every public attribute to
    JavaScript and walks into nested objects, which froze the window when a Window
    object was reachable from here."""

    def __init__(self, settings_path: Path = SETTINGS_PATH):
        self._settings_path = settings_path
        self._window = None  # set by the app so file dialogs can open
        self._mic_info: dict | None = None

    def attach(self, window) -> None:
        self._window = window

    # -- reading ----------------------------------------------------------------------------------

    def get_catalog(self) -> dict:
        groups = []
        for group in KIND_GROUPS:
            kinds = []
            for key in group:
                kind = KINDS[key]
                choices = [{"value": c, "label": CHOICE_LABELS.get(key, {}).get(c, c)} for c in kind.choices]
                kinds.append({"key": key, "title": kind.title, "prompt": kind.prompt,
                              "choices": choices, "placeholder": PLACEHOLDERS.get(key, "")})
            groups.append(kinds)
        return {"groups": groups, "counts": list(COUNTS), "gap": {"min": MIN_GAP_S, "max": MAX_GAP_LIMIT_S,
                                                                  "default": MAX_GAP_S}}

    def get_state(self) -> dict:
        settings = PatternSettings.load(self._settings_path)
        tray = load_tray_settings()
        status = self._read_status()
        running = tray_running()
        return {
            "patterns": [self._pattern(count, settings.action(count)) for count in COUNTS],
            "options": {
                "maxGap": settings.max_gap_s,
                "ignoreWhileTyping": settings.ignore_while_typing,
                "notifyActions": bool(tray.get("notifyActions", False)),
                "startWithWindows": startup.is_enabled(),
                "paused": bool(tray.get("paused", False)),
            },
            "tray": {
                "running": running,
                "state": status.get("state", "stopped") if running else "stopped",
                "text": status.get("text", "") if running else "Not running",
                "lastAction": status.get("lastAction", ""),
            },
            "mic": self._mic(),
        }

    def get_activity(self, limit: int = 60) -> list[dict]:
        """Recent bursts, newest first, from the last few pattern logs."""
        entries = []
        logs = sorted(LOGS_DIR.glob("patterns-*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
        for path in logs:
            try:
                with path.open(encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle):
                        if not row.get("recognised", "").startswith("BURST"):
                            continue
                        outcome, action = row.get("trigger", ""), row.get("action", "")
                        status = "failed" if action.startswith("failed") else \
                            "ran" if action and action != "actions off" else "ignored"
                        entries.append({"time": row["timestamp"], "count": int(row["recognised"].split()[1]),
                                        "outcome": outcome, "action": action, "status": status})
            except (OSError, ValueError, KeyError, IndexError):
                continue
        entries.sort(key=lambda e: e["time"], reverse=True)
        return entries[:limit]

    # -- editing ----------------------------------------------------------------------------------

    def save_pattern(self, count: int, kind: str, value: str) -> dict:
        action = ZoneAction(kind, (value or "").strip() if kind not in ("copy_text", "speak_text") else value or "")
        try:
            plan(action)
        except ActionError as error:
            return {"ok": False, "error": str(error)}
        settings = PatternSettings.load(self._settings_path)
        settings.actions[int(count)] = action
        settings.save(self._settings_path)
        return {"ok": True, "pattern": self._pattern(int(count), action)}

    def try_action(self, kind: str, value: str) -> dict:
        dispatcher = ActionDispatcher()
        try:
            done = dispatcher.perform(ZoneAction(kind, value or ""))
            dispatcher.wait(15)
        except (ActionError, OSError) as error:
            return {"ok": False, "error": str(error)}
        errors = dispatcher.poll_errors()
        if errors:
            return {"ok": False, "error": errors[0]}
        return {"ok": True, "message": f"Ran: {done}" if done else "Nothing to run"}

    def set_option(self, name: str, value) -> dict:
        try:
            if name in ("maxGap", "ignoreWhileTyping"):
                settings = PatternSettings.load(self._settings_path)
                if name == "maxGap":
                    settings.max_gap_s = round(min(max(float(value), MIN_GAP_S), MAX_GAP_LIMIT_S), 2)
                else:
                    settings.ignore_while_typing = bool(value)
                settings.save(self._settings_path)
            elif name == "notifyActions":
                update_tray_settings(notifyActions=bool(value))
            elif name == "paused":
                update_tray_settings(paused=bool(value))
            elif name == "startWithWindows":
                startup.enable() if value else startup.disable()
            else:
                return {"ok": False, "error": f"Unknown option {name}"}
        except OSError as error:
            return {"ok": False, "error": str(error)}
        return {"ok": True}

    def start_listening(self) -> dict:
        """Resumes listening, starting the tray app if it isn't running."""
        update_tray_settings(paused=False)
        if not tray_running():
            launch("tray")
        return {"ok": True}

    def browse(self) -> str | None:
        if self._window is None:
            return None
        import webview
        result = self._window.create_file_dialog(
            webview.FileDialog.OPEN, allow_multiple=False,
            file_types=("Apps and shortcuts (*.exe;*.lnk;*.url)", "All files (*.*)"))
        return result[0] if result else None

    def open_logs(self) -> None:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(LOGS_DIR)

    def open_author(self) -> None:
        webbrowser.open(AUTHOR_URL)

    def open_repo(self) -> None:
        """Opens the GitHub page in the browser, where people can star the project."""
        webbrowser.open(REPO_URL)

    # -- helpers ----------------------------------------------------------------------------------

    @staticmethod
    def _pattern(count: int, action: ZoneAction) -> dict:
        return {"count": count, "kind": action.kind, "value": action.value,
                "summary": "No action" if action.kind == "none" else action.summary,
                "title": KINDS[action.kind].title}

    @staticmethod
    def _read_status() -> dict:
        import json
        try:
            return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _mic(self) -> dict:
        """Queried once: the status refresh calls get_state every 2 s."""
        if self._mic_info is None:
            try:
                from ..audio import default_input_device
                from ..console import clean_device_name
                self._mic_info = {"name": clean_device_name(default_input_device().name),
                                  "mode": "Raw (Windows mic effects off)"}
            except Exception:
                return {"name": "Unavailable", "mode": ""}
        return self._mic_info
