"""System-tray app: listens for clap patterns in the background, without a terminal window.

Menu: status, pause/resume (releases the microphone), edit actions, open logs,
notifications, start with Windows, quit. Errors go to logs/tray.log and a
notification, and listening is retried automatically.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import subprocess
import threading
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pystray
from PIL import Image, ImageDraw

from . import startup
from .audio import resolve_device
from .engine import PatternEngine
from .livelog import LOGS_DIR

ROOT = Path(__file__).resolve().parent.parent
TRAY_SETTINGS = ROOT / "tray.json"  # preferences shared with the settings window (paused, notifyActions)
STATUS_PATH = ROOT / "tray-status.json"  # written by the tray, read by the settings window
MUTEX_NAME = "Local\\PlauditTray"
RETRY_S = 5.0
FLASH_S = 0.8

COLOURS = {
    "listening": (46, 158, 107),
    "starting": (217, 164, 0),
    "paused": (138, 138, 138),
    "error": (217, 68, 43),
    "action": (47, 111, 222),
}

log = logging.getLogger("plaudit.tray")


def icon_image(state: str) -> Image.Image:
    """A filled circle with three sound-wave arcs, coloured by state."""
    size = 64
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((2, 2, size - 2, size - 2), fill=COLOURS[state] + (255,))
    for radius, width in ((8, 5), (17, 5), (26, 5)):
        box = (20 - radius, 32 - radius, 20 + radius, 32 + radius)
        draw.arc(box, start=-50, end=50, fill=(255, 255, 255, 255), width=width)
    draw.ellipse((15, 27, 25, 37), fill=(255, 255, 255, 255))
    return image


def already_running() -> bool:
    """Holds a named mutex for the process's lifetime; True if another tray app owns it."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    already_running.handle = kernel32.CreateMutexW(None, False, MUTEX_NAME)
    return ctypes.get_last_error() == 183  # ERROR_ALREADY_EXISTS


def load_tray_settings() -> dict:
    try:
        return json.loads(TRAY_SETTINGS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_tray_settings(settings: dict) -> None:
    temporary = TRAY_SETTINGS.with_suffix(".tmp")
    temporary.write_text(json.dumps(settings, indent=1), encoding="utf-8")
    temporary.replace(TRAY_SETTINGS)


def update_tray_settings(**changes) -> None:
    settings = load_tray_settings()
    settings.update(changes)
    save_tray_settings(settings)


def write_status(state: str, text: str, last_action: str) -> None:
    status = {"state": state, "text": text, "lastAction": last_action, "pid": os.getpid(),
              "updated": datetime.now().isoformat(timespec="seconds")}
    try:
        temporary = STATUS_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(status), encoding="utf-8")
        temporary.replace(STATUS_PATH)
    except OSError:
        log.exception("could not write tray status")


def open_settings_window() -> None:
    """Starts the settings window without a console."""
    pythonw = Path(os.sys.executable).with_name("pythonw.exe")
    subprocess.Popen([str(pythonw), str(ROOT / "plaudit_settings.pyw")], cwd=ROOT)


class TrayApp:
    def __init__(self, device: str | None = None, mode: str = "raw"):
        self.device_selector = device
        self.mode = mode
        self.engine: PatternEngine | None = None
        self.state = "starting"
        self.status_text = "Starting..."
        self.last_action = ""
        preferences = load_tray_settings()
        self.notify_actions = bool(preferences.get("notifyActions", False))
        self.paused = bool(preferences.get("paused", False))
        self._preferences_mtime = self._preferences_stamp()
        self._stop = threading.Event()
        self._flash_until = 0.0
        self.icon = pystray.Icon("plaudit", icon_image("starting"), "Plaudit - starting", menu=self._menu())
        if self.paused:
            self.state, self.status_text = "paused", "Paused - microphone off"

    # -- menu -----------------------------------------------------------------------------------

    def _menu(self) -> pystray.Menu:
        item = pystray.MenuItem
        return pystray.Menu(
            item(lambda _: self.status_text, None, enabled=False),
            item(lambda _: f"Last: {self.last_action}" if self.last_action else "No actions yet", None, enabled=False),
            pystray.Menu.SEPARATOR,
            item(lambda _: "Resume listening" if self.paused else "Pause listening", self.toggle_pause),
            item("Open Plaudit...", self.open_settings),
            item("Open logs folder", self.open_logs),
            pystray.Menu.SEPARATOR,
            item("Notify when an action runs", self.toggle_notify, checked=lambda _: self.notify_actions),
            item("Start with Windows", self.toggle_startup, checked=lambda _: startup.is_enabled()),
            pystray.Menu.SEPARATOR,
            item("Quit Plaudit", self.quit),
        )

    def toggle_pause(self, *_):
        self._apply_pause(not self.paused)
        update_tray_settings(paused=self.paused)
        self._preferences_mtime = self._preferences_stamp()

    def _apply_pause(self, paused: bool) -> None:
        self.paused = paused
        self._set_state("paused" if paused else "starting",
                        "Paused - microphone off" if paused else "Starting...")

    @staticmethod
    def _preferences_stamp() -> float:
        try:
            return TRAY_SETTINGS.stat().st_mtime
        except OSError:
            return 0.0

    def _follow_preferences(self) -> None:
        """Applies pause/notification changes made in the settings window."""
        stamp = self._preferences_stamp()
        if stamp == self._preferences_mtime:
            return
        self._preferences_mtime = stamp
        preferences = load_tray_settings()
        self.notify_actions = bool(preferences.get("notifyActions", self.notify_actions))
        paused = bool(preferences.get("paused", False))
        if paused != self.paused:
            self._apply_pause(paused)

    def open_settings(self, *_):
        open_settings_window()

    def open_logs(self, *_):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(LOGS_DIR)

    def toggle_notify(self, *_):
        self.notify_actions = not self.notify_actions
        update_tray_settings(notifyActions=self.notify_actions)
        self._preferences_mtime = self._preferences_stamp()

    def toggle_startup(self, *_):
        try:
            if startup.is_enabled():
                startup.disable()
            else:
                startup.enable()
        except OSError as error:
            self._notify(f"Couldn't change the startup setting: {error}")

    def quit(self, *_):
        self._stop.set()
        write_status("stopped", "Not running", self.last_action)
        self.icon.stop()

    # -- state ----------------------------------------------------------------------------------

    def _set_state(self, state: str, text: str) -> None:
        if (state, text) != (self.state, self.status_text):
            log.info("state: %s - %s", state, text)
        self.state, self.status_text = state, text
        self._refresh()

    def _refresh(self) -> None:
        shown = "action" if time.monotonic() < self._flash_until else self.state
        self.icon.icon = icon_image(shown)
        title = f"Plaudit - {self.status_text}"
        if self.last_action:
            title += f"\nLast: {self.last_action}"
        self.icon.title = title[:127]
        self.icon.update_menu()
        if not self._stop.is_set():
            write_status(self.state, self.status_text, self.last_action)

    def _notify(self, message: str) -> None:
        try:
            self.icon.notify(message, "Plaudit")
        except Exception:  # notifications are best-effort
            log.exception("notification failed")

    # -- worker ---------------------------------------------------------------------------------

    def run(self) -> None:
        self.icon.run(setup=self._setup)

    def _setup(self, icon) -> None:
        icon.visible = True
        self._refresh()
        threading.Thread(target=self._work, name="plaudit-listener", daemon=True).start()

    def _work(self) -> None:
        flashing = False
        while not self._stop.is_set():
            try:
                self._follow_preferences()
                if self.paused:
                    if self.engine is not None:
                        self.engine.stop()
                    time.sleep(0.2)
                    continue
                if self.engine is None:
                    self.engine = PatternEngine(resolve_device(self.device_selector), self.mode)
                if not self.engine.listening:
                    self._set_state("starting", "Starting - learning the room...")
                    self.engine.start()
                    while self.engine.warming_up and not self._stop.is_set() and not self.paused:
                        self.engine.step()
                    if not self.paused:
                        self._set_state("listening", "Listening")
                    continue
                if self.engine.reload_settings_if_changed():
                    log.info("patterns.json changed; actions reloaded")
                update = self.engine.step()
                for result in update.bursts:
                    log.info("burst: %s %s", result.outcome, result.action)
                    if result.fired:
                        self.last_action = f"{result.outcome} {result.action} ({datetime.now():%H:%M})"
                        self._flash_until = time.monotonic() + FLASH_S
                        flashing = True
                        self._refresh()
                        if self.notify_actions:
                            self._notify(f"{result.outcome} → {result.action}")
                    elif result.failed:
                        self._notify(f"{result.outcome} action {result.action}")
                for error in update.errors:
                    log.warning(error)
                    self._notify(error)
                if flashing and time.monotonic() >= self._flash_until:
                    flashing = False
                    self._refresh()
            except Exception as error:  # keep the tray alive; retry after any failure
                log.exception("listening failed")
                if self.engine is not None:
                    self.engine.stop()
                self._set_state("error", f"Not listening: {error}"[:100])
                self._notify(f"Plaudit stopped listening: {error}. Retrying in {RETRY_S:.0f} s.")
                self._stop.wait(RETRY_S)
        if self.engine is not None:
            self.engine.close()


def main(device: str | None = None, mode: str = "raw") -> int:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(LOGS_DIR / "tray.log", maxBytes=500_000, backupCount=1, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    if already_running():
        log.info("another Plaudit tray app is already running; exiting")
        return 1
    log.info("tray app starting")
    try:
        TrayApp(device, mode).run()
    except Exception:
        log.exception("tray app crashed")
        raise
    log.info("tray app stopped")
    return 0
