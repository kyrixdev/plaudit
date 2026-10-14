"""The install and uninstall screens (pywebview) and the steps behind them.

Both screens share web/setup.html; JavaScript asks get_info() which flow to show.
Like the settings window's API, only methods are public: pywebview exposes every
public attribute to JavaScript and walks into nested objects.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

from .. import REPO_URL, __version__, startup
from ..paths import APP_DIR, DATA_DIR, FROZEN
from . import system

WEB = Path(__file__).resolve().parent / "web"


class _Flow:
    """Shared progress reporting and window controls."""

    mode = ""

    def __init__(self):
        self._window = None
        self._progress = {"percent": 0, "message": "", "done": False, "error": None}
        self._lock = threading.Lock()

    def attach(self, window) -> None:
        self._window = window

    def progress(self) -> dict:
        with self._lock:
            return dict(self._progress)

    def _report(self, percent: float, message: str | None = None) -> None:
        with self._lock:
            self._progress["percent"] = round(min(max(percent, 0), 100))
            if message is not None:
                self._progress["message"] = message

    def _finish(self, error: str | None = None) -> None:
        with self._lock:
            self._progress.update(done=True, error=error, percent=100 if error is None else self._progress["percent"])

    def _run(self, work) -> dict:
        with self._lock:
            self._progress = {"percent": 0, "message": "Preparing...", "done": False, "error": None}

        def runner():
            try:
                work()
                self._finish()
            except Exception as error:  # shown on the error screen
                self._finish(str(error) or error.__class__.__name__)

        threading.Thread(target=runner, daemon=True).start()
        return {"ok": True}

    def open_repo(self) -> None:
        webbrowser.open(REPO_URL)

    def minimize(self) -> None:
        if self._window is not None:
            self._window.minimize()

    def close(self) -> None:
        if self._window is not None:
            self._window.destroy()


class SetupApi(_Flow):
    mode = "install"

    def __init__(self, payload: Path, install_dir: Path | None = None):
        super().__init__()
        self._payload = payload
        self._install_dir = install_dir or system.default_install_dir()

    def get_info(self) -> dict:
        return {"mode": self.mode, "version": __version__, "installDir": str(self._install_dir),
                "existingVersion": system.installed_version(), "repo": REPO_URL}

    def install(self, options: dict) -> dict:
        return self._run(lambda: self._install(options or {}))

    def _install(self, options: dict) -> None:
        if not self._payload.is_file():
            raise system.InstallError("The installer is missing its app files. Download it again.")
        target = self._install_dir
        self._report(3, "Closing Plaudit if it's running...")
        system.stop_running()
        self._report(8, "Removing the previous version...")
        system.remove_previous(target)
        self._report(10, "Copying files...")
        system.extract(self._payload, target, lambda f: self._report(10 + f * 75))
        exe = target / system.EXE_NAME
        if not exe.is_file():
            raise system.InstallError("The app files are incomplete. Download the installer again.")
        self._report(88, "Creating shortcuts...")
        system.create_shortcut(system.start_menu_shortcut(), exe, "Clap, snap or knock to control your PC")
        if options.get("desktopShortcut", True):
            system.create_shortcut(system.desktop_shortcut(), exe, "Clap, snap or knock to control your PC")
        else:
            system.desktop_shortcut().unlink(missing_ok=True)
        self._report(94, "Registering with Windows...")
        system.register(target, __version__)
        if options.get("startWithWindows", True):
            startup.enable(command=f'"{exe}" --tray')
        else:
            startup.disable()
        self._report(100, "Done")

    def launch(self) -> None:
        exe = self._install_dir / system.EXE_NAME
        if exe.is_file():
            subprocess.Popen([str(exe)], cwd=exe.parent, close_fds=True)


class UninstallApi(_Flow):
    mode = "uninstall"

    def get_info(self) -> dict:
        return {"mode": self.mode, "version": __version__, "installDir": str(APP_DIR), "dataDir": str(DATA_DIR),
                "repo": REPO_URL}

    def uninstall(self, remove_data: bool = False) -> dict:
        return self._run(lambda: self._uninstall(bool(remove_data)))

    def _uninstall(self, remove_data: bool) -> None:
        # Destructive steps run only from an installed Plaudit.exe in the install location,
        # never from source or a build folder.
        installed = (FROZEN and system.looks_like_install_dir(APP_DIR)
                     and APP_DIR.resolve() == system.default_install_dir().resolve())
        if not installed:
            raise system.InstallError("This copy of Plaudit isn't installed (it runs from source), so there's "
                                      "nothing to uninstall.")
        self._report(10, "Closing Plaudit...")
        system.stop_running(exclude_pid=os.getpid())
        self._report(35, "Removing shortcuts and startup entry...")
        startup.disable()
        system.remove_shortcuts()
        self._report(60, "Removing Plaudit from Installed apps...")
        system.unregister()
        appdata = Path(os.environ.get("APPDATA", ""))
        if remove_data and DATA_DIR.name == system.APP_NAME and DATA_DIR.parent == appdata and DATA_DIR.exists():
            self._report(80, "Removing your settings and logs...")
            shutil.rmtree(DATA_DIR, ignore_errors=True)
        self._report(95, "Finishing...")
        system.remove_after_exit(APP_DIR, os.getpid())  # the app's own files go once this window closes
        self._report(100, "Done")


def _open(api: _Flow, title: str) -> int:
    import webview

    window = webview.create_window(title, url=str(WEB / "setup.html"), js_api=api, width=760, height=500,
                                   resizable=False, frameless=True, easy_drag=False, background_color="#1c1c1e")
    api.attach(window)
    webview.start()
    return 0


def run_installer(payload: Path) -> int:
    return _open(SetupApi(payload), "Plaudit Setup")


def run_uninstaller() -> int:
    return _open(UninstallApi(), "Uninstall Plaudit")


def setup_main() -> int:
    """Entry point of Plaudit-Setup.exe: the app files travel inside it as payload.zip."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return run_installer(base / "payload.zip")
