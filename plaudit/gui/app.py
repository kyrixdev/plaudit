"""Opens the settings window."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import webview

from ..livelog import LOGS_DIR
from ..windowing import activate_settings_window, claim_single_instance, release_single_instance
from .api import SettingsApi

WEB = Path(__file__).resolve().parent / "web"
log = logging.getLogger("plaudit.settings")


def main() -> int:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(LOGS_DIR / "settings.log", maxBytes=500_000, backupCount=1, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])  # pywebview's own errors land here too
    if not claim_single_instance():
        log.info("settings window already open; bringing it to the front")
        activate_settings_window()
        return 0
    log.info("settings window starting")
    api = SettingsApi()
    window = webview.create_window(
        "Plaudit",
        url=str(WEB / "index.html"),
        js_api=api,
        width=920,
        height=640,
        min_size=(760, 540),
        background_color="#F2F2F5",
        text_select=False,
    )
    api.attach(window)
    try:
        webview.start()
    except Exception:
        log.exception("settings window crashed")
        raise
    finally:
        release_single_instance()
    log.info("settings window closed")
    return 0
