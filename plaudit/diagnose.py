"""Plaudit.exe --diagnose: writes an audio report to logs/diagnostics.txt (handy for bug reports)."""

from __future__ import annotations

import platform
import sys
import threading
import time
import traceback
from datetime import datetime

from . import __version__
from .livelog import LOGS_DIR


def _try_open(lines: list[str], label: str) -> None:
    from .audio import AudioInput, AudioSetupError, default_input_device
    try:
        device = default_input_device()
        lines.append(f"  [{label}] default input #{device.index}: {device.name} ({device.channels} ch, "
                     f"{device.default_sample_rate:.0f} Hz)")
    except Exception as error:
        lines.append(f"  [{label}] no default input: {error}")
        return
    for mode in ("raw", "shared"):
        try:
            audio = AudioInput(device, mode).start()
            time.sleep(0.4)
            block = audio.read(0.5)
            audio.stop()
            lines.append(f"  [{label}] {mode}: OK, got {'no audio' if block is None else block.shape}")
        except AudioSetupError as error:
            lines.append(f"  [{label}] {mode}: FAILED {error}")
        except Exception:
            lines.append(f"  [{label}] {mode}: CRASHED\n" + traceback.format_exc())


def report() -> str:
    lines = [f"Plaudit {__version__} diagnostics, {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"Python {sys.version.split()[0]} ({'frozen' if getattr(sys, 'frozen', False) else 'source'}), "
             f"Windows {platform.version()}"]
    try:
        import sounddevice as sd
        lines.append(f"sounddevice {sd.__version__}, {sd.get_portaudio_version()[1]}")
        lines.append("Host APIs: " + ", ".join(f"{i}={a['name']}" for i, a in enumerate(sd.query_hostapis())))
        for index, info in enumerate(sd.query_devices()):
            if info["max_input_channels"] > 0:
                api = sd.query_hostapis(info["hostapi"])["name"]
                lines.append(f"  input #{index}: {info['name']} [{api}, {info['max_input_channels']} ch]")
    except Exception:
        lines.append("sounddevice failed:\n" + traceback.format_exc())
    lines.append("Opening the microphone:")
    _try_open(lines, "main thread")
    worker = threading.Thread(target=_try_open, args=(lines, "worker thread"))
    worker.start()
    worker.join(15)
    return "\n".join(lines) + "\n"


def main() -> int:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / "diagnostics.txt"
    path.write_text(report(), encoding="utf-8")
    return 0
