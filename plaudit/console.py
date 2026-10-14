"""Terminal helpers: colors, keyboard polling, the desk diagram and the audio listener loop."""

from __future__ import annotations

import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass

import numpy as np

from .audio import InputDevice, WavWriter, open_input
from .detector import ImpactMetrics, StreamingTapDetector
from .features import TapFeatureExtractor
from .models import DeskZone, DetectedTap, TapFeatureVector
from .onsets import FastOnsetDetector, Onset

try:
    import msvcrt
except ImportError:  # pragma: no cover - Windows-only tool, but keep imports testable elsewhere
    msvcrt = None


def init_terminal() -> None:
    if os.name == "nt":
        os.system("")  # enables ANSI escape handling in classic consoles
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"


def paint(text: str, *styles: str) -> str:
    return "".join(styles) + text + C.RESET


_ANSI = re.compile(r"\033\[[0-9;]*[A-Za-z]")


def fit(text: str) -> str:
    """Trims a line to the terminal width. A wrapped line can't be redrawn in place, which
    garbles the live status line, so an overlong line loses its colours and ends in '...'."""
    width = shutil.get_terminal_size((120, 30)).columns - 1
    plain = _ANSI.sub("", text)
    if len(plain) <= width:
        return text
    return plain[: max(width - 3, 1)] + "..."


def status(text: str) -> None:
    """Rewrites the current terminal line."""
    sys.stdout.write("\r\033[2K" + fit(text))
    sys.stdout.flush()


def println(text: str = "") -> None:
    sys.stdout.write("\r\033[2K" + text + "\n")
    sys.stdout.flush()


def read_key() -> str | None:
    if msvcrt is None or not msvcrt.kbhit():
        return None
    key = msvcrt.getwch()
    if key in ("\x00", "\xe0"):  # arrow/function key prefix
        msvcrt.getwch()
        return None
    return key


def drain_keys() -> None:
    while read_key() is not None:
        pass


def clean_device_name(name: str) -> str:
    return name.replace("�", "").replace("  ", " ")


def dbfs(value: float) -> float:
    return 20 * math.log10(max(value, 1e-9))


def level_bar(value: float, width: int = 24) -> str:
    """Maps -80..0 dBFS onto a bar."""
    fraction = min(max((dbfs(value) + 80) / 80, 0), 1)
    filled = int(round(fraction * width))
    return "█" * filled + "·" * (width - filled)


def desk_diagram(active: DeskZone | None = None, done: set[DeskZone] | None = None) -> str:
    done = done or set()

    def mark(zone: DeskZone) -> str:
        if zone == active:
            return paint(f"[{zone.short_name}]", C.BOLD, C.CYAN)
        if zone in done:
            return paint(f" {zone.short_name}✓", C.GREEN)
        return paint(f" {zone.short_name} ", C.DIM)

    return "\n".join([
        f"   {mark(DeskZone.LEFT_REAR)}    ┌────────────────┐    {mark(DeskZone.RIGHT_REAR)}",
        "           │     screen     │",
        "           ├────────────────┤",
        f"   {mark(DeskZone.LEFT_FRONT)}    │    keyboard    │    {mark(DeskZone.RIGHT_FRONT)}",
        "           └────────────────┘",
    ])


@dataclass
class TapEvent:
    event: DetectedTap
    feature: TapFeatureVector
    seconds: float  # stream time of the onset
    processing_ms: float
    block_time: float = 0.0  # perf_counter() when the completing audio block arrived
    onset_age_s: float = 0.0  # audio between the onset and the end of that block, plus driver latency

    def response_ms(self) -> float:
        """Time from the tap reaching the mic to now (call right after classifying)."""
        return (time.perf_counter() - self.block_time + self.onset_age_s) * 1000


@dataclass
class GateRejection:
    event: DetectedTap
    metrics: ImpactMetrics | None
    seconds: float


class Listener:
    """Pulls audio blocks, runs detection and feature extraction, and reports taps."""

    def __init__(self, device: InputDevice, mode: str, record_path=None):
        self._open_audio(device, mode, record_path)
        self.detector = StreamingTapDetector(self.sample_rate, self.channels)
        self.extractor = TapFeatureExtractor(self.sample_rate)

    def _open_audio(self, device: InputDevice, mode: str, record_path) -> None:
        self.audio, self.warning = open_input(device, mode)
        self.device = device
        self.mode = self.audio.mode
        self.sample_rate = self.audio.sample_rate
        self.channels = self.audio.channels
        self.recorder = WavWriter(record_path, self.sample_rate, self.channels) if record_path else None
        self.level = 0.0
        self._warmup_blocks: list[np.ndarray] = []

    @property
    def stream_seconds(self) -> float:
        return self.detector.total_samples / self.sample_rate

    def _note_block(self, block: np.ndarray) -> None:
        if self.recorder:
            self.recorder.write(block)
        if self.detector.is_warming_up:
            self._warmup_blocks.append(block)
        self.level = 0.8 * self.level + 0.2 * float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))

    def poll(self, timeout: float = 0.05) -> tuple[list[TapEvent], list[GateRejection]]:
        taps: list[TapEvent] = []
        rejections: list[GateRejection] = []
        block = self.audio.read(timeout)
        while block is not None:
            started = time.perf_counter()
            self._note_block(block)
            before = len(self.detector.gate_rejections)
            events = self.detector.process(block)
            block_end = self.detector.total_samples
            for event in events:
                feature = self.extractor.extract(event)
                onset_age = (block_end - event.stream_sample_index) / self.sample_rate + self.audio.input_latency
                taps.append(TapEvent(event, feature, event.stream_sample_index / self.sample_rate,
                                     (time.perf_counter() - started) * 1000,
                                     self.audio.last_block_time, onset_age))
            for event, metrics in self.detector.gate_rejections[before:]:
                rejections.append(GateRejection(event, metrics, event.stream_sample_index / self.sample_rate))
            block = self.audio.read(0)
        return taps, rejections

    def flush(self) -> None:
        """Discards audio queued while the user was in a menu."""
        while self.audio.read(0) is not None:
            pass

    @property
    def warming_up(self) -> bool:
        return self.detector.is_warming_up

    def channel_report(self) -> str | None:
        """Describes whether the two mic channels carry independent signals."""
        if self.channels < 2 or not self._warmup_blocks:
            return None
        audio = np.concatenate(self._warmup_blocks, axis=1).astype(np.float64)
        if np.max(np.abs(audio[0] - audio[1])) < 1e-7:
            return paint("Both channels are identical", C.YELLOW) + \
                " (Windows is merging the mics) - left/right cues will be unavailable."
        corr = float(np.corrcoef(audio[0], audio[1])[0, 1])
        return paint("Two independent mic channels", C.GREEN) + f" (correlation {corr:.2f}) - left/right cues available."

    @property
    def channel_balance(self) -> float:
        """log10(mic 1 / mic 2) loudness of the room noise, i.e. the mics' sensitivity mismatch."""
        if self.channels < 2 or not self._warmup_blocks:
            return 0.0
        audio = np.concatenate(self._warmup_blocks, axis=1).astype(np.float64)
        levels = np.sqrt(np.mean(audio[:2] ** 2, axis=1))
        return math.log10((levels[0] + 1e-12) / (levels[1] + 1e-12))

    def close(self) -> None:
        self.audio.stop()
        if self.recorder:
            self.recorder.close()


class OnsetListener(Listener):
    """For counting claps, snaps and knocks: fast onsets instead of Holo's tap windows."""

    def __init__(self, device: InputDevice, mode: str, record_path=None):
        self._open_audio(device, mode, record_path)
        self.detector = FastOnsetDetector(self.sample_rate)

    def poll(self, timeout: float = 0.05) -> tuple[list[Onset], list[Onset]]:
        """Returns (accepted onsets, onsets rejected as sustained sounds)."""
        accepted: list[Onset] = []
        rejected: list[Onset] = []
        block = self.audio.read(timeout)
        while block is not None:
            self._note_block(block)
            for onset in self.detector.process(block):
                (accepted if onset.accepted else rejected).append(onset)
            block = self.audio.read(0)
        return accepted, rejected
