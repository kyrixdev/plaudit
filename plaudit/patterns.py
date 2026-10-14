"""Tap patterns: run an action by tapping the desk N times, anywhere.

Location classification turned out to drift between sessions on this hardware,
while tap *detection* stayed reliable. Counting taps needs only detection, so it
needs no calibration and doesn't depend on where or how hard you tap.
"""

from __future__ import annotations

import ctypes
import json
from dataclasses import dataclass, field
from pathlib import Path

from .actions import ZoneAction
from .paths import DATA_DIR

SETTINGS_PATH = DATA_DIR / "patterns.json"
COUNTS = (2, 3, 4, 5)  # a single tap is too easy to trigger by accident
MAX_GAP_S = 0.70  # a burst ends after this much quiet
REBOUND_GAP_S = 0.30  # the desk's rebound arrives just after the detector's ~230 ms dead time...
REBOUND_MAX_RATIO = 0.35  # ...and is much quieter than the tap that caused it
MIN_SNR_DB = 6.0
MIN_PEAK = 0.01
INPUT_GUARD_S = 0.5  # keyboard/mouse use this long before a burst (or during it) cancels the burst


@dataclass
class PatternSettings:
    actions: dict[int, ZoneAction] = field(default_factory=lambda: {n: ZoneAction() for n in COUNTS})
    max_gap_s: float = MAX_GAP_S
    min_snr_db: float = MIN_SNR_DB
    ignore_while_typing: bool = True

    def action(self, count: int) -> ZoneAction:
        return self.actions.get(count, ZoneAction())

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "actions": {str(n): self.action(n).to_dict() for n in COUNTS},
            "maxGapSeconds": self.max_gap_s,
            "minSnrDb": self.min_snr_db,
            "ignoreWhileTyping": self.ignore_while_typing,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PatternSettings":
        saved = data.get("actions") or {}
        return cls(
            actions={n: ZoneAction.from_dict(saved.get(str(n))) for n in COUNTS},
            max_gap_s=float(data.get("maxGapSeconds", MAX_GAP_S)),
            min_snr_db=float(data.get("minSnrDb", MIN_SNR_DB)),
            ignore_while_typing=bool(data.get("ignoreWhileTyping", True)),
        )

    @classmethod
    def load(cls, path: Path = SETTINGS_PATH) -> "PatternSettings":
        if not path.exists():
            return cls()
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path = SETTINGS_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        temporary.replace(path)
        return path


def onset_counter(max_gap_s: float = MAX_GAP_S) -> "TapCounter":
    """A counter for FastOnsetDetector onsets. Its detection threshold (20 dB over the room)
    and decaying follower already exclude quiet sounds and echoes, so the counter's own
    loudness and rebound rules are off; they would also drop a softer fast second clap."""
    return TapCounter(max_gap_s, min_snr_db=0.0, min_peak=0.0, rebound_max_ratio=0.0)


@dataclass
class Burst:
    count: int
    start_s: float
    end_s: float
    peaks: list[float]


class TapCounter:
    """Groups detected taps into bursts. Times are audio-stream seconds."""

    def __init__(self, max_gap_s: float = MAX_GAP_S, min_snr_db: float = MIN_SNR_DB, min_peak: float = MIN_PEAK,
                 rebound_max_ratio: float = REBOUND_MAX_RATIO):
        self.max_gap_s = max_gap_s
        self.min_snr_db = min_snr_db
        self.min_peak = min_peak
        self.rebound_max_ratio = rebound_max_ratio
        self._times: list[float] = []
        self._peaks: list[float] = []

    @property
    def pending(self) -> int:
        return len(self._times)

    def add(self, time_s: float, peak: float, snr_db: float) -> str:
        """Returns 'counted', 'too quiet' or 'rebound'."""
        if snr_db < self.min_snr_db or peak < self.min_peak:
            return "too quiet"
        if self._times:
            gap = time_s - self._times[-1]
            if gap < REBOUND_GAP_S and peak < self._peaks[-1] * self.rebound_max_ratio:
                return "rebound"
        self._times.append(time_s)
        self._peaks.append(peak)
        return "counted"

    def poll(self, now_s: float) -> Burst | None:
        """Returns a finished burst once max_gap_s has passed since its last tap."""
        if not self._times or now_s - self._times[-1] < self.max_gap_s:
            return None
        burst = Burst(len(self._times), self._times[0], self._times[-1], list(self._peaks))
        self._times.clear()
        self._peaks.clear()
        return burst

    def reset(self) -> None:
        self._times.clear()
        self._peaks.clear()


def seconds_since_input() -> float:
    """Seconds since the last keyboard or mouse input anywhere on the PC."""

    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint32)]

    info = LASTINPUTINFO(ctypes.sizeof(LASTINPUTINFO), 0)
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
        return float("inf")
    kernel32 = ctypes.windll.kernel32
    kernel32.GetTickCount.restype = ctypes.c_uint32
    return ((kernel32.GetTickCount() - info.dwTime) & 0xFFFFFFFF) / 1000.0


def burst_overlaps_input(burst: Burst, now_s: float, idle_s: float) -> bool:
    """True if keyboard/mouse input happened from INPUT_GUARD_S before the burst until now."""
    return idle_s < (now_s - burst.start_s) + INPUT_GUARD_S
