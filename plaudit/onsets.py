"""Fast onset detection for counting claps, snaps and knocks.

Holo's detector captures a 90 ms window per event and then ignores 140 ms so the
event's own ringing isn't counted twice: 230 ms of deafness, which merges fast
claps. Counting needs no analysis window, so this detector instead follows the
fading envelope of each sound (its echo and ringing only ever decay) and reports
a new onset whenever the level jumps well above that envelope. Sounds ~0.1 s
apart stay separate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .dsp import cascade_lowpass, onset_lowpass_alpha

FRAME_S = 0.001
DECAY_TAU_S = 0.07  # how fast the follower forgets a sound; a room's clap echo fades slower than this
RISE_FACTOR = 2.5  # a new onset must be this much louder than the fading previous sound
FLOOR_FACTOR = 10.0  # ...and this much louder than the room (20 dB)
MIN_GAP_S = 0.08
SUSTAIN_WINDOW_S = 0.06  # how long after an onset its shape is judged
SUSTAIN_LEVEL = 0.4  # "sustained" frames stay above this fraction of the peak...
MAX_SUSTAINED_FRACTION = 0.5  # ...and a clap/knock keeps under half its first 60 ms there; a syllable doesn't
PEAK_WINDOW_S = 0.015


@dataclass
class Onset:
    seconds: float  # stream time of the onset
    peak: float  # peak of the onset-path envelope in the first 15 ms
    snr_db: float  # peak over the room's noise floor
    sustained_fraction: float
    accepted: bool  # False when it looks like a sustained sound (e.g. a syllable of speech)


class FastOnsetDetector:
    def __init__(self, sample_rate: float, warm_up_duration: float = 0.75):
        self.sample_rate = float(sample_rate)
        self.hop = max(int(round(self.sample_rate * FRAME_S)), 1)
        self.total_samples = 0
        self._alpha = onset_lowpass_alpha(self.sample_rate)
        self._filter_state = [0.0] * 4
        self._leftover = np.zeros(0)
        self._frame_index = 0
        self._warm_up_frames = int(warm_up_duration / FRAME_S)
        self._warm_up_values: list[float] = []
        self._floor = 0.0
        self._follower = 0.0
        self._decay = math.exp(-FRAME_S / DECAY_TAU_S)
        self._last_onset_frame = -10**9
        self._pending: list[dict] = []  # onsets still collecting their first 60 ms

    @property
    def is_warming_up(self) -> bool:
        return self._frame_index < self._warm_up_frames

    @property
    def noise_floor_rms(self) -> float:
        """Room level (named to match StreamingTapDetector for shared display code)."""
        return self._floor

    def process(self, incoming: np.ndarray) -> list[Onset]:
        """Feeds a (channels, frames) block. Returns onsets whose first 60 ms are complete."""
        block = np.atleast_2d(np.asarray(incoming, dtype=np.float64))
        if block.shape[1] == 0:
            return []
        mono = block.mean(axis=0)
        self.total_samples += block.shape[1]
        filtered = np.concatenate([self._leftover, cascade_lowpass(mono, self._alpha, self._filter_state)])
        usable = len(filtered) // self.hop * self.hop
        self._leftover = filtered[usable:]
        envelope = np.abs(filtered[:usable]).reshape(-1, self.hop).max(axis=1) if usable else np.zeros(0)
        finished: list[Onset] = []
        for value in envelope:
            finished += self._frame(float(value))
        return finished

    def _frame(self, value: float) -> list[Onset]:
        index = self._frame_index
        self._frame_index += 1
        if index < self._warm_up_frames:
            self._warm_up_values.append(value)
            if index == self._warm_up_frames - 1:
                self._floor = max(float(np.median(self._warm_up_values)), 1e-5)
            self._follower = max(self._follower * self._decay, value)
            return []

        for event in self._pending:
            event["values"].append(value)
        threshold = max(self._floor * FLOOR_FACTOR, self._follower * RISE_FACTOR)
        min_gap_frames = int(MIN_GAP_S / FRAME_S)
        if value > threshold and index - self._last_onset_frame >= min_gap_frames:
            self._last_onset_frame = index
            self._pending.append({"frame": index, "floor": self._floor, "values": [value]})
        elif value < 3 * self._floor:
            self._floor = 0.999 * self._floor + 0.001 * value  # follow slow changes in the room
        self._follower = max(self._follower * self._decay, value)

        done, keep = [], []
        window = int(SUSTAIN_WINDOW_S / FRAME_S)
        for event in self._pending:
            (done if len(event["values"]) >= window else keep).append(event)
        self._pending = keep
        return [self._finish(event) for event in done]

    def _finish(self, event: dict) -> Onset:
        # MIN_GAP_S exceeds SUSTAIN_WINDOW_S, so a following onset never falls inside this window.
        values = np.asarray(event["values"])
        peak = float(values[: int(PEAK_WINDOW_S / FRAME_S)].max())
        sustained = float(np.mean(values >= SUSTAIN_LEVEL * peak))
        return Onset(
            seconds=event["frame"] * FRAME_S,
            peak=peak,
            snr_db=20 * math.log10(max(peak, 1e-9) / max(event["floor"], 1e-9)),
            sustained_fraction=sustained,
            accepted=sustained <= MAX_SUSTAINED_FRACTION,
        )
