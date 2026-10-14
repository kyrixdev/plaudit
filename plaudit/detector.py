"""Streaming onset detection and the sustained-sound gate.

Ported from HoloCore's StreamingTapDetector.swift and ImpactEventGate.swift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .dsp import cascade_lowpass, onset_lowpass_alpha, rms
from .models import DetectedTap


@dataclass
class ImpactMetrics:
    onset_contrast: float
    effective_duration_s: float
    early_energy_fraction: float
    late_to_impact_rms: float


def impact_metrics(event: DetectedTap, sample_rate: float) -> ImpactMetrics | None:
    if sample_rate <= 0 or event.channels.size == 0:
        return None
    mono = np.asarray(event.channels, dtype=np.float64).mean(axis=0)
    if len(mono) == 0:
        return None

    # Match the detector's sub-6 kHz onset path.
    filtered = cascade_lowpass(mono, onset_lowpass_alpha(sample_rate), [0.0] * 4)
    n = len(filtered)
    onset = min(max(event.onset_offset, 0), n - 1)
    frame_samples = max(int(sample_rate * 0.005), 32)

    frame_rms: list[float] = []
    frame_lengths: list[int] = []
    offset = onset
    while offset < n:
        end = min(offset + frame_samples, n)
        frame_rms.append(rms(filtered[offset:end]))
        frame_lengths.append(end - offset)
        offset = end
    if not frame_rms:
        return None

    impact_frames = min(len(frame_rms), max(math.ceil(0.025 * sample_rate / frame_samples), 1))
    impact_rms = max(frame_rms[:impact_frames])
    if impact_rms <= 0:
        return None

    pre_start = max(0, onset - int(sample_rate * 0.012))
    pre_rms = rms(filtered[pre_start:onset])
    reference_floor = max(pre_rms, event.noise_floor_rms, 0.000_001)
    impact_end = min(onset + impact_frames * frame_samples, n)
    impact_peak = float(np.max(np.abs(filtered[onset:impact_end]))) if impact_end > onset else 0.0
    # A fingertip impact may be only a few samples wide; keep a peak alternative.
    onset_contrast = max(impact_rms / reference_floor, 0.45 * impact_peak / reference_floor)

    effective_threshold = max(impact_rms * 0.40, event.noise_floor_rms * 2.2)
    effective_samples = sum(length for value, length in zip(frame_rms, frame_lengths) if value >= effective_threshold)

    early_end = min(onset + int(sample_rate * 0.025), n)
    tail = filtered[onset:]
    total_energy = float(np.dot(tail, tail))
    early = filtered[onset:early_end]
    early_fraction = float(np.dot(early, early)) / max(total_energy, 1e-15)

    late_start = min(onset + int(sample_rate * 0.040), n)
    late_rms = rms(filtered[late_start:])

    return ImpactMetrics(
        onset_contrast=onset_contrast,
        effective_duration_s=effective_samples / sample_rate,
        early_energy_fraction=early_fraction,
        late_to_impact_rms=late_rms / max(impact_rms, 1e-12),
    )


def impact_gate_accepts(event: DetectedTap, sample_rate: float, metrics: ImpactMetrics | None = None) -> bool:
    """Rejects candidates that are clearly sustained (speech-like) rather than an impact."""
    metrics = metrics or impact_metrics(event, sample_rate)
    if metrics is None:
        return False
    # A contact impact should emerge clearly from the preceding audio.
    if metrics.onset_contrast < 1.8:
        return False
    # All three sustained cues are required, which keeps resonant desks usable.
    clearly_sustained = (
        metrics.effective_duration_s >= 0.040
        and metrics.late_to_impact_rms >= 0.36
        and metrics.early_energy_fraction < 0.60
    )
    return not clearly_sustained


class StreamingTapDetector:
    """Adapts to the room's noise floor, keeps a short pre-roll and emits fixed-length windows."""

    def __init__(
        self,
        sample_rate: float,
        channel_count: int,
        analysis_duration: float = 0.090,
        pre_roll_duration: float = 0.012,
        warm_up_duration: float = 0.75,
        initial_noise_floor_rms: float = 0.0005,
    ):
        self.sample_rate = float(sample_rate)
        self.channel_count = max(int(channel_count), 1)
        self.analysis_window_samples = max(int(sample_rate * analysis_duration), 1_024)
        self.pre_roll_samples = max(int(sample_rate * pre_roll_duration), 128)
        self.warm_up_samples = max(int(sample_rate * warm_up_duration), 0)
        self._initial_noise_floor = max(initial_noise_floor_rms, 0.000_01)
        self._alpha = onset_lowpass_alpha(self.sample_rate)
        # Candidates the impact gate dropped, kept for diagnostics: (event, metrics).
        self.gate_rejections: list[tuple[DetectedTap, ImpactMetrics | None]] = []
        self.reset()

    @property
    def is_warming_up(self) -> bool:
        return self._warm_up_remaining > 0

    def reset(self) -> None:
        self.total_samples = 0
        self.noise_floor_rms = self._initial_noise_floor
        self._pre_roll = np.zeros((self.channel_count, 0), dtype=np.float32)
        self._capture: np.ndarray | None = None
        self._capture_onset_offset = 0
        self._capture_stream_index = 0
        self._capture_noise_floor = 0.0
        self._refractory_remaining = 0
        self._adapt_during_refractory = False
        self._warm_up_remaining = self.warm_up_samples
        self._filter_state = [0.0] * 4

    def process(self, incoming: np.ndarray) -> list[DetectedTap]:
        """Feeds one buffer shaped (channels, frames). Returns completed, gate-accepted taps."""
        incoming = np.atleast_2d(np.asarray(incoming, dtype=np.float32))
        frame_count = incoming.shape[1]
        if incoming.shape[0] == 0 or frame_count == 0:
            return []

        channels = self._normalized(incoming)
        mono = channels.mean(axis=0) if self.channel_count > 1 else channels[0]
        onset_signal = cascade_lowpass(mono, self._alpha, self._filter_state)
        try:
            return self._process(channels, onset_signal, frame_count)
        finally:
            self.total_samples += frame_count

    def _process(self, channels: np.ndarray, onset_signal: np.ndarray, frame_count: int) -> list[DetectedTap]:
        level = rms(onset_signal)
        magnitude = np.abs(onset_signal)
        peak = float(magnitude.max())

        # Learn the room before detecting; a laptop mic can sit well above the bootstrap floor.
        if self._warm_up_remaining > 0:
            self._adapt_noise_floor(level, warm_up=True)
            self._warm_up_remaining = max(0, self._warm_up_remaining - frame_count)
            self._append_pre_roll(channels)
            return []

        if self._refractory_remaining > 0:
            if self._adapt_during_refractory:
                self._adapt_noise_floor(level, warm_up=False)
            self._refractory_remaining = max(0, self._refractory_remaining - frame_count)
            if self._refractory_remaining == 0:
                self._adapt_during_refractory = False
            self._append_pre_roll(channels)
            return []

        if self._capture is not None:
            self._capture = np.concatenate([self._capture, channels], axis=1)
            event = self._complete_capture_if_ready()
            return [event] if event else []

        rms_threshold = max(self.noise_floor_rms * 1.18, 0.0008)
        peak_threshold = max(self.noise_floor_rms * 4.0, 0.007)
        crest = peak / max(level, 0.000_001)
        strong_threshold = max(peak_threshold, peak * 0.55)
        strong_fraction = float(np.count_nonzero(magnitude >= strong_threshold)) / max(frame_count, 1)
        is_impulse = level > rms_threshold and peak > peak_threshold and crest > 2.0 and strong_fraction < 0.20

        if is_impulse:
            crossing = int(np.argmax(magnitude >= peak_threshold))
            self._capture = self._pre_roll.copy()
            self._capture_onset_offset = self._pre_roll.shape[1] + crossing
            self._capture_stream_index = self.total_samples + crossing
            self._capture_noise_floor = self.noise_floor_rms
            self._capture = np.concatenate([self._capture, channels], axis=1)
            event = self._complete_capture_if_ready()
            return [event] if event else []

        self._adapt_noise_floor(level, warm_up=False)
        self._append_pre_roll(channels)
        return []

    def _normalized(self, incoming: np.ndarray) -> np.ndarray:
        if incoming.shape[0] == self.channel_count:
            return incoming
        rows = [incoming[min(index, incoming.shape[0] - 1)] for index in range(self.channel_count)]
        return np.stack(rows)

    def _adapt_noise_floor(self, measured_rms: float, warm_up: bool) -> None:
        if not math.isfinite(measured_rms):
            return
        measured = max(measured_rms, 0.000_01)
        if warm_up:
            alpha = 0.14
            self.noise_floor_rms = (1 - alpha) * self.noise_floor_rms + alpha * measured
            return
        # Follow sustained room changes, but cap one-step growth so a tap never becomes the baseline.
        capped = min(measured, max(self.noise_floor_rms * 3.5, 0.020))
        alpha = 0.06 if capped > self.noise_floor_rms else 0.025
        self.noise_floor_rms = (1 - alpha) * self.noise_floor_rms + alpha * capped

    def _append_pre_roll(self, channels: np.ndarray) -> None:
        combined = np.concatenate([self._pre_roll, channels], axis=1)
        self._pre_roll = combined[:, -self.pre_roll_samples:]

    def _complete_capture_if_ready(self) -> DetectedTap | None:
        if self._capture is None or self._capture.shape[1] < self.analysis_window_samples:
            return None
        event = DetectedTap(
            channels=self._capture[:, : self.analysis_window_samples].copy(),
            onset_offset=min(self._capture_onset_offset, self.analysis_window_samples - 1),
            stream_sample_index=self._capture_stream_index,
            noise_floor_rms=self._capture_noise_floor,
        )
        metrics = impact_metrics(event, self.sample_rate)
        accepted = impact_gate_accepts(event, self.sample_rate, metrics)
        self._capture = None
        self._pre_roll = np.zeros((self.channel_count, 0), dtype=np.float32)
        self._refractory_remaining = int(self.sample_rate * 0.14)
        # A rejected sustained event is likely speech; let the floor follow it so
        # conversation cannot re-arm the detector every 140 ms.
        self._adapt_during_refractory = not accepted
        if not accepted:
            self.gate_rejections.append((event, metrics))
            del self.gate_rejections[:-20]
            return None
        return event
