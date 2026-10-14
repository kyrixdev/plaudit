"""Passive tap features. Ported from HoloCore's TapFeatureExtractor.swift.

Holo's active-probe and hybrid strategies (an inaudible chirp from the speakers)
are not ported yet; passive sensing is what Holo uses by default.
"""

from __future__ import annotations

import math

import numpy as np

from .dsp import next_power_of_two, power_spectrum, rms
from .models import DetectedTap, SignalQuality, TapFeatureVector

PASSIVE_FEATURE_NAMES = (
    ["log_rms", "crest_factor", "zero_crossing_rate", "attack_position", "temporal_centroid", "early_late_ratio",
     "spectral_centroid", "spectral_bandwidth", "spectral_rolloff_85", "spectral_flatness"]
    + [f"mfcc_{i}" for i in range(1, 9)]
    + [f"band_{i}" for i in range(10)]
    + ["channel_energy_delta", "interchannel_delay"]
)


def _hz_to_mel(hz: float) -> float:
    return 2_595 * math.log10(1 + hz / 700)


def _mel_to_hz(mel: float) -> float:
    return 700 * (10 ** (mel / 2_595) - 1)


class TapFeatureExtractor:
    def __init__(self, sample_rate: float):
        self.sample_rate = float(sample_rate)

    def extract(self, event: DetectedTap) -> TapFeatureVector:
        channels = np.asarray(event.channels, dtype=np.float32)
        mono = channels.astype(np.float64).mean(axis=0)
        spectrum = power_spectrum(mono, size=min(next_power_of_two(len(mono)), 4096))
        values = self._passive_features(mono, channels, event.onset_offset, spectrum)
        values = [v if math.isfinite(v) else 0.0 for v in values]

        level = rms(mono)
        peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
        clipping = float(np.count_nonzero(np.abs(mono) >= 0.995)) / max(len(mono), 1)
        snr = 20 * math.log10(max(level, 1e-12) / max(event.noise_floor_rms, 1e-12))
        quality = SignalQuality(
            signal_to_noise_db=snr,
            peak_amplitude=peak,
            rms_amplitude=level,
            clipping_fraction=clipping,
            noise_floor_rms=event.noise_floor_rms,
            duration_ms=len(mono) / self.sample_rate * 1_000,
        )
        return TapFeatureVector(names=list(PASSIVE_FEATURE_NAMES), values=values, quality=quality)

    # -- feature groups -------------------------------------------------------

    def _passive_features(self, signal: np.ndarray, channels: np.ndarray, onset: int, spectrum: np.ndarray) -> list[float]:
        n = len(signal)
        if n == 0:
            return [0.0] * len(PASSIVE_FEATURE_NAMES)

        level = rms(signal)
        magnitude = np.abs(signal)
        peak = float(magnitude.max())
        crest = peak / max(level, 1e-12)
        signs = signal >= 0
        zero_crossing_rate = float(np.count_nonzero(signs[1:] != signs[:-1])) / max(n - 1, 1)
        peak_index = int(np.argmax(magnitude))
        attack_position = max(peak_index - onset, 0) / max(n - onset, 1)

        energies = signal * signal
        total_energy = float(energies.sum()) + 1e-15
        temporal_centroid = float(np.dot(np.arange(n), energies)) / total_energy / max(n - 1, 1)
        split = min(max(onset + int(self.sample_rate * 0.025), 1), n - 1)
        early = float(energies[:split].sum())
        late = float(energies[split:].sum())
        early_late_ratio = math.log10((early + 1e-12) / (late + 1e-12))

        temporal = [math.log10(level + 1e-12), crest, zero_crossing_rate, attack_position,
                    temporal_centroid, early_late_ratio]
        return (temporal + self._spectral(spectrum) + self._mel_cepstral(spectrum, 18, 8)
                + self._broad_bands(spectrum, 10) + self._spatial(channels))

    def _bin_frequency(self, spectrum_count: int) -> float:
        return self.sample_rate / max((spectrum_count - 1) * 2, 1)

    def _frequency_bin(self, frequency: float, spectrum_count: int) -> int:
        fft_size = max((spectrum_count - 1) * 2, 1)
        return min(max(int(frequency / self.sample_rate * fft_size), 0), spectrum_count - 1)

    def _spectral(self, spectrum: np.ndarray) -> list[float]:
        frequencies = np.arange(len(spectrum)) * self._bin_frequency(len(spectrum))
        mask = (frequencies >= 60) & (frequencies <= min(18_000, self.sample_rate * 0.48))
        freqs = frequencies[mask]
        power = spectrum[mask]
        total = float(power.sum()) + 1e-15
        centroid = float(np.dot(freqs, power)) / total
        bandwidth = math.sqrt(float(np.dot((freqs - centroid) ** 2, power)) / total)

        rolloff = 0.0
        cumulative = np.cumsum(power)
        reached = np.nonzero(cumulative >= total * 0.85)[0]
        if len(reached):
            rolloff = float(freqs[reached[0]])
        count = max(len(power), 1)
        arithmetic = total / count
        geometric = math.exp(float(np.log(power + 1e-15).sum()) / count)
        flatness = geometric / max(arithmetic, 1e-15)
        sr = self.sample_rate
        return [centroid / sr, bandwidth / sr, rolloff / sr, flatness]

    def _mel_cepstral(self, spectrum: np.ndarray, filter_count: int, coefficient_count: int) -> list[float]:
        count = len(spectrum)
        min_mel = _hz_to_mel(80)
        max_mel = _hz_to_mel(min(16_000, self.sample_rate * 0.46))
        points = [_mel_to_hz(min_mel + (max_mel - min_mel) * i / (filter_count + 1)) for i in range(filter_count + 2)]
        bins = [self._frequency_bin(p, count) for p in points]

        log_energies = np.zeros(filter_count)
        for f in range(filter_count):
            left = bins[f]
            center = max(bins[f + 1], left + 1)
            right = max(bins[f + 2], center + 1)
            energy = 0.0
            if left < count:
                rising = np.arange(left, min(center, count))
                energy += float(np.dot(spectrum[rising], (rising - left) / max(center - left, 1)))
                if center < count:
                    falling = np.arange(center, min(right, count))
                    energy += float(np.dot(spectrum[falling], (right - falling) / max(right - center, 1)))
            log_energies[f] = math.log(energy + 1e-15)

        k = np.arange(filter_count) + 0.5
        return [float(np.dot(log_energies, np.cos(math.pi * c * k / filter_count))) / filter_count
                for c in range(1, coefficient_count + 1)]

    def _broad_bands(self, spectrum: np.ndarray, count: int) -> list[float]:
        minimum = 80.0
        maximum = min(self.sample_rate * 0.46, 18_000)
        total = float(spectrum.sum()) + 1e-15
        result = []
        for band in range(count):
            low = minimum * (maximum / minimum) ** (band / count)
            high = minimum * (maximum / minimum) ** ((band + 1) / count)
            start = self._frequency_bin(low, len(spectrum))
            end = max(start, self._frequency_bin(high, len(spectrum)))
            energy = float(spectrum[start:min(end, len(spectrum) - 1) + 1].sum()) / total
            result.append(math.log10(energy + 1e-12))
        return result

    def _spatial(self, channels: np.ndarray) -> list[float]:
        """Left/right energy difference and arrival delay between the first two mics."""
        if channels.shape[0] < 2 or channels.shape[1] == 0:
            return [0.0, 0.0]
        first = channels[0].astype(np.float64)
        second = channels[1].astype(np.float64)
        energy_delta = math.log10((rms(first) + 1e-12) / (rms(second) + 1e-12))
        max_lag = min(int(self.sample_rate * 0.000_75), 36)
        n = len(first)
        best_lag, best_corr = 0, -math.inf
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                corr = float(np.dot(first[: n - lag], second[lag:]))
            else:
                corr = float(np.dot(first[-lag:], second[: n + lag]))
            if corr > best_corr:
                best_corr, best_lag = corr, lag
        return [energy_delta, best_lag / max(max_lag, 1)]
