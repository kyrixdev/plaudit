"""Small DSP helpers shared by the detector, impact gate and feature extractor."""

from __future__ import annotations

import math

import numpy as np


def onset_lowpass_alpha(sample_rate: float) -> float:
    """Coefficient for the sub-6 kHz one-pole stages used by the onset path."""
    cutoff = min(6_000.0, sample_rate * 0.20)
    return 1 - math.exp(-2 * math.pi * cutoff / sample_rate)


def cascade_lowpass(values: np.ndarray, alpha: float, states: list[float]) -> np.ndarray:
    """Runs len(states) cascaded one-pole low-pass stages. Updates states in place."""
    out = values.tolist()
    for stage in range(len(states)):
        state = states[stage]
        for index, value in enumerate(out):
            state += alpha * (value - state)
            out[index] = state
        states[stage] = state
    return np.asarray(out, dtype=np.float64)


def next_power_of_two(value: int) -> int:
    result = 1
    while result < value:
        result <<= 1
    return result


def power_spectrum(signal: np.ndarray, size: int | None = None) -> np.ndarray:
    """Hann-windowed power spectrum, matching HoloCore's Radix2FFT.powerSpectrum.

    The window spans only the copied samples; the rest is zero-padded. Returns
    size/2 + 1 bins scaled by 1/size.
    """
    desired = size if size is not None else len(signal)
    n = next_power_of_two(max(desired, 2))
    copied = min(len(signal), n)
    buffer = np.zeros(n, dtype=np.float64)
    if copied > 1:
        index = np.arange(copied)
        window = 0.5 - 0.5 * np.cos(2 * np.pi * index / (copied - 1))
        buffer[:copied] = np.asarray(signal[:copied], dtype=np.float64) * window
    spectrum = np.fft.rfft(buffer)
    return (spectrum.real ** 2 + spectrum.imag ** 2) / n


def rms(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    v = np.asarray(values, dtype=np.float64)
    return float(math.sqrt(float(np.dot(v, v)) / len(v)))
