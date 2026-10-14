"""Shared data types. Ported from HoloCore's Zone.swift and SignalModels.swift."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum

import numpy as np


class DeskZone(IntEnum):
    LEFT_REAR = 0
    LEFT_FRONT = 1
    RIGHT_REAR = 2
    RIGHT_FRONT = 3

    @property
    def vertical_index(self) -> int:
        return self.value % 2

    @property
    def row(self) -> int:
        return self.vertical_index

    @property
    def column(self) -> int:
        return 0 if self.value < 2 else 1

    @property
    def is_left(self) -> bool:
        return self.value < 2

    @property
    def short_name(self) -> str:
        return ["LR", "LF", "RR", "RF"][self.value]

    @property
    def display_name(self) -> str:
        return ["Left Rear", "Left Front", "Right Rear", "Right Front"][self.value]

    @property
    def instruction(self) -> str:
        side = "LEFT" if self.is_left else "RIGHT"
        edge = "the screen hinge (rear)" if self.vertical_index == 0 else "the front edge, by the touchpad"
        return f"Tap the desk to the {side} of the laptop, level with {edge}"


class RejectionReason(str, Enum):
    WEAK_SIGNAL = "weakSignal"
    LOW_SIGNAL_TO_NOISE = "lowSignalToNoise"
    CLIPPED_SIGNAL = "clippedSignal"
    OUT_OF_DISTRIBUTION = "outOfDistribution"
    AMBIGUOUS_ZONE = "ambiguousZone"
    RESEMBLES_NEGATIVE_EXAMPLE = "resemblesNegativeExample"
    SCHEMA_MISMATCH = "schemaMismatch"

    @property
    def display_name(self) -> str:
        return {
            "weakSignal": "Signal too weak",
            "lowSignalToNoise": "Background noise too high",
            "clippedSignal": "Signal clipped",
            "outOfDistribution": "Unlike calibrated taps",
            "ambiguousZone": "Zone ambiguous",
            "resemblesNegativeExample": "Recognized non-desk sound",
            "schemaMismatch": "Profile is incompatible",
        }[self.value]


@dataclass
class SignalQuality:
    MINIMUM_RELIABLE_PEAK_AMPLITUDE = 0.003
    MAXIMUM_RELIABLE_CLIPPING_FRACTION = 0.20
    MINIMUM_CLASSIFICATION_SNR_DB = 6.0

    signal_to_noise_db: float
    peak_amplitude: float
    rms_amplitude: float
    clipping_fraction: float
    noise_floor_rms: float
    duration_ms: float

    @property
    def score(self) -> float:
        snr = min(max((self.signal_to_noise_db - 4) / 30, 0), 1)
        strength = min(max((self.peak_amplitude - self.MINIMUM_RELIABLE_PEAK_AMPLITUDE) / 0.15, 0), 1)
        clean = 1 - min(self.clipping_fraction * 4, 1)
        return 0.55 * snr + 0.25 * strength + 0.20 * clean

    @property
    def summary(self) -> str:
        if self.clipping_fraction > self.MAXIMUM_RELIABLE_CLIPPING_FRACTION:
            return "Clipped"
        if self.peak_amplitude < self.MINIMUM_RELIABLE_PEAK_AMPLITUDE:
            return "Weak"
        if self.signal_to_noise_db < self.MINIMUM_CLASSIFICATION_SNR_DB:
            return "Noisy"
        if self.score > 0.72:
            return "Excellent"
        if self.score > 0.48:
            return "Good"
        return "Fair"

    def to_dict(self) -> dict:
        return {
            "signalToNoiseDB": self.signal_to_noise_db,
            "peakAmplitude": self.peak_amplitude,
            "rmsAmplitude": self.rms_amplitude,
            "clippingFraction": self.clipping_fraction,
            "noiseFloorRMS": self.noise_floor_rms,
            "durationMilliseconds": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SignalQuality":
        return cls(
            signal_to_noise_db=data["signalToNoiseDB"],
            peak_amplitude=data["peakAmplitude"],
            rms_amplitude=data["rmsAmplitude"],
            clipping_fraction=data["clippingFraction"],
            noise_floor_rms=data["noiseFloorRMS"],
            duration_ms=data["durationMilliseconds"],
        )


def good_quality() -> SignalQuality:
    """A clean reference quality, used by tests and synthetic fixtures."""
    return SignalQuality(28, 0.12, 0.025, 0, 0.0004, 90)


@dataclass
class TapFeatureVector:
    SCHEMA_VERSION = 1

    names: list[str]
    values: list[float]
    quality: SignalQuality
    strategy: str = "passive"
    captured_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "strategy": self.strategy,
            "names": list(self.names),
            "values": [float(v) for v in self.values],
            "quality": self.quality.to_dict(),
            "capturedAt": self.captured_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TapFeatureVector":
        return cls(
            names=list(data["names"]),
            values=[float(v) for v in data["values"]],
            quality=SignalQuality.from_dict(data["quality"]),
            strategy=data.get("strategy", "passive"),
            captured_at=data.get("capturedAt", ""),
            version=data.get("version", cls.SCHEMA_VERSION),
        )


@dataclass
class LabeledTap:
    zone: DeskZone | None
    feature: TapFeatureVector
    negative_label: str | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "zone": None if self.zone is None else int(self.zone),
            "negativeLabel": self.negative_label,
            "feature": self.feature.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LabeledTap":
        zone = data.get("zone")
        return cls(
            zone=None if zone is None else DeskZone(zone),
            feature=TapFeatureVector.from_dict(data["feature"]),
            negative_label=data.get("negativeLabel"),
            id=data.get("id") or str(uuid.uuid4()),
        )


@dataclass
class DetectedTap:
    """A fixed-length capture window around one onset. channels has shape (C, N)."""

    channels: np.ndarray
    onset_offset: int
    stream_sample_index: int
    noise_floor_rms: float
