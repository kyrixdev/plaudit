"""Desk profiles: a trained classifier plus the capture setup it was trained with."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .actions import ZoneAction
from .classifier import TrainedTapClassifier
from .models import DeskZone
from .paths import DATA_DIR

PROFILE_VERSION = 1
PROFILES_DIR = DATA_DIR / "profiles"


@dataclass
class DeskProfile:
    name: str
    classifier: TrainedTapClassifier
    sample_rate: float
    channel_count: int
    device_name: str
    audio_mode: str
    surface: str = ""
    laptop_position: str = ""
    leave_one_out_accuracy: float | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    actions: dict[DeskZone, ZoneAction] = field(default_factory=lambda: {z: ZoneAction() for z in DeskZone})

    def action(self, zone: DeskZone) -> ZoneAction:
        return self.actions.get(zone, ZoneAction())

    def to_dict(self) -> dict:
        return {
            "version": PROFILE_VERSION,
            "name": self.name,
            "createdAt": self.created_at,
            "audio": {
                "sampleRate": self.sample_rate,
                "channelCount": self.channel_count,
                "deviceName": self.device_name,
                "mode": self.audio_mode,
            },
            "surface": self.surface,
            "laptopPosition": self.laptop_position,
            "calibration": {
                "samplesPerZone": [
                    sum(1 for t in self.classifier.positive_examples if t.zone == z) for z in range(4)
                ],
                "negativeCount": len(self.classifier.negative_examples),
                "leaveOneOutAccuracy": self.leave_one_out_accuracy,
            },
            "actions": {z.short_name: self.action(z).to_dict() for z in DeskZone},
            "classifier": self.classifier.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DeskProfile":
        if data.get("version") != PROFILE_VERSION:
            raise ValueError(f"Unsupported profile version {data.get('version')}")
        audio = data["audio"]
        saved_actions = data.get("actions") or {}
        return cls(
            name=data["name"],
            classifier=TrainedTapClassifier.from_dict(data["classifier"]),
            sample_rate=audio["sampleRate"],
            channel_count=audio["channelCount"],
            device_name=audio.get("deviceName", ""),
            audio_mode=audio.get("mode", "raw"),
            surface=data.get("surface", ""),
            laptop_position=data.get("laptopPosition", ""),
            leave_one_out_accuracy=data.get("calibration", {}).get("leaveOneOutAccuracy"),
            created_at=data.get("createdAt", ""),
            actions={z: ZoneAction.from_dict(saved_actions.get(z.short_name)) for z in DeskZone},
        )

    def save(self, path: Path | None = None) -> Path:
        path = path or profile_path(self.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, path: Path) -> "DeskProfile":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def profile_path(name: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-").lower() or "desk"
    return PROFILES_DIR / f"{slug}.json"


def latest_profile() -> Path | None:
    if not PROFILES_DIR.exists():
        return None
    candidates = sorted(PROFILES_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None
