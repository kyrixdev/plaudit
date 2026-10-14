"""Held-out accuracy test reporting. Ported from HoloCore's Evaluation.swift.

Every armed tap counts, and a rejection counts as incorrect.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .classifier import ClassificationDecision
from .models import DeskZone, RejectionReason, TapFeatureVector

TAPS_PER_ZONE = 15
MINIMUM_ACCURACY = 0.80
MAXIMUM_MEDIAN_RESPONSE_MS = 200.0
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EvaluationRecord:
    expected: DeskZone
    predicted: DeskZone | None
    confidence: float
    response_ms: float
    rejection_reason: RejectionReason | None = None
    feature: TapFeatureVector | None = None
    captured_at: str = field(default_factory=_now)

    @classmethod
    def from_decision(cls, expected: DeskZone, decision: ClassificationDecision, response_ms: float,
                      feature: TapFeatureVector | None = None) -> "EvaluationRecord":
        return cls(expected, decision.zone, decision.confidence, response_ms, decision.rejection_reason, feature)

    @property
    def is_correct(self) -> bool:
        return self.expected == self.predicted

    def to_dict(self) -> dict:
        return {
            "expectedZone": self.expected.short_name,
            "predictedZone": None if self.predicted is None else self.predicted.short_name,
            "confidence": self.confidence,
            "responseLatencyMilliseconds": self.response_ms,
            "rejectionReason": None if self.rejection_reason is None else self.rejection_reason.value,
            "capturedAt": self.captured_at,
            "feature": None if self.feature is None else self.feature.to_dict(),
        }


@dataclass
class EvaluationReport:
    profile_name: str
    profile_path: str
    records: list[EvaluationRecord]
    taps_per_zone: int = TAPS_PER_ZONE
    undone: int = 0
    started_at: str = field(default_factory=_now)
    completed_at: str = field(default_factory=_now)

    # -- metrics ------------------------------------------------------------------------

    @property
    def overall_accuracy(self) -> float:
        return sum(r.is_correct for r in self.records) / len(self.records) if self.records else 0.0

    def zone_accuracy(self, zone: DeskZone) -> tuple[int, int]:
        matching = [r for r in self.records if r.expected == zone]
        return sum(r.is_correct for r in matching), len(matching)

    @property
    def confusion(self) -> np.ndarray:
        """Rows: expected zone. Columns: predicted LR, LF, RR, RF, then rejected."""
        matrix = np.zeros((len(DeskZone), len(DeskZone) + 1), dtype=int)
        for r in self.records:
            matrix[r.expected, len(DeskZone) if r.predicted is None else r.predicted] += 1
        return matrix

    def axis_accuracy(self, axis: str) -> float:
        """Share of all taps that landed on the right side ('side') or right depth ('depth').

        Rejected taps count as incorrect, as in the overall score.
        """
        key = (lambda z: z.is_left) if axis == "side" else (lambda z: z.vertical_index)
        if not self.records:
            return 0.0
        return sum(r.predicted is not None and key(r.predicted) == key(r.expected) for r in self.records) / len(self.records)

    @property
    def rejection_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.records:
            if r.predicted is None:
                name = r.rejection_reason.display_name if r.rejection_reason else "Rejected"
                counts[name] = counts.get(name, 0) + 1
        return counts

    @property
    def median_response_ms(self) -> float:
        valid = [r.response_ms for r in self.records if r.response_ms >= 0]
        return statistics.median(valid) if valid else 0.0

    @property
    def median_confidence_accepted(self) -> float:
        accepted = [r.confidence for r in self.records if r.predicted is not None]
        return statistics.median(accepted) if accepted else 0.0

    @property
    def is_balanced(self) -> bool:
        return all(self.zone_accuracy(z)[1] == self.taps_per_zone for z in DeskZone)

    @property
    def is_full_acceptance_session(self) -> bool:
        return self.is_balanced and self.taps_per_zone == TAPS_PER_ZONE

    @property
    def meets_accuracy_target(self) -> bool:
        return bool(self.records) and self.overall_accuracy >= MINIMUM_ACCURACY

    @property
    def meets_latency_target(self) -> bool:
        return bool(self.records) and self.median_response_ms < MAXIMUM_MEDIAN_RESPONSE_MS

    # -- output -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "profileName": self.profile_name,
            "profilePath": self.profile_path,
            "startedAt": self.started_at,
            "completedAt": self.completed_at,
            "tapsPerZone": self.taps_per_zone,
            "undone": self.undone,
            "summary": {
                "overallAccuracy": self.overall_accuracy,
                "perZone": {z.short_name: dict(zip(("correct", "total"), self.zone_accuracy(z))) for z in DeskZone},
                "sideAccuracy": self.axis_accuracy("side"),
                "depthAccuracy": self.axis_accuracy("depth"),
                "confusionMatrix": self.confusion.tolist(),
                "rejections": self.rejection_counts,
                "medianResponseMilliseconds": self.median_response_ms,
                "meetsAccuracyTarget": self.meets_accuracy_target,
                "meetsLatencyTarget": self.meets_latency_target,
                "balancedAcceptanceSession": self.is_full_acceptance_session,
            },
            "records": [r.to_dict() for r in self.records],
        }

    def csv(self) -> str:
        lines = ["timestamp,expected,predicted,confidence,latency_ms,rejection_reason,correct"]
        for r in self.records:
            lines.append(",".join([
                r.captured_at,
                r.expected.short_name,
                r.predicted.short_name if r.predicted is not None else "REJECTED",
                f"{r.confidence:.4f}",
                f"{r.response_ms:.2f}",
                r.rejection_reason.value if r.rejection_reason else "",
                "1" if r.is_correct else "0",
            ]))
        return "\n".join(lines) + "\n"

    def save(self, directory: Path = REPORTS_DIR) -> tuple[Path, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        json_path = directory / f"accuracy-{stamp}.json"
        csv_path = directory / f"accuracy-{stamp}.csv"
        json_path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        csv_path.write_text(self.csv(), encoding="utf-8")
        return json_path, csv_path
