"""Zone classifier and leave-one-out review. Ported from HoloCore's TapClassifier.swift
and the ClassifierEvaluator in Evaluation.swift.

A ridge-regression model supplies the zone boundary; a nearest-example model
handles novelty and negative-example (talking, typing...) rejection.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .models import DeskZone, LabeledTap, RejectionReason, SignalQuality, TapFeatureVector

MINIMUM_CONFIDENCE = 0.36
MINIMUM_RELATIVE_SEPARATION = 0.035
MINIMUM_LINEAR_SCORE_MARGIN = 0.075
ZONE_COUNT = len(DeskZone)


class ClassifierTrainingError(ValueError):
    pass


@dataclass
class ClassificationDecision:
    zone: DeskZone | None
    confidence: float
    signal_strength: float
    zone_distances: list[float]
    rejection_reason: RejectionReason | None

    @property
    def was_accepted(self) -> bool:
        return self.zone is not None and self.rejection_reason is None


def _normalize(values, center: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - center) / np.maximum(scales, 1e-9)


def _weighted_distance(lhs: np.ndarray, rhs: np.ndarray, weights: np.ndarray) -> float:
    delta = lhs - rhs
    return math.sqrt(float(np.dot(weights, delta * delta)) / max(float(weights.sum()), 1e-9))


def _unweighted_distance(lhs: np.ndarray, rhs: np.ndarray) -> float:
    delta = lhs - rhs
    return math.sqrt(float(np.dot(delta, delta)) / max(min(len(lhs), len(rhs)), 1))


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values), min(max(probability, 0), 1)))


@dataclass
class TrainedTapClassifier:
    strategy: str
    feature_names: list[str]
    center: np.ndarray
    scales: np.ndarray
    feature_weights: np.ndarray
    positive_examples: list[LabeledTap]
    negative_examples: list[LabeledTap]
    outlier_threshold: float
    positive_novelty_threshold: float | None
    linear_coefficients: np.ndarray | None  # shape (zones, features + 1)
    minimum_confidence: float = MINIMUM_CONFIDENCE
    _positive_matrix: np.ndarray = field(init=False, repr=False)
    _negative_matrix: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64)
        self.scales = np.asarray(self.scales, dtype=np.float64)
        self.feature_weights = np.asarray(self.feature_weights, dtype=np.float64)
        if self.linear_coefficients is not None:
            self.linear_coefficients = np.asarray(self.linear_coefficients, dtype=np.float64)
        self._positive_matrix = np.array(
            [_normalize(t.feature.values, self.center, self.scales) for t in self.positive_examples]
        )
        self._negative_matrix = np.array(
            [_normalize(t.feature.values, self.center, self.scales) for t in self.negative_examples]
        )

    # -- training -----------------------------------------------------------------

    @classmethod
    def train(
        cls,
        positive_examples: list[LabeledTap],
        negative_examples: list[LabeledTap] | None = None,
        minimum_confidence: float = MINIMUM_CONFIDENCE,
    ) -> "TrainedTapClassifier":
        negatives = list(negative_examples or [])
        positives = [t for t in positive_examples if t.zone is not None]
        grouped: dict[DeskZone, list[int]] = defaultdict(list)
        for index, tap in enumerate(positives):
            grouped[tap.zone].append(index)
        if len(grouped) < 2:
            raise ClassifierTrainingError("Examples from at least two zones are required.")
        if any(len(indices) < 2 for indices in grouped.values()):
            raise ClassifierTrainingError("At least two examples per represented zone are required.")
        names = positives[0].feature.names
        strategy = positives[0].feature.strategy
        if not names or any(
            t.feature.names != names or len(t.feature.values) != len(names) or t.feature.strategy != strategy
            for t in positives + negatives
        ):
            raise ClassifierTrainingError("All calibration examples must use the same feature schema.")

        raw = np.array([t.feature.values for t in positives], dtype=np.float64)
        center = np.median(raw, axis=0)
        robust = np.median(np.abs(raw - center), axis=0) * 1.4826
        std = raw.std(axis=0, ddof=1) if len(raw) > 1 else np.zeros(raw.shape[1])
        scales = np.maximum(np.maximum(robust, std * 0.35), 1e-6)

        normalized = (raw - center) / scales
        overall = normalized.mean(axis=0)
        between = np.zeros(raw.shape[1])
        within = np.zeros(raw.shape[1])
        for indices in grouped.values():
            zone_values = normalized[indices]
            zone_mean = zone_values.mean(axis=0)
            between += len(indices) * (zone_mean - overall) ** 2
            within += ((zone_values - zone_mean) ** 2).sum(axis=0)
        weights = np.clip(np.sqrt(between / np.maximum(within, 1e-6) + 0.02), 0.12, 4.0)
        weights = weights / max(float(weights.mean()), 1e-9)

        nearest, nearest_novelty = [], []
        for index, tap in enumerate(positives):
            others = [j for j in grouped[tap.zone] if j != index]
            if others:
                nearest.append(min(_weighted_distance(normalized[index], normalized[j], weights) for j in others))
                nearest_novelty.append(min(_unweighted_distance(normalized[index], normalized[j]) for j in others))
        outlier_threshold = max(_quantile(nearest, 0.95) * 2.6, 0.85)
        novelty_threshold = max(_quantile(nearest_novelty, 0.95) * 2.6, 0.95)

        return cls(
            strategy=strategy,
            feature_names=list(names),
            center=center,
            scales=scales,
            feature_weights=weights,
            positive_examples=positives,
            negative_examples=negatives,
            outlier_threshold=outlier_threshold,
            positive_novelty_threshold=novelty_threshold,
            linear_coefficients=_train_linear_model(normalized, positives),
            minimum_confidence=minimum_confidence,
        )

    # -- prediction ---------------------------------------------------------------

    def predict(self, feature: TapFeatureVector) -> ClassificationDecision:
        if (feature.strategy != self.strategy or feature.names != self.feature_names
                or len(feature.values) != len(self.center)):
            return self._rejected(feature, RejectionReason.SCHEMA_MISMATCH)
        quality = feature.quality
        if quality.clipping_fraction > SignalQuality.MAXIMUM_RELIABLE_CLIPPING_FRACTION:
            return self._rejected(feature, RejectionReason.CLIPPED_SIGNAL)
        if quality.peak_amplitude < SignalQuality.MINIMUM_RELIABLE_PEAK_AMPLITUDE:
            return self._rejected(feature, RejectionReason.WEAK_SIGNAL)
        if quality.signal_to_noise_db < SignalQuality.MINIMUM_CLASSIFICATION_SNR_DB:
            return self._rejected(feature, RejectionReason.LOW_SIGNAL_TO_NOISE)

        x = _normalize(feature.values, self.center, self.scales)
        distances = [math.inf] * ZONE_COUNT
        rank_weights = [0.58, 0.28, 0.14]
        for zone in DeskZone:
            rows = [i for i, t in enumerate(self.positive_examples) if t.zone == zone]
            if not rows:
                continue
            candidates = sorted(_weighted_distance(x, self._positive_matrix[i], self.feature_weights) for i in rows)[:3]
            used = rank_weights[: len(candidates)]
            distances[zone] = sum(d * w for d, w in zip(candidates, used)) / sum(used)

        ranked = sorted((d, z) for z, d in enumerate(distances) if math.isfinite(d))
        if len(ranked) < 2:
            return self._rejected(feature, RejectionReason.SCHEMA_MISMATCH, distances=distances)
        best, second = ranked[0][0], ranked[1][0]

        if best > self.outlier_threshold:
            return self._rejected(feature, RejectionReason.OUT_OF_DISTRIBUTION, distances=distances)

        nearest_novelty = min(_unweighted_distance(x, row) for row in self._positive_matrix)
        if self.positive_novelty_threshold is not None and nearest_novelty > self.positive_novelty_threshold:
            return self._rejected(feature, RejectionReason.OUT_OF_DISTRIBUTION, distances=distances)

        if len(self._negative_matrix):
            nearest_negative = min(_unweighted_distance(x, row) for row in self._negative_matrix)
            if nearest_negative <= nearest_novelty * 1.10:
                return self._rejected(feature, RejectionReason.RESEMBLES_NEGATIVE_EXAMPLE, distances=distances)

        linear = self._linear_decision(x, nearest_novelty, distances, feature)
        if linear is not None:
            return linear

        separation = (second - best) / max(second, 1e-9)
        separation_score = min(max(separation / 0.38, 0), 1)
        fit_score = min(max(1 - best / self.outlier_threshold, 0), 1)
        confidence = 0.72 * separation_score + 0.28 * fit_score
        if separation < MINIMUM_RELATIVE_SEPARATION or confidence < self.minimum_confidence:
            return self._rejected(feature, RejectionReason.AMBIGUOUS_ZONE, confidence, distances)
        return ClassificationDecision(DeskZone(ranked[0][1]), confidence, feature.quality.score, distances, None)

    def _linear_decision(self, x: np.ndarray, nearest_novelty: float, distances: list[float],
                         feature: TapFeatureVector) -> ClassificationDecision | None:
        model = self.linear_coefficients
        if model is None or model.shape != (ZONE_COUNT, len(x) + 1):
            return None
        represented = {t.zone for t in self.positive_examples}
        row = np.concatenate([[1.0], x])
        scores = sorted(
            ((float(np.dot(model[z], row)), z) for z in DeskZone if z in represented),
            key=lambda item: item[0], reverse=True,
        )
        scores = [s for s in scores if math.isfinite(s[0])]
        if len(scores) < 2:
            return None
        margin = scores[0][0] - scores[1][0]
        margin_score = min(max(margin / 0.30, 0), 1)
        fit_threshold = self.positive_novelty_threshold or max(self.outlier_threshold, 1e-9)
        fit_score = min(max(1 - nearest_novelty / fit_threshold, 0), 1)
        confidence = 0.76 * margin_score + 0.24 * fit_score
        if margin < MINIMUM_LINEAR_SCORE_MARGIN or confidence < self.minimum_confidence:
            # A weak linear boundary defers to the nearest-example ambiguity gate.
            return None
        return ClassificationDecision(scores[0][1], confidence, feature.quality.score, distances, None)

    def _rejected(self, feature, reason, confidence: float = 0.0, distances=None) -> ClassificationDecision:
        return ClassificationDecision(None, confidence, feature.quality.score, list(distances or []), reason)

    # -- persistence ----------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "featureNames": self.feature_names,
            "center": self.center.tolist(),
            "scales": self.scales.tolist(),
            "featureWeights": self.feature_weights.tolist(),
            "positiveExamples": [t.to_dict() for t in self.positive_examples],
            "negativeExamples": [t.to_dict() for t in self.negative_examples],
            "outlierThreshold": self.outlier_threshold,
            "positiveNoveltyThreshold": self.positive_novelty_threshold,
            "linearZoneModel": None if self.linear_coefficients is None
            else {"coefficients": self.linear_coefficients.tolist()},
            "minimumConfidence": self.minimum_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrainedTapClassifier":
        linear = data.get("linearZoneModel")
        return cls(
            strategy=data.get("strategy", "passive"),
            feature_names=list(data["featureNames"]),
            center=data["center"],
            scales=data["scales"],
            feature_weights=data["featureWeights"],
            positive_examples=[LabeledTap.from_dict(t) for t in data["positiveExamples"]],
            negative_examples=[LabeledTap.from_dict(t) for t in data.get("negativeExamples", [])],
            outlier_threshold=data["outlierThreshold"],
            positive_novelty_threshold=data.get("positiveNoveltyThreshold"),
            linear_coefficients=None if not linear else linear["coefficients"],
            minimum_confidence=data.get("minimumConfidence", MINIMUM_CONFIDENCE),
        )


def _train_linear_model(normalized: np.ndarray, positives: list[LabeledTap]) -> np.ndarray | None:
    """One-vs-rest ridge regression; the intercept is not penalized."""
    if normalized.ndim != 2 or normalized.shape[1] == 0:
        return None
    design = np.hstack([np.ones((len(normalized), 1)), normalized])
    targets = np.zeros((len(normalized), ZONE_COUNT))
    for index, tap in enumerate(positives):
        targets[index, tap.zone] = 1.0
    normal = design.T @ design
    # Scale shrinkage with the number of captures so profile sizes stay comparable.
    regularization = max(3.0, len(positives) * 0.25)
    normal[1:, 1:] += np.eye(normal.shape[0] - 1) * regularization
    try:
        lower = np.linalg.cholesky(normal)
    except np.linalg.LinAlgError:
        return None
    solution = np.linalg.solve(lower.T, np.linalg.solve(lower, design.T @ targets))
    coefficients = solution.T
    return coefficients if np.all(np.isfinite(coefficients)) else None


@dataclass
class CrossValidationResult:
    accuracy: float
    accepted_accuracy: float
    rejection_rate: float
    per_zone: dict[DeskZone, tuple[int, int]]  # zone -> (correct, total)
    confusion: np.ndarray  # rows: expected zone, columns: predicted zone + "rejected"
    predictions: list[tuple[DeskZone, ClassificationDecision]]

    @property
    def weakest_zone(self) -> DeskZone | None:
        """The lowest-agreement zone, or None when every zone agrees perfectly."""
        scored = [(c / t, z) for z, (c, t) in self.per_zone.items() if t and c < t]
        return min(scored)[1] if scored else None


def leave_one_out(samples: list[LabeledTap], minimum_confidence: float = MINIMUM_CONFIDENCE) -> CrossValidationResult:
    """Classifies each tap with a model trained on all the others. A consistency check, not an accuracy test."""
    positives = [s for s in samples if s.zone is not None]
    if len(positives) < 4:
        raise ClassifierTrainingError("At least two examples per represented zone are required.")
    predictions = []
    for index, held_out in enumerate(positives):
        training = positives[:index] + positives[index + 1:]
        classifier = TrainedTapClassifier.train(training, minimum_confidence=minimum_confidence)
        predictions.append((held_out.zone, classifier.predict(held_out.feature)))

    confusion = np.zeros((ZONE_COUNT, ZONE_COUNT + 1), dtype=int)
    for expected, decision in predictions:
        confusion[expected, ZONE_COUNT if decision.zone is None else decision.zone] += 1
    correct = sum(1 for e, d in predictions if d.zone == e)
    accepted = [(e, d) for e, d in predictions if d.zone is not None]
    per_zone = {z: (int(confusion[z, z]), int(confusion[z].sum())) for z in DeskZone}
    return CrossValidationResult(
        accuracy=correct / len(predictions),
        accepted_accuracy=(sum(1 for e, d in accepted if d.zone == e) / len(accepted)) if accepted else 0.0,
        rejection_rate=(len(predictions) - len(accepted)) / len(predictions),
        per_zone=per_zone,
        confusion=confusion,
        predictions=predictions,
    )
