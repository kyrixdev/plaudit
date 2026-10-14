"""When a recognised tap should run its zone's action.

Holo runs an action on every accepted tap. On Windows, live mode keeps
listening while you use the PC, so stray clicks and bumps get through. The
default here is a double tap: the same zone twice within a short window.
"""

from __future__ import annotations

from dataclasses import dataclass

from .classifier import ClassificationDecision
from .models import DeskZone

DOUBLE_WINDOW_S = 0.70
# The detector ignores ~230 ms after each tap (analysis window + refractory), so
# a second tap can't arrive sooner; anything faster is merged into the first.
DOUBLE_MIN_GAP_S = 0.20
# A hard tap's rebound arrives ~230 ms later at 15-30% of its loudness.
REBOUND_MAX_RATIO = 0.25
SINGLE_MIN_CONFIDENCE = 0.80
DOUBLE_MIN_MEAN_CONFIDENCE = 0.50


@dataclass
class TriggerResult:
    fire: bool
    zone: DeskZone | None
    note: str  # fired | first tap | low confidence | rebound | rejected


@dataclass
class _Pending:
    zone: DeskZone
    time: float
    confidence: float
    peak: float


class TapTrigger:
    def __init__(self, mode: str = "double", min_confidence: float | None = None,
                 window_s: float = DOUBLE_WINDOW_S):
        if mode not in ("double", "single"):
            raise ValueError("mode must be 'double' or 'single'")
        self.mode = mode
        self.window_s = window_s
        default = DOUBLE_MIN_MEAN_CONFIDENCE if mode == "double" else SINGLE_MIN_CONFIDENCE
        self.min_confidence = default if min_confidence is None else min_confidence
        self._pending: _Pending | None = None

    def observe(self, decision: ClassificationDecision, peak: float, time_s: float) -> TriggerResult:
        """time_s is the tap's stream time in seconds; peak its peak amplitude."""
        if not decision.was_accepted:
            return TriggerResult(False, None, "rejected")
        zone, confidence = decision.zone, decision.confidence

        if self.mode == "single":
            if confidence < self.min_confidence:
                return TriggerResult(False, zone, "low confidence")
            return TriggerResult(True, zone, "fired")

        pending = self._pending
        if pending is not None and time_s - pending.time > self.window_s:
            pending = self._pending = None
        if pending is not None and pending.zone == zone and time_s - pending.time >= DOUBLE_MIN_GAP_S:
            if peak < pending.peak * REBOUND_MAX_RATIO:
                return TriggerResult(False, zone, "rebound")
            self._pending = None
            if (pending.confidence + confidence) / 2 < self.min_confidence:
                return TriggerResult(False, zone, "low confidence")
            return TriggerResult(True, zone, "fired")
        self._pending = _Pending(zone, time_s, confidence, peak)
        return TriggerResult(False, zone, "first tap")
