import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plaudit.classifier import ClassificationDecision
from plaudit.livelog import COLUMNS, LiveLog
from plaudit.models import DeskZone, RejectionReason
from plaudit.trigger import TapTrigger

LR, LF, RR, RF = DeskZone


def tap(zone, confidence=0.9):
    return ClassificationDecision(zone, confidence, 0.8, [], None)


def rejected():
    return ClassificationDecision(None, 0.0, 0.2, [], RejectionReason.OUT_OF_DISTRIBUTION)


def test_double_tap_fires_on_second_tap_of_same_zone():
    trigger = TapTrigger("double")
    assert trigger.observe(tap(RF), 0.2, 10.0).note == "first tap"
    result = trigger.observe(tap(RF), 0.2, 10.45)
    assert result.fire and result.zone == RF
    # A third tap starts a new pair rather than firing again.
    assert trigger.observe(tap(RF), 0.2, 10.9).note == "first tap"


def test_single_stray_taps_never_fire():
    trigger = TapTrigger("double")
    times = [1.0, 3.0, 5.5, 9.0]
    assert not any(trigger.observe(tap(z), 0.2, t).fire for z, t in zip(DeskZone, times))


def test_too_slow_or_different_zone_does_not_fire():
    trigger = TapTrigger("double", window_s=0.7)
    trigger.observe(tap(LR), 0.2, 1.0)
    assert not trigger.observe(tap(LR), 0.2, 1.8).fire  # 0.8 s later: starts over
    assert trigger.observe(tap(LR), 0.2, 2.2).fire       # ...and pairs with that one
    trigger.observe(tap(LR), 0.2, 5.0)
    assert not trigger.observe(tap(RR), 0.2, 5.4).fire   # different zone
    assert trigger.observe(tap(RR), 0.2, 5.8).fire       # RR pair


def test_rebound_and_rejections():
    trigger = TapTrigger("double")
    trigger.observe(tap(RF), 0.40, 1.0)
    assert trigger.observe(tap(RF), 0.05, 1.23).note == "rebound"  # 12% as loud, 230 ms later
    assert trigger.observe(tap(RF), 0.30, 1.5).fire                # a real second tap still works
    trigger.observe(tap(LF), 0.2, 3.0)
    assert trigger.observe(rejected(), 0.1, 3.3).note == "rejected"
    assert trigger.observe(tap(LF), 0.2, 3.5).fire                 # a rejected sound in between is ignored


def test_confidence_rules():
    double = TapTrigger("double")
    double.observe(tap(LF, 0.45), 0.2, 1.0)
    assert double.observe(tap(LF, 0.60), 0.2, 1.4).fire          # average 0.525 >= 0.5
    double.observe(tap(LF, 0.40), 0.2, 3.0)
    assert double.observe(tap(LF, 0.44), 0.2, 3.4).note == "low confidence"
    single = TapTrigger("single")
    assert single.observe(tap(RF, 0.95), 0.2, 1.0).fire
    assert single.observe(tap(RF, 0.70), 0.2, 2.0).note == "low confidence"
    assert TapTrigger("single", min_confidence=0.6).observe(tap(RF, 0.70), 0.2, 1.0).fire


def test_live_log(tmp_path):
    log = LiveLog(tmp_path)
    log.write(stream_seconds=1.5, recognised="RF", confidence=0.93, response_ms=104.2, rejection_reason="",
              peak=0.21, snr_db=18.0, trigger="fired", action="Open website: https://example.com")
    log.close()
    rows = list(csv.reader(log.path.open(encoding="utf-8")))
    assert rows[0] == COLUMNS
    assert rows[1][2:4] == ["RF", "0.930"] and rows[1][-2:] == ["fired", "Open website: https://example.com"]
