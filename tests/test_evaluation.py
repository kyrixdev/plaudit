import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plaudit.evaluation import EvaluationRecord, EvaluationReport
from plaudit.models import DeskZone, RejectionReason

LR, LF, RR, RF = DeskZone


def record(expected, predicted, response=95.0, reason=None):
    return EvaluationRecord(expected, predicted, 0.9 if predicted is not None else 0.0, response, reason)


def balanced_report():
    records = []
    for zone in DeskZone:
        records += [record(zone, zone) for _ in range(13)]
    records += [record(LR, LF), record(LR, None, reason=RejectionReason.AMBIGUOUS_ZONE)]
    records += [record(LF, LR), record(LF, RF)]
    records += [record(RR, RF), record(RR, None, reason=RejectionReason.LOW_SIGNAL_TO_NOISE)]
    records += [record(RF, RF), record(RF, RF)]
    return EvaluationReport("Desk", "p.json", records)


def test_rejections_count_as_incorrect_and_matrix_adds_up():
    report = balanced_report()
    assert len(report.records) == 60 and report.is_balanced and report.is_full_acceptance_session
    assert report.overall_accuracy == 54 / 60
    assert report.zone_accuracy(LR) == (13, 15)
    assert report.zone_accuracy(RF) == (15, 15)
    matrix = report.confusion
    assert matrix.sum() == 60
    assert matrix[LR].tolist() == [13, 1, 0, 0, 1]
    assert matrix[RR].tolist() == [0, 0, 13, 1, 1]
    assert report.rejection_counts == {"Zone ambiguous": 1, "Background noise too high": 1}


def test_side_and_depth_accuracy():
    report = balanced_report()
    # Wrong side: LF->RF only. Rejections are wrong on both axes.
    assert report.axis_accuracy("side") == (60 - 1 - 2) / 60
    # Wrong depth: LR->LF, LF->LR, RR->RF (LF->RF keeps the front row).
    assert report.axis_accuracy("depth") == (60 - 3 - 2) / 60


def test_targets():
    report = balanced_report()
    assert report.meets_accuracy_target and report.meets_latency_target
    slow = EvaluationReport("Desk", "", [record(LR, LR, 250.0)] * 3, taps_per_zone=1)
    assert not slow.meets_latency_target
    assert not slow.is_full_acceptance_session
    poor = EvaluationReport("Desk", "", [record(LR, LF)] * 4 + [record(LR, LR)])
    assert not poor.meets_accuracy_target


def test_report_files(tmp_path):
    report = balanced_report()
    json_path, csv_path = report.save(tmp_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["summary"]["overallAccuracy"] == report.overall_accuracy
    assert data["summary"]["confusionMatrix"][0] == [13, 1, 0, 0, 1]
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "timestamp,expected,predicted,confidence,latency_ms,rejection_reason,correct"
    assert len(lines) == 61
    assert any(",LR,REJECTED,0.0000,95.00,ambiguousZone,0" in line for line in lines)
