"""Ports of HoloCore's DetectorTests, ClassifierTests and PipelineIntegrationTests."""

import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plaudit.audio import WavWriter, read_wav
from plaudit.classifier import (MINIMUM_CONFIDENCE, MINIMUM_RELATIVE_SEPARATION, TrainedTapClassifier,
                             leave_one_out)
from plaudit.detector import StreamingTapDetector, impact_gate_accepts, impact_metrics
from plaudit.features import PASSIVE_FEATURE_NAMES, TapFeatureExtractor
from plaudit.models import DeskZone, DetectedTap, LabeledTap, RejectionReason, SignalQuality, TapFeatureVector, good_quality
from plaudit.profile import DeskProfile

SR = 48_000.0


def feed(detector, signal, block=512):
    signal = np.atleast_2d(np.asarray(signal, dtype=np.float32))
    events = []
    for start in range(0, signal.shape[1], block):
        events += detector.process(signal[:, start:start + block])
    return events


class DeterministicNoise:
    def __init__(self, state):
        self.state = state

    def unit(self):
        self.state = (self.state * 6_364_136_223_846_793_005 + 1) % (1 << 64)
        return (self.state >> 11) / ((1 << 64) - 1 >> 11)


def background_chunk(noise):
    chunk = np.zeros(512, dtype=np.float32)
    for start in range(0, 512, 8):
        chunk[start:start + 8] = (noise.unit() * 2 - 1) * 0.014
    return chunk


def candidate_signal(onset, kind, duration=0.090):
    count = max(int(SR * duration), onset + 1)
    index = np.arange(count)
    background = 0.0007 * np.sin(2 * np.pi * 83 * index / SR)
    t = np.maximum(index - onset, 0) / SR
    if kind == "tap":
        body = 0.24 * np.exp(-t * 2_100) * np.cos(2 * np.pi * 1_900 * t) + 0.075 * np.exp(-t * 24) * np.sin(2 * np.pi * 620 * t)
    else:
        voiced = 0.075 * np.minimum(t / 0.010, 1) * (np.sin(2 * np.pi * 145 * t) + 0.55 * np.sin(2 * np.pi * 290 * t)
                                                      + 0.30 * np.sin(2 * np.pi * 435 * t))
        body = 0.24 * np.exp(-t * 850) * np.sin(2 * np.pi * 2_700 * t) + voiced
    return (background + np.where(index >= onset, body, 0)).astype(np.float32)


# -- detector ---------------------------------------------------------------------------

def test_detector_ignores_steady_background_and_finds_impulse():
    detector = StreamingTapDetector(SR, 1)
    events = []
    for _ in range(80):
        events += detector.process(np.full((1, 512), 0.0004, dtype=np.float32))
    assert events == []
    impulse = np.full(512, 0.0004, dtype=np.float32)
    impulse[120], impulse[121] = 0.2, -0.13
    events += detector.process(impulse[None])
    for _ in range(10):
        events += detector.process(np.full((1, 512), 0.0002, dtype=np.float32))
    assert len(events) == 1
    assert events[0].channels.shape[1] == detector.analysis_window_samples
    assert events[0].onset_offset > 0


def test_detector_honors_refractory_period():
    detector = StreamingTapDetector(SR, 1, analysis_duration=0.025, warm_up_duration=0)
    chunk = np.zeros((1, 512), dtype=np.float32)
    chunk[0, 100] = 0.3
    events = []
    for _ in range(4):
        events += detector.process(chunk)
    assert len(events) <= 1


def test_detector_learns_elevated_background_before_emitting_tap():
    detector = StreamingTapDetector(SR, 1)
    noise = DeterministicNoise(0x484F4C4F)
    background_events = []
    for _ in range(120):
        background_events += detector.process(background_chunk(noise)[None])
    assert background_events == []
    assert detector.noise_floor_rms > 0.002
    tap = background_chunk(noise)
    tap[120], tap[124] = 0.48, -0.31
    events = detector.process(tap[None])
    for _ in range(10):
        events += detector.process(background_chunk(noise)[None])
    assert len(events) == 1


def test_impact_gate_rejects_plosive_speech_but_keeps_resonant_tap():
    onset = 576
    speech = DetectedTap(candidate_signal(onset, "speech")[None], onset, 0, 0.001)
    tap = DetectedTap(candidate_signal(onset, "tap")[None], onset, 0, 0.001)
    metrics = impact_metrics(speech, SR)
    assert metrics is not None and metrics.effective_duration_s > 0.04
    assert not impact_gate_accepts(speech, SR)
    assert impact_gate_accepts(tap, SR)


def test_detector_does_not_emit_plosive_speech_candidate():
    detector = StreamingTapDetector(SR, 1, warm_up_duration=0)
    assert feed(detector, candidate_signal(900, "speech", duration=0.18)) == []


def test_detector_handles_stereo_input():
    detector = StreamingTapDetector(SR, 2, warm_up_duration=0)
    left = candidate_signal(1_100, "tap", duration=0.14)
    events = feed(detector, np.stack([left, left * 0.6]))
    assert len(events) == 1
    assert events[0].channels.shape == (2, detector.analysis_window_samples)


# -- classifier -------------------------------------------------------------------------

def feature(zone, jitter=0.0):
    return TapFeatureVector(
        names=["row", "column", "diagonal", "texture"],
        values=[zone.row * 2.2 + jitter, zone.column * 2.0 - jitter,
                (zone.row + zone.column) * 0.8 + jitter * 0.5, zone.value * 0.35 - jitter * 0.2],
        quality=good_quality(),
    )


def one_dimension_feature(value):
    return TapFeatureVector(names=["signature"], values=[value], quality=good_quality())


def training_samples():
    return [LabeledTap(zone, feature(zone, (sample - 3) * 0.012)) for zone in DeskZone for sample in range(8)]


def test_classifier_identifies_all_four_synthetic_zones():
    classifier = TrainedTapClassifier.train(training_samples())
    assert classifier.linear_coefficients.shape == (4, len(classifier.feature_names) + 1)
    for zone in DeskZone:
        decision = classifier.predict(feature(zone, 0.015))
        assert decision.zone == zone
        assert decision.rejection_reason is None
        assert decision.confidence > 0.4


def test_classifier_rejects_weak_and_out_of_distribution():
    classifier = TrainedTapClassifier.train(training_samples())
    weak = feature(DeskZone.LEFT_FRONT)
    weak.quality.peak_amplitude = 0.001
    assert classifier.predict(weak).rejection_reason == RejectionReason.WEAK_SIGNAL
    alien = feature(DeskZone.LEFT_FRONT)
    alien.values = [100, -100, 80, -70]
    assert classifier.predict(alien).rejection_reason == RejectionReason.OUT_OF_DISTRIBUTION


def test_classifier_rejects_clipped_and_noisy_first():
    classifier = TrainedTapClassifier.train(training_samples())
    clipped = feature(DeskZone.RIGHT_REAR)
    clipped.quality.clipping_fraction = SignalQuality.MAXIMUM_RELIABLE_CLIPPING_FRACTION + 0.01
    clipped.quality.peak_amplitude = 0.001
    assert classifier.predict(clipped).rejection_reason == RejectionReason.CLIPPED_SIGNAL
    noisy = feature(DeskZone.RIGHT_REAR)
    noisy.quality.signal_to_noise_db = SignalQuality.MINIMUM_CLASSIFICATION_SNR_DB - 0.1
    assert classifier.predict(noisy).rejection_reason == RejectionReason.LOW_SIGNAL_TO_NOISE


def test_schema_mismatch_is_rejected():
    classifier = TrainedTapClassifier.train(training_samples())
    mismatched = feature(DeskZone.LEFT_FRONT)
    mismatched.names[0] = "other"
    assert classifier.predict(mismatched).rejection_reason == RejectionReason.SCHEMA_MISMATCH


def test_classifier_rejects_calibrated_negative_example():
    typing = feature(DeskZone.LEFT_REAR, 0.042)
    classifier = TrainedTapClassifier.train(training_samples(), [LabeledTap(None, typing, "Typing")])
    decision = classifier.predict(typing)
    assert decision.zone is None
    assert decision.rejection_reason == RejectionReason.RESEMBLES_NEGATIVE_EXAMPLE


def test_classifier_rejects_indistinguishable_zones_as_ambiguous():
    collided = []
    for sample in training_samples():
        if sample.zone == DeskZone.LEFT_FRONT:
            sample = LabeledTap(sample.zone, feature(DeskZone.LEFT_REAR, -sample.feature.values[1]))
        collided.append(sample)
    classifier = TrainedTapClassifier.train(collided)
    decision = classifier.predict(feature(DeskZone.LEFT_REAR, 0))
    assert decision.zone is None
    assert decision.rejection_reason == RejectionReason.AMBIGUOUS_ZONE


def test_classifier_accepts_close_but_usable_decision():
    rear = [-100.0, -100.0, -0.1, -0.1, -0.1, 100.0, 100.0, 100.0]
    front = [-100.0, -100.0, -0.1055, -0.1055, -0.1055, 100.0, 100.0, 100.0]
    samples = [LabeledTap(DeskZone.LEFT_REAR, one_dimension_feature(v)) for v in rear] + \
              [LabeledTap(DeskZone.LEFT_FRONT, one_dimension_feature(v)) for v in front]
    decision = TrainedTapClassifier.train(samples).predict(one_dimension_feature(0))
    ranked = sorted(d for d in decision.zone_distances if math.isfinite(d))
    separation = (ranked[1] - ranked[0]) / ranked[1]
    assert decision.zone == DeskZone.LEFT_REAR
    assert decision.rejection_reason is None
    assert MINIMUM_RELATIVE_SEPARATION <= separation < 0.055
    assert decision.confidence >= MINIMUM_CONFIDENCE


def test_leave_one_out_uses_every_sample():
    samples = training_samples()
    result = leave_one_out(samples, minimum_confidence=0.2)
    assert len(result.predictions) == len(samples)
    assert result.accuracy > 0.95
    assert all(total > 0 for _, total in result.per_zone.values())


# -- end to end -------------------------------------------------------------------------

def detected_feature(zone, frequency_scale, stereo=False):
    detector = StreamingTapDetector(SR, 2 if stereo else 1, warm_up_duration=0)
    extractor = TapFeatureExtractor(SR)
    frequency = [280.0, 470, 760, 1_120][zone] * frequency_scale
    onset = 1_100
    index = np.arange(int(SR * 0.14))
    t = np.maximum(index - onset, 0) / SR
    impact = 0.48 * np.exp(-t * 2_500) * np.cos(2 * np.pi * 1_800 * t)
    envelope = np.exp(-t * (46 + zone.vertical_index * 5))
    side = 0.24 * np.sin(2 * np.pi * frequency * (2.1 if zone.is_left else 2.7) * t)
    signal = np.where(index >= onset, impact + 0.13 * envelope * (np.sin(2 * np.pi * frequency * t) + side), 0.0002)
    signal = signal.astype(np.float32)
    if stereo:
        near, far = (1.0, 0.55) if zone.is_left else (0.55, 1.0)
        signal = np.stack([signal * near, signal * far])
    events = feed(detector, signal)
    assert len(events) == 1, f"detector missed {zone.display_name}"
    return extractor.extract(events[0])


@pytest.mark.parametrize("stereo", [False, True])
def test_streaming_detector_through_classifier_for_all_zones(stereo):
    training = [LabeledTap(zone, detected_feature(zone, 1 + (i - 2) * 0.004, stereo))
                for zone in DeskZone for i in range(5)]
    classifier = TrainedTapClassifier.train(training)
    for zone in DeskZone:
        decision = classifier.predict(detected_feature(zone, 1.002, stereo))
        assert decision.zone == zone
        assert decision.rejection_reason is None


def test_feature_vector_shape_and_spatial_cue():
    left = detected_feature(DeskZone.LEFT_REAR, 1.0, stereo=True)
    right = detected_feature(DeskZone.RIGHT_REAR, 1.0, stereo=True)
    assert left.names == PASSIVE_FEATURE_NAMES and len(left.values) == 30
    delta = PASSIVE_FEATURE_NAMES.index("channel_energy_delta")
    assert left.values[delta] > 0 > right.values[delta]


def test_profile_round_trip(tmp_path):
    classifier = TrainedTapClassifier.train(training_samples(), [LabeledTap(None, feature(DeskZone.LEFT_REAR, 0.5), "Talking")])
    profile = DeskProfile("Test Desk", classifier, SR, 2, "Mic", "raw", leave_one_out_accuracy=1.0)
    loaded = DeskProfile.load(profile.save(tmp_path / "p.json"))
    for zone in DeskZone:
        assert loaded.classifier.predict(feature(zone, 0.015)).zone == zone
    assert len(loaded.classifier.negative_examples) == 1


def test_wav_round_trip(tmp_path):
    audio = np.random.default_rng(1).uniform(-0.5, 0.5, size=(2, 3000)).astype(np.float32)
    writer = WavWriter(tmp_path / "a.wav", SR, 2)
    writer.write(audio[:, :1000])
    writer.write(audio[:, 1000:])
    writer.close()
    loaded, rate = read_wav(tmp_path / "a.wav")
    assert rate == SR and np.array_equal(loaded, audio)


def test_real_time_budget():
    """Detection plus features must keep up with 512-frame stereo buffers (10.7 ms each)."""
    detector = StreamingTapDetector(SR, 2, warm_up_duration=0)
    extractor = TapFeatureExtractor(SR)
    tap = candidate_signal(1_100, "tap", duration=0.5)
    stereo = np.stack([tap, tap * 0.7])
    started = time.perf_counter()
    events = feed(detector, stereo)
    features = [extractor.extract(e) for e in events]
    elapsed = time.perf_counter() - started
    assert features
    assert elapsed < 0.5 * 0.5  # under half of real time
