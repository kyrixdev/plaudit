import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plaudit.onsets import FastOnsetDetector
from plaudit.patterns import onset_counter

SR = 48_000.0
rng = np.random.default_rng(3)


def clap(length_s, amplitude=0.8, ring_tau=0.03):
    """A sharp broadband hit with a short room-like decay, followed by silence."""
    t = np.arange(int(SR * length_s)) / SR
    return amplitude * np.exp(-t / ring_tau) * rng.uniform(-1, 1, len(t))


def stream(*segments, lead_s=1.0):
    audio = np.concatenate([np.zeros(int(SR * lead_s)), *segments])
    audio = audio + rng.normal(0, 0.0005, len(audio))
    return np.stack([audio, audio * 0.8]).astype(np.float32)


def run(audio):
    detector = FastOnsetDetector(SR)
    onsets = []
    for start in range(0, audio.shape[1], 512):
        onsets += detector.process(audio[:, start:start + 512])
    return onsets


def test_fast_claps_stay_separate():
    # 4 claps 0.15 s apart: Holo's detector is deaf for 230 ms after each event.
    onsets = run(stream(*[clap(0.15) for _ in range(4)], np.zeros(int(SR * 0.5))))
    accepted = [o for o in onsets if o.accepted]
    assert len(accepted) == 4
    assert np.allclose(np.diff([o.seconds for o in accepted]), 0.15, atol=0.01)


def test_ringing_and_echo_are_not_counted_again():
    # A clap-like hit (recorded claps sit at sustained fractions <= 0.28): one accepted onset.
    onsets = run(stream(clap(1.0, ring_tau=0.02)))
    assert len(onsets) == 1 and onsets[0].accepted
    # However long the ring, its fading tail never produces a second onset. A very long
    # ring is judged sustained, like a syllable.
    for tau in (0.05, 0.08, 0.15):
        assert len(run(stream(clap(1.0, ring_tau=tau)))) == 1


def test_sustained_sound_is_rejected():
    # A syllable-like tone: sudden start, then steady for 200 ms.
    t = np.arange(int(SR * 0.2)) / SR
    syllable = 0.3 * np.sin(2 * np.pi * 220 * t) * np.minimum(t / 0.005, 1)
    onsets = run(stream(syllable, np.zeros(int(SR * 0.3))))
    assert onsets and not any(o.accepted for o in onsets)


def test_steady_room_noise_produces_nothing():
    noise = np.stack([rng.normal(0, 0.01, int(SR * 3))] * 2).astype(np.float32)
    assert run(noise) == []


def test_counter_groups_fast_claps_into_one_burst():
    onsets = [o for o in run(stream(*[clap(0.2) for _ in range(3)], np.zeros(int(SR * 1.0)))) if o.accepted]
    counter = onset_counter()
    for onset in onsets:
        assert counter.add(onset.seconds, onset.peak, onset.snr_db) == "counted"
    burst = counter.poll(onsets[-1].seconds + 1.0)
    assert burst is not None and burst.count == 3
