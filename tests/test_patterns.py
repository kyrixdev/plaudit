import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plaudit.actions import ZoneAction
from plaudit.patterns import Burst, PatternSettings, TapCounter, burst_overlaps_input, seconds_since_input


def feed(counter, times, peak=0.1, snr=15.0):
    return [counter.add(t, peak, snr) for t in times]


def test_counts_a_burst_after_the_quiet_gap():
    counter = TapCounter(max_gap_s=0.6)
    feed(counter, [10.0, 10.35, 10.7])
    assert counter.poll(11.0) is None          # only 0.3 s of quiet so far
    burst = counter.poll(11.31)
    assert burst.count == 3 and burst.start_s == 10.0 and burst.end_s == 10.7
    assert counter.poll(12.0) is None          # nothing pending afterwards


def test_separate_bursts():
    counter = TapCounter(max_gap_s=0.6)
    feed(counter, [1.0, 1.3])
    assert counter.poll(2.0).count == 2
    feed(counter, [3.0, 3.4, 3.8, 4.2])
    assert counter.poll(4.9).count == 4


def test_quiet_sounds_and_rebounds_are_not_counted():
    counter = TapCounter()
    assert counter.add(1.0, 0.005, 20.0) == "too quiet"   # below minimum peak
    assert counter.add(1.0, 0.2, 3.0) == "too quiet"      # below minimum SNR
    assert counter.add(2.0, 0.40, 25.0) == "counted"
    assert counter.add(2.24, 0.08, 12.0) == "rebound"     # 20% as loud, 240 ms later
    assert counter.add(2.27, 0.30, 20.0) == "counted"     # a real quick second tap
    assert counter.add(2.9, 0.05, 12.0) == "counted"      # quiet but well after: a real tap
    assert counter.poll(4.0).count == 3


def test_input_guard_window():
    burst = Burst(3, start_s=10.0, end_s=10.8, peaks=[0.1] * 3)
    now = 11.4  # burst finished; it began 1.4 s ago
    assert burst_overlaps_input(burst, now, idle_s=0.2)       # typed during the burst
    assert burst_overlaps_input(burst, now, idle_s=1.7)       # typed just before it began
    assert not burst_overlaps_input(burst, now, idle_s=2.5)   # hands off the keyboard


@pytest.mark.skipif(sys.platform != "win32", reason="Windows API")
def test_seconds_since_input_is_available():
    assert seconds_since_input() >= 0


def test_settings_round_trip(tmp_path):
    settings = PatternSettings()
    settings.actions[2] = ZoneAction("media", "play_pause")
    settings.actions[4] = ZoneAction("hotkey", "win+d")
    settings.ignore_while_typing = False
    path = settings.save(tmp_path / "patterns.json")
    loaded = PatternSettings.load(path)
    assert loaded.action(2) == ZoneAction("media", "play_pause")
    assert loaded.action(3) == ZoneAction()
    assert loaded.action(4).value == "win+d" and not loaded.ignore_while_typing
    assert PatternSettings.load(tmp_path / "missing.json").action(2) == ZoneAction()
