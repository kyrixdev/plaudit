"""The pattern-listening loop shared by `plaudit patterns run` and the tray app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .actions import ActionDispatcher, ActionError
from .audio import InputDevice
from .console import OnsetListener
from .livelog import LiveLog
from .patterns import (COUNTS, SETTINGS_PATH, PatternSettings, burst_overlaps_input, onset_counter,
                       seconds_since_input)


@dataclass
class BurstResult:
    count: int
    outcome: str  # e.g. "2×", "1× - ignored (single sounds do nothing)"
    action: str = ""  # what ran, or "failed: ..." / "actions off"

    @property
    def fired(self) -> bool:
        return bool(self.action) and not self.failed and self.action != "actions off"

    @property
    def failed(self) -> bool:
        return self.action.startswith("failed")


@dataclass
class EngineUpdate:
    counting: int  # sounds in the burst in progress
    bursts: list[BurstResult]
    errors: list[str]  # background action failures


class PatternEngine:
    """Owns the microphone, the counter, the action dispatcher and the session log."""

    def __init__(self, device: InputDevice, mode: str = "raw", run_actions: bool = True,
                 settings_path: Path = SETTINGS_PATH):
        self.device = device
        self.mode = mode
        self.settings_path = settings_path
        self.settings = PatternSettings.load(settings_path)
        self._settings_mtime = self._mtime()
        self.dispatcher = ActionDispatcher() if run_actions else None
        self.counter = onset_counter(self.settings.max_gap_s)
        self.listener: OnsetListener | None = None
        self.log: LiveLog | None = None

    # -- lifecycle ------------------------------------------------------------------------------

    @property
    def listening(self) -> bool:
        return self.listener is not None

    def start(self) -> None:
        """Opens the mic. Call warm_up() (or step() until not warming_up) before relying on counts."""
        if self.listener is None:
            self.listener = OnsetListener(self.device, self.mode)
            self.counter.reset()
            if self.log is None:
                self.log = LiveLog(prefix="patterns")

    @property
    def warming_up(self) -> bool:
        return self.listener is not None and self.listener.warming_up

    def stop(self) -> None:
        """Releases the mic (Windows' microphone indicator goes off)."""
        if self.listener is not None:
            self.listener.close()
            self.listener = None
        self.counter.reset()

    def close(self) -> None:
        self.stop()
        if self.log is not None:
            self.log.close()
            self.log = None

    # -- settings -------------------------------------------------------------------------------

    def _mtime(self) -> float:
        try:
            return self.settings_path.stat().st_mtime
        except OSError:
            return 0.0

    def reload_settings_if_changed(self) -> bool:
        mtime = self._mtime()
        if mtime == self._settings_mtime:
            return False
        try:
            self.settings = PatternSettings.load(self.settings_path)
        except (ValueError, OSError):
            return False  # half-written file; try again next time
        self._settings_mtime = mtime
        self.counter.max_gap_s = self.settings.max_gap_s
        return True

    # -- the loop -------------------------------------------------------------------------------

    def step(self, timeout: float = 0.05) -> EngineUpdate:
        if self.listener is None:
            return EngineUpdate(0, [], [])
        onsets, sustained = self.listener.poll(timeout)
        now = self.listener.stream_seconds
        for onset in onsets:
            note = self.counter.add(onset.seconds, onset.peak, onset.snr_db)
            self._log(onset.seconds, "SOUND", (now - onset.seconds) * 1000, "" if note == "counted" else note,
                      onset.peak, onset.snr_db, note, "")
        for onset in sustained:
            self._log(onset.seconds, "SUSTAINED", (now - onset.seconds) * 1000, "sustained sound",
                      onset.peak, onset.snr_db, "ignored", "")
        bursts = []
        burst = self.counter.poll(now)
        if burst is not None:
            result = self._handle_burst(burst, now)
            bursts.append(result)
            self._log(burst.end_s, f"BURST {burst.count}", (now - burst.end_s) * 1000, "", max(burst.peaks), 0.0,
                      result.outcome, result.action)
        errors = self.dispatcher.poll_errors() if self.dispatcher else []
        return EngineUpdate(self.counter.pending, bursts, errors)

    def _handle_burst(self, burst, now: float) -> BurstResult:
        count = burst.count
        if count not in COUNTS:
            return BurstResult(count, "1× - ignored (single sounds do nothing)" if count == 1
                               else f"{count}× - too many, ignored")
        if self.settings.ignore_while_typing and burst_overlaps_input(burst, now, seconds_since_input()):
            return BurstResult(count, f"{count}× - ignored (keyboard/mouse in use)")
        action = self.settings.action(count)
        if action.kind == "none":
            return BurstResult(count, f"{count}× - nothing assigned")
        if self.dispatcher is None:
            return BurstResult(count, f"{count}×", "actions off")
        try:
            return BurstResult(count, f"{count}×", self.dispatcher.perform(action) or "")
        except (ActionError, OSError) as error:
            return BurstResult(count, f"{count}×", f"failed: {error}")

    def _log(self, seconds, recognised, response_ms, reason, peak, snr_db, trigger, action) -> None:
        if self.log is not None:
            self.log.write(stream_seconds=seconds, recognised=recognised, confidence=0.0, response_ms=response_ms,
                           rejection_reason=reason, peak=peak, snr_db=snr_db, trigger=trigger, action=action)
