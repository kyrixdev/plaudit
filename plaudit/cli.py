"""Command-line entry point: plaudit devices | listen | calibrate | live | replay."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from . import __version__
from .actions import KINDS, MEDIA_KEYS, ActionDispatcher, ActionError, ZoneAction, plan
from .audio import MODES, AudioSetupError, read_wav, resolve_device, wasapi_input_devices, default_input_device
from .classifier import ClassifierTrainingError, TrainedTapClassifier, leave_one_out
from .console import (C, Listener, OnsetListener, clean_device_name, dbfs, desk_diagram, drain_keys, fit, init_terminal,
                      level_bar, paint, println, read_key, status)
from .detector import StreamingTapDetector
from .evaluation import (MAXIMUM_MEDIAN_RESPONSE_MS, MINIMUM_ACCURACY, TAPS_PER_ZONE, EvaluationRecord,
                         EvaluationReport)
from .features import TapFeatureExtractor
from .livelog import LiveLog
from .models import DeskZone, LabeledTap, SignalQuality
from .engine import PatternEngine
from .patterns import COUNTS, PatternSettings, onset_counter
from .paths import DATA_DIR
from .profile import DeskProfile, latest_profile, profile_path
from .trigger import DOUBLE_WINDOW_S, TapTrigger

REPORTS = DATA_DIR / "reports"
GUIDED_MIN_SNR_DB = 7.0


def guided_quality_issue(quality: SignalQuality) -> str | None:
    """Ported from GuidedCaptureQuality.swift: why a calibration tap should be retried."""
    if quality.clipping_fraction > SignalQuality.MAXIMUM_RELIABLE_CLIPPING_FRACTION:
        return "That tap clipped. Use a lighter, more natural tap."
    if quality.peak_amplitude < SignalQuality.MINIMUM_RELIABLE_PEAK_AMPLITUDE:
        return "That tap was too soft. Try a slightly firmer tap."
    if quality.signal_to_noise_db < GUIDED_MIN_SNR_DB:
        return "Background noise masked that tap. Wait for quiet, then try again."
    return None


def open_listener(args, record_path=None, onsets: bool = False) -> Listener:
    device = resolve_device(args.device)
    listener = (OnsetListener if onsets else Listener)(device, args.mode, record_path)
    println(f"Mic     {clean_device_name(device.name)}")
    println(f"Format  {listener.sample_rate:.0f} Hz, {listener.channels} channel(s), {listener.mode} mode")
    if device.looks_external:
        println(paint("Warning: this looks like an external mic. Plaudit is designed for the built-in one.", C.YELLOW))
    if listener.warning:
        println(paint(listener.warning, C.YELLOW))
    return listener


def wait_for_warm_up(listener: Listener) -> None:
    status(paint("Learning the room's background noise - stay quiet for a moment...", C.DIM))
    while listener.warming_up:
        listener.poll()
    println(f"Noise   floor {dbfs(listener.detector.noise_floor_rms):.0f} dBFS")
    report = listener.channel_report()
    if report:
        println(f"Stereo  {report}")


# -- devices --------------------------------------------------------------------------

def cmd_devices(args) -> int:
    try:
        default = default_input_device().index
    except AudioSetupError:
        default = -1
    println("WASAPI input devices (use the number or part of the name with --device):\n")
    for device in wasapi_input_devices():
        marker = paint(" (default)", C.GREEN) if device.index == default else ""
        println(f"  {device.index:3d}  {clean_device_name(device.name)}  "
                f"[{device.channels} ch, {device.default_sample_rate:.0f} Hz]{marker}")
    return 0


# -- listen (step 1) -------------------------------------------------------------------

def cmd_listen(args) -> int:
    record = None
    if args.record:
        record = Path(args.record) if args.record != "auto" else \
            DATA_DIR / "recordings" / f"listen-{datetime.now():%Y%m%d-%H%M%S}.wav"
    listener = open_listener(args, record)
    taps = []
    ignored = 0
    started = time.monotonic()
    try:
        wait_for_warm_up(listener)
        balance = listener.channel_balance
        println()
        println("Tap the desk around the laptop. Each detected tap is listed below.")
        println(paint("L/R compares the two mics, corrected for their sensitivity difference: > 0 nearer mic 1, "
                      "< 0 nearer mic 2. Press Q or Ctrl+C to stop.", C.DIM))
        println()
        while True:
            new_taps, rejections = listener.poll()
            for tap in new_taps:
                taps.append(tap)
                q = tap.feature.quality
                values = dict(zip(tap.feature.names, tap.feature.values))
                colour = {"Excellent": C.GREEN, "Good": C.GREEN, "Fair": C.YELLOW}.get(q.summary, C.RED)
                println(f"  tap {len(taps):3d}  {tap.seconds:7.2f}s  peak {q.peak_amplitude:6.3f}  "
                        f"SNR {q.signal_to_noise_db:5.1f} dB  {paint(f'{q.summary:9s}', colour)}  "
                        f"L/R {values['channel_energy_delta'] - balance:+.3f}  delay {values['interchannel_delay']:+.2f}  "
                        f"{tap.processing_ms:4.1f} ms")
            for rejection in rejections:
                ignored += 1
                m = rejection.metrics
                detail = "" if m is None else \
                    f" (contrast {m.onset_contrast:.1f}, sustained {m.effective_duration_s * 1000:.0f} ms)"
                println(paint(f"  ignored  {rejection.seconds:7.2f}s  sustained or gradual sound{detail}", C.DIM))
            status(f"  level {level_bar(listener.level)} {dbfs(listener.level):6.1f} dBFS   "
                   f"floor {dbfs(listener.detector.noise_floor_rms):6.1f} dBFS   taps {len(taps)}")
            key = read_key()
            if key and key.lower() == "q":
                break
            if args.seconds and time.monotonic() - started >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        listener.close()

    println()
    println(paint("Summary", C.BOLD))
    println(f"  Detected taps: {len(taps)}    ignored sustained sounds: {ignored}    "
            f"audio overflows: {listener.audio.overflows}")
    if taps:
        snrs = [t.feature.quality.signal_to_noise_db for t in taps]
        summaries = [t.feature.quality.summary for t in taps]
        usable = sum(1 for t in taps if guided_quality_issue(t.feature.quality) is None)
        println(f"  Median SNR {statistics.median(snrs):.1f} dB    usable for calibration {usable}/{len(taps)}    "
                + ", ".join(f"{s} {summaries.count(s)}" for s in dict.fromkeys(summaries)))
        println(f"  Median processing time {statistics.median(t.processing_ms for t in taps):.1f} ms")
        if usable >= max(3, len(taps) * 0.7):
            println(paint("  Taps come through cleanly. Next: plaudit calibrate", C.GREEN))
        else:
            println(paint("  Many taps were weak or noisy. Try firmer fingertip taps, a quieter room, "
                          "or compare --mode shared / --mode exclusive.", C.YELLOW))
    else:
        println(paint("  No taps detected. Check the mic isn't muted (Settings > System > Sound) and tap "
                      "firmly on a hard desk.", C.YELLOW))
    if listener.recorder:
        println(f"  Recording saved: {listener.recorder.path}")
    return 0


# -- calibrate (step 2) -----------------------------------------------------------------

class CalibrationQuit(Exception):
    pass


def collect_zone(listener: Listener, zone: DeskZone, target: int, done: set[DeskZone],
                 settle_seconds: float) -> list[LabeledTap]:
    """Collects `target` clean taps for one zone. Keys: U undo, R restart zone, P pause, Q quit."""
    println()
    println(desk_diagram(zone, done))
    println()
    println(paint(f"{zone.display_name}: ", C.BOLD, C.CYAN) + zone.instruction + ".")
    println(paint("Spread your taps around that area. Keys: U undo, R restart zone, P pause, Q quit.", C.DIM))

    samples: list[LabeledTap] = []
    accept_after = time.monotonic() + settle_seconds
    paused = False
    last_message = ""
    while len(samples) < target:
        taps, _ = listener.poll()
        now = time.monotonic()
        key = read_key()
        if key:
            key = key.lower()
            if key == "q":
                raise CalibrationQuit
            if key == "p":
                paused = not paused
                if not paused:
                    listener.flush()
                    accept_after = time.monotonic() + 1.0
                continue
            if key == "u" and samples:
                samples.pop()
                println(paint(f"  undone - {len(samples)}/{target}", C.YELLOW))
                accept_after = now + 0.35
            if key == "r" and samples:
                samples.clear()
                println(paint(f"  zone restarted - 0/{target}", C.YELLOW))
                accept_after = now + 0.35
        if paused:
            status(paint("  Paused - press P to resume", C.YELLOW))
            continue
        if now < accept_after:
            status(paint("  Get ready...", C.DIM))
            continue
        for tap in taps:
            if time.monotonic() < accept_after:
                continue
            issue = guided_quality_issue(tap.feature.quality)
            if issue:
                println(paint(f"  retry: {issue}", C.YELLOW))
                continue
            samples.append(LabeledTap(zone=zone, feature=tap.feature))
            q = tap.feature.quality
            println(f"  {paint('✓', C.GREEN)} tap {len(samples):2d}/{target}  "
                    f"{q.summary:9s} SNR {q.signal_to_noise_db:4.1f} dB")
            accept_after = time.monotonic() + 0.40
            if len(samples) >= target:
                break
        message = f"  Listening - tap {len(samples) + 1} of {target}   level {level_bar(listener.level, 16)}"
        if message != last_message:
            status(message)
            last_message = message
    return samples


def collect_negatives(listener: Listener, label: str, prompt: str, seconds: float) -> list[LabeledTap]:
    println()
    println(paint(f"{label} examples: ", C.BOLD) + prompt)
    println(paint(f"Recording for {seconds:.0f} s. Press Enter to stop early.", C.DIM))
    listener.flush()
    drain_keys()
    samples: list[LabeledTap] = []
    ignored = 0
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        taps, rejections = listener.poll()
        ignored += len(rejections)
        for tap in taps:
            samples.append(LabeledTap(zone=None, negative_label=label, feature=tap.feature))
        status(f"  {end - time.monotonic():4.1f}s left   tap-like sounds captured {len(samples)}   "
               f"already ignored by the detector {ignored}")
        key = read_key()
        if key in ("\r", "\n"):
            break
    println(f"  {label}: {len(samples)} tap-like sounds will be taught as 'not a tap'; "
            f"{ignored} were already ignored.")
    return samples


def print_review(result) -> None:
    println()
    println(paint("Consistency review", C.BOLD) + paint(
        "  (each tap classified while left out of training - a consistency check, not the accuracy test)", C.DIM))
    println(f"  Agreement {result.accuracy * 100:5.1f}%    rejected {result.rejection_rate * 100:4.1f}%    "
            f"accuracy when accepted {result.accepted_accuracy * 100:5.1f}%")
    println()
    print_confusion(result.confusion, result.per_zone, "agreement")


def print_confusion(confusion, per_zone: dict, score_label: str) -> None:
    """Rows are the zone tapped, columns the zone recognised, then rejections."""
    header = "  tapped \\ got   " + "".join(f"{z.short_name:>5}" for z in DeskZone) + f"  rej.   {score_label}"
    println(paint(header, C.DIM))
    for zone in DeskZone:
        correct, total = per_zone[zone]
        row = confusion[zone]
        cells = "".join(paint(f"{v:5d}", C.GREEN if i == zone and v else (C.RED if v else C.DIM))
                        for i, v in enumerate(row[:4]))
        pct = correct / total * 100 if total else 0
        colour = C.GREEN if pct >= 80 else C.YELLOW if pct >= 60 else C.RED
        println(f"  {zone.display_name:15s} {cells}{row[4]:5d}   {paint(f'{pct:5.1f}%', colour)}")


def cmd_calibrate(args) -> int:
    println(paint(f"Plaudit - calibration ({args.taps} taps per zone)", C.BOLD))
    println("Keep the laptop where it will stay, use a quiet room, and tap the desk with a fingertip.")
    println(paint("Tap at the strength you'll use day to day, not extra hard - and vary it a little between taps.",
                  C.BOLD))
    println()
    listener = open_listener(args)
    positives: list[LabeledTap] = []
    negatives: list[LabeledTap] = []
    try:
        wait_for_warm_up(listener)
        done: set[DeskZone] = set()
        for index, zone in enumerate(DeskZone):
            positives += collect_zone(listener, zone, args.taps, done, settle_seconds=1.0 if index == 0 else 2.0)
            done.add(zone)
            println(paint(f"  {zone.display_name} saved.", C.GREEN))

        while True:
            result = leave_one_out(positives)
            print_review(result)
            weakest = result.weakest_zone
            println()
            if result.accuracy < 0.80 and weakest is not None:
                println(paint(f"Agreement is below 80%. Redoing {weakest.display_name} usually helps.", C.YELLOW))
            talk_count = sum(1 for n in negatives if n.negative_label == "Talking")
            type_count = sum(1 for n in negatives if n.negative_label == "Typing")
            println("  [S] save profile")
            println(f"  [T] teach it to ignore talking (recommended){f' - {talk_count} so far' if talk_count else ''}")
            println(f"  [K] teach it to ignore typing{f' - {type_count} so far' if type_count else ''}")
            if weakest is not None:
                println(f"  [W] redo the weakest zone ({weakest.display_name})")
            println("  [Q] quit without saving")
            listener.flush()
            drain_keys()
            choice = None
            while choice not in ("s", "t", "k", "w", "q"):
                listener.poll()  # keep the queue drained and the noise floor current
                key = read_key()
                choice = key.lower() if key else None
            if choice == "q":
                raise CalibrationQuit
            if choice == "s":
                break
            if choice == "t":
                negatives += collect_negatives(listener, "Talking",
                                               "talk normally near the laptop (read something aloud).", 20)
            elif choice == "k":
                negatives += collect_negatives(listener, "Typing", "type normally on the keyboard.", 20)
            elif choice == "w" and weakest is not None:
                positives = [p for p in positives if p.zone != weakest]
                positives += collect_zone(listener, weakest, args.taps, set(DeskZone) - {weakest}, 2.0)
    except CalibrationQuit:
        println()
        println("Calibration cancelled; nothing was saved.")
        return 1
    except KeyboardInterrupt:
        println()
        println("Calibration cancelled; nothing was saved.")
        return 1
    finally:
        listener.close()

    classifier = TrainedTapClassifier.train(positives, negatives)
    profile = DeskProfile(
        name=args.name,
        classifier=classifier,
        sample_rate=listener.sample_rate,
        channel_count=listener.channels,
        device_name=clean_device_name(listener.device.name),
        audio_mode=listener.mode,
        surface=args.surface,
        laptop_position=args.position,
        leave_one_out_accuracy=result.accuracy,
    )
    target = Path(args.output) if args.output else profile_path(args.name)
    kept = ""
    if target.exists():
        try:
            profile.actions = DeskProfile.load(target).actions
            kept = " (zone actions kept)"
        except (ValueError, KeyError, OSError):
            pass
    path = profile.save(target)
    println()
    println(paint(f"Profile saved: {path}{kept}", C.GREEN))
    println("Next: assign actions with  plaudit actions   then run  plaudit live")
    return 0


# -- live (quick check) ---------------------------------------------------------------

def load_profile(args) -> DeskProfile:
    return load_profile_with_path(args)[0]


def load_profile_with_path(args) -> tuple[DeskProfile, Path]:
    if args.profile:
        path = Path(args.profile)
        if not path.exists():
            path = profile_path(args.profile)
    else:
        path = latest_profile()
    if path is None or not path.exists():
        raise AudioSetupError("No profile found. Run 'plaudit calibrate' first.")
    return DeskProfile.load(path), path


def print_action_table(profile: DeskProfile) -> None:
    for zone in DeskZone:
        action = profile.action(zone)
        style = (C.DIM,) if action.kind == "none" else ()
        println(f"  {zone.short_name}  {zone.display_name:12s}  {paint(action.summary, *style)}")


def cmd_live(args) -> int:
    profile = load_profile(args)
    dispatcher = None if args.no_actions else ActionDispatcher()
    trigger = TapTrigger(args.trigger, args.min_confidence, args.double_window)
    if args.mode is None:
        args.mode = profile.audio_mode
    println(paint(f"Plaudit live - profile '{profile.name}'", C.BOLD))
    listener = open_listener(args)
    if listener.sample_rate != profile.sample_rate or listener.channels != profile.channel_count:
        println(paint(f"Warning: profile was made at {profile.sample_rate:.0f} Hz / {profile.channel_count} ch; "
                      "recalibrate if results look wrong.", C.YELLOW))
    log = LiveLog()
    fired = {z: 0 for z in DeskZone}
    held_back = 0
    rejected = 0
    try:
        wait_for_warm_up(listener)
        println()
        println(desk_diagram())
        println()
        if dispatcher:
            print_action_table(profile)
            println()
        else:
            println(paint("Actions are off (--no-actions); taps are only displayed.", C.DIM))
        if trigger.mode == "double":
            println(paint(f"Double-tap a zone to run its action (two taps, {trigger.window_s:.1f} s apart at most, "
                          "at an easy pace). Press Q to stop.", C.BOLD))
        else:
            println(paint(f"Single taps run actions when confidence >= {trigger.min_confidence * 100:.0f}%. "
                          "Press Q to stop.", C.BOLD))
        while True:
            taps, _ = listener.poll()
            for tap in taps:
                decision = profile.classifier.predict(tap.feature)
                response = tap.response_ms()
                quality = tap.feature.quality
                result = trigger.observe(decision, quality.peak_amplitude, tap.seconds)
                action_text = ""
                if result.fire:
                    fired[result.zone] += 1
                    if dispatcher:
                        try:
                            action_text = dispatcher.perform(profile.action(result.zone)) or "feedback only"
                        except (ActionError, OSError) as error:
                            action_text = f"failed: {error}"
                    else:
                        action_text = "actions off"
                log.write(stream_seconds=tap.seconds,
                          recognised=decision.zone.short_name if decision.zone is not None else "REJECTED",
                          confidence=decision.confidence, response_ms=response,
                          rejection_reason=decision.rejection_reason.value if decision.rejection_reason else "",
                          peak=quality.peak_amplitude, snr_db=quality.signal_to_noise_db,
                          trigger=result.note, action=action_text)

                if result.note == "rejected":
                    rejected += 1
                    println(paint(f"  ✗ rejected: {decision.rejection_reason.display_name}", C.DIM))
                    continue
                head = f"  {decision.zone.display_name:12s} {decision.confidence * 100:3.0f}%  {response:3.0f} ms  "
                if result.fire:
                    style = C.RED if action_text.startswith("failed") else C.GREEN
                    println(fit(paint("→ ", C.BOLD, C.CYAN) + head + paint(f"▶ {action_text}", style)))
                elif result.note == "first tap":
                    println(fit(paint("· " + head + "tap again to run", C.DIM)))
                else:
                    held_back += 1
                    reason = "low confidence" if result.note == "low confidence" else "looked like a rebound"
                    println(fit(paint("· " + head + f"not run ({reason})", C.YELLOW)))
            if dispatcher:
                for error in dispatcher.poll_errors():
                    println(paint(f"  {error}", C.RED))
            status(f"  level {level_bar(listener.level, 12)}  ran "
                   + " ".join(f"{z.short_name} {fired[z]}" for z in DeskZone)
                   + f"  held {held_back}  rejected {rejected}  Q quits")
            key = read_key()
            if key and key.lower() == "q":
                break
    except KeyboardInterrupt:
        pass
    finally:
        listener.close()
        log.close()
    println()
    println(f"Actions run: {sum(fired.values())}   held back: {held_back}   rejected sounds: {rejected}")
    println(f"Session log: {log.path}")
    return 0


# -- actions (step 4) ----------------------------------------------------------------------

def parse_zone(text: str) -> DeskZone:
    lookup = {z.short_name.lower(): z for z in DeskZone}
    lookup.update({z.display_name.lower().replace(" ", ""): z for z in DeskZone})
    zone = lookup.get(text.strip().lower().replace(" ", "").replace("-", ""))
    if zone is None:
        raise AudioSetupError(f"Unknown zone '{text}'. Use LR, LF, RR or RF.")
    return zone


def test_action(action: ZoneAction) -> None:
    dispatcher = ActionDispatcher()
    try:
        done = dispatcher.perform(action)
        dispatcher.wait(15)
    except (ActionError, OSError) as error:
        println(paint(f"  Test failed: {error}", C.RED))
        return
    errors = dispatcher.poll_errors()
    for error in errors:
        println(paint(f"  Test failed: {error}", C.RED))
    if not errors:
        println(paint(f"  ▶ {done}" if done else "  Feedback only - nothing to run.", C.GREEN))


def prompt(text: str) -> str:
    try:
        return input(text)
    except EOFError:
        return ""


def edit_action(label: str, current: ZoneAction) -> ZoneAction | None:
    """Asks for a new action. Returns None to keep the current one."""
    println()
    println(paint(label, C.BOLD, C.CYAN) + f" - currently: {current.summary}")
    kinds = list(KINDS.values())
    for number, kind in enumerate(kinds, 1):
        println(f"  {number:2d}. {kind.title}")
    choice = prompt("Action number (Enter to keep): ").strip()
    if not choice:
        return None
    if not choice.isdigit() or not 1 <= int(choice) <= len(kinds):
        println(paint("  Not a valid number; unchanged.", C.YELLOW))
        return None
    kind = kinds[int(choice) - 1]
    value = ""
    if kind.choices:
        for number, option in enumerate(kind.choices, 1):
            label = MEDIA_KEYS[option][1] if kind.key == "media" else option
            println(f"      {number}. {label}")
        picked = prompt(f"  {kind.prompt.capitalize()} (number): ").strip()
        if not picked.isdigit() or not 1 <= int(picked) <= len(kind.choices):
            # sounds also accept a .wav path
            if kind.key == "sound" and picked.lower().endswith(".wav"):
                value = picked
            else:
                println(paint("  Not a valid choice; unchanged.", C.YELLOW))
                return None
        else:
            value = kind.choices[int(picked) - 1]
    elif kind.prompt:
        value = prompt(f"  {kind.prompt.capitalize()}: ").strip()
    action = ZoneAction(kind.key, value)
    try:
        plan(action)
    except ActionError as error:
        println(paint(f"  {error} Unchanged.", C.YELLOW))
        return None
    if kind.key == "command":
        println(paint("  This command will run automatically, hidden, with your permissions, on every tap.", C.YELLOW))
    return action


def cmd_actions(args) -> int:
    profile, path = load_profile_with_path(args)
    if args.action_command == "list":
        print_action_table(profile)
        return 0
    if args.action_command == "test":
        zone = parse_zone(args.zone)
        println(f"Testing {zone.display_name}: {profile.action(zone).summary}")
        test_action(profile.action(zone))
        return 0
    if args.action_command in ("set", "clear"):
        zone = parse_zone(args.zone)
        if args.action_command == "clear":
            action = ZoneAction()
        else:
            if args.kind not in KINDS:
                raise AudioSetupError(f"Unknown action type '{args.kind}'. Use one of: {', '.join(KINDS)}.")
            action = ZoneAction(args.kind, " ".join(args.value))
            plan(action)
        profile.actions[zone] = action
        profile.save(path)
        println(f"{zone.display_name}: {action.summary}")
        return 0

    # Interactive editor
    println(paint(f"Zone actions for '{profile.name}'", C.BOLD) +
          paint("  (they run only in 'plaudit live', never while calibrating or testing)", C.DIM))
    while True:
        println()
        print_action_table(profile)
        println()
        answer = prompt("Zone to change (LR/LF/RR/RF), 'test LR' to try one, or Enter to finish: ").strip()
        if not answer:
            return 0
        words = answer.split()
        try:
            if words[0].lower() == "test" and len(words) == 2:
                zone = parse_zone(words[1])
                println(f"Testing {zone.display_name}: {profile.action(zone).summary}")
                test_action(profile.action(zone))
                continue
            zone = parse_zone(answer)
        except AudioSetupError as error:
            println(paint(str(error), C.YELLOW))
            continue
        action = edit_action(zone.display_name, profile.action(zone))
        if action is None:
            continue
        profile.actions[zone] = action
        profile.save(path)
        println(paint(f"  Saved: {zone.display_name} -> {action.summary}", C.GREEN))
        if action.kind != "none" and prompt("  Try it now? [y/N] ").strip().lower() == "y":
            test_action(action)


# -- tap patterns ------------------------------------------------------------------------

def pattern_label(count: int) -> str:
    return f"{count}×"


def print_pattern_table(settings: PatternSettings) -> None:
    for count in COUNTS:
        action = settings.action(count)
        style = (C.DIM,) if action.kind == "none" else ()
        println(f"  {pattern_label(count):7s}  {paint(action.summary if action.kind != 'none' else 'Nothing', *style)}")


def parse_count(text: str) -> int:
    if not text.strip().isdigit() or int(text) not in COUNTS:
        raise AudioSetupError(f"Tap count must be one of: {', '.join(map(str, COUNTS))}.")
    return int(text)


def cmd_patterns(args) -> int:
    settings = PatternSettings.load()
    command = args.pattern_command
    if command == "run":
        return run_patterns(args, settings)
    if command == "test":
        return test_patterns(args, settings)
    if command == "list":
        print_pattern_table(settings)
        return 0
    if command == "try":
        count = parse_count(args.count)
        println(f"Running {pattern_label(count)}: {settings.action(count).summary}")
        test_action(settings.action(count))
        return 0
    if command in ("set", "clear"):
        count = parse_count(args.count)
        if command == "clear":
            action = ZoneAction()
        else:
            if args.kind not in KINDS:
                raise AudioSetupError(f"Unknown action type '{args.kind}'. Use one of: {', '.join(KINDS)}.")
            action = ZoneAction(args.kind, " ".join(args.value))
            plan(action)
        settings.actions[count] = action
        settings.save()
        println(f"{pattern_label(count)}: {action.summary}")
        return 0

    println(paint("Patterns", C.BOLD) + paint("  (clap, snap or knock N times; run with 'plaudit patterns run')",
                                                    C.DIM))
    while True:
        println()
        print_pattern_table(settings)
        println()
        answer = prompt(f"Tap count to change ({'/'.join(map(str, COUNTS))}), 'try 2' to run one, "
                        "or Enter to finish: ").strip()
        if not answer:
            return 0
        words = answer.split()
        try:
            if words[0].lower() == "try" and len(words) == 2:
                count = parse_count(words[1])
                test_action(settings.action(count))
                continue
            count = parse_count(answer)
        except AudioSetupError as error:
            println(paint(str(error), C.YELLOW))
            continue
        action = edit_action(pattern_label(count), settings.action(count))
        if action is None:
            continue
        settings.actions[count] = action
        settings.save()
        println(paint(f"  Saved: {pattern_label(count)} -> {action.summary}", C.GREEN))
        if action.kind != "none" and prompt("  Try it now? [y/N] ").strip().lower() == "y":
            test_action(action)


def run_patterns(args, settings: PatternSettings) -> int:
    println(paint("Plaudit patterns", C.BOLD))
    device = resolve_device(args.device)
    engine = PatternEngine(device, args.mode, run_actions=not args.no_actions)
    ran, ignored = 0, 0
    try:
        engine.start()
        listener = engine.listener
        println(f"Mic     {clean_device_name(device.name)}")
        println(f"Format  {listener.sample_rate:.0f} Hz, {listener.channels} channel(s), {listener.mode} mode")
        if listener.warning:
            println(paint(listener.warning, C.YELLOW))
        wait_for_warm_up(listener)
        println()
        print_pattern_table(engine.settings)
        println()
        guard = " Bursts are ignored while you type or use the mouse." if engine.settings.ignore_while_typing else ""
        println(paint(f"Clap, snap or knock 2-5 times.{guard} Press Q to stop.", C.BOLD))
        while True:
            if engine.reload_settings_if_changed():
                println(paint("  Actions updated from patterns.json:", C.DIM))
                print_pattern_table(engine.settings)
            update = engine.step()
            for result in update.bursts:
                if result.fired:
                    ran += 1
                    println(fit(paint("→ ", C.BOLD, C.CYAN) + paint(f"{result.outcome:4s}", C.BOLD)
                                + paint(f"▶ {result.action}", C.GREEN)))
                elif result.failed:
                    println(fit(paint(f"→ {result.outcome}  {result.action}", C.RED)))
                else:
                    ignored += 1
                    println(fit(paint(f"· {result.outcome}" + (f"  ({result.action})" if result.action else ""), C.DIM)))
            for error in update.errors:
                println(paint(f"  {error}", C.RED))
            if update.counting:
                status(paint(f"  {'●' * update.counting}  counting...", C.CYAN))
            else:
                status(f"  level {level_bar(listener.level, 12)}  ran {ran}  ignored {ignored}  Q quits")
            key = read_key()
            if key and key.lower() == "q":
                break
    except KeyboardInterrupt:
        pass
    finally:
        log_path = engine.log.path if engine.log else None
        engine.close()
    println()
    println(f"Actions run: {ran}   bursts ignored: {ignored}")
    if log_path:
        println(f"Session log: {log_path}")
    return 0


def wait_for_quiet(listener: Listener, quiet_s: float = 1.0, limit_s: float = 5.0) -> None:
    """Waits until nothing has been heard for quiet_s, so a burst still in progress can't
    spill into the next prompt."""
    last_sound = time.monotonic()
    give_up = last_sound + limit_s
    while time.monotonic() - last_sound < quiet_s and time.monotonic() < give_up:
        onsets, _ = listener.poll()
        if onsets:
            last_sound = time.monotonic()


def test_patterns(args, settings: PatternSettings) -> int:
    """Guided check of tap counting: asks for 2, 3 and 4 taps in random order and scores the count."""
    import random
    trials = [n for n in (2, 3, 4) for _ in range(args.rounds)]
    random.shuffle(trials)
    counter = onset_counter(settings.max_gap_s)
    println(paint(f"Count test ({len(trials)} prompts)", C.BOLD))
    println("Clap, snap or knock the requested number of times, then wait. Don't touch the keyboard; "
            "prompts advance on their own. Q quits.")
    println()
    listener = open_listener(args, Path(args.record) if args.record else None, onsets=True)
    results: list[tuple[int, int]] = []
    try:
        wait_for_warm_up(listener)
        for index, expected in enumerate(trials, 1):
            println()
            println(paint(f"[{index}/{len(trials)}]  {expected} times", C.BOLD, C.CYAN))
            counter.reset()
            deadline = time.monotonic() + 8.0
            counted = 0
            while time.monotonic() < deadline:
                onsets, _ = listener.poll()
                for onset in onsets:
                    counter.add(onset.seconds, onset.peak, onset.snr_db)
                if counter.pending:
                    status(paint(f"  {'●' * counter.pending}", C.CYAN))
                    deadline = max(deadline, time.monotonic() + 2.0)
                burst = counter.poll(listener.stream_seconds)
                if burst is not None:
                    counted = burst.count
                    break
                key = read_key()
                if key and key.lower() == "q":
                    raise CalibrationQuit
            results.append((expected, counted))
            verdict = paint("✓", C.GREEN) if counted == expected else paint("✗", C.RED)
            println(f"  {verdict} counted {counted}" + ("" if counted else " (nothing heard)"))
            wait_for_quiet(listener)
    except (CalibrationQuit, KeyboardInterrupt):
        println()
        println("Tap-count test cancelled.")
        return 1
    finally:
        listener.close()

    correct = sum(e == c for e, c in results)
    println()
    println(paint("Count results", C.BOLD))
    colour = C.GREEN if correct / len(results) >= 0.9 else C.YELLOW if correct / len(results) >= 0.75 else C.RED
    println(f"  Correct {paint(f'{correct}/{len(results)} ({correct / len(results) * 100:.0f}%)', C.BOLD, colour)}")
    for expected in (2, 3, 4):
        counted = [c for e, c in results if e == expected]
        histogram = ", ".join(f"{n} x{counted.count(n)}" for n in sorted(set(counted)))
        println(f"  asked {expected}: {sum(c == expected for c in counted)}/{len(counted)} right   counted: {histogram}")
    report = {
        "completedAt": datetime.now().astimezone().isoformat(),
        "correct": correct, "total": len(results),
        "trials": [{"expected": e, "counted": c} for e, c in results],
        "settings": settings.to_dict(),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"patterns-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    println(f"  Report saved: {path}")
    return 0


# -- test (step 3: held-out accuracy) ---------------------------------------------------

def wait_for_ready(listener: Listener, message: str) -> None:
    """Keeps audio flowing until the user presses Enter or Space. Q quits."""
    println(paint(message, C.BOLD))
    drain_keys()
    while True:
        listener.poll()
        status(paint("  Press Enter when you're in position...", C.DIM))
        key = read_key()
        if key in ("\r", "\n", " "):
            return
        if key and key.lower() == "q":
            raise CalibrationQuit


def run_test_zone(listener: Listener, profile: DeskProfile, zone: DeskZone, target: int,
                  done: set[DeskZone], records: list[EvaluationRecord]) -> int:
    """Records `target` attempts for one zone. Every tap counts; U undoes the last one. Returns undo count."""
    println()
    println(desk_diagram(zone, done))
    println()
    println(paint(f"{zone.display_name}: ", C.BOLD, C.CYAN) + zone.instruction + ".")
    wait_for_ready(listener, f"Get ready to tap {zone.display_name} {target} times, spread around the area.")
    listener.flush()
    accept_after = time.monotonic() + 1.0
    undone = 0

    def count() -> int:
        return sum(1 for r in records if r.expected == zone)

    while count() < target:
        taps, _ = listener.poll()
        now = time.monotonic()
        key = read_key()
        if key and key.lower() == "q":
            raise CalibrationQuit
        if key and key.lower() == "u" and count():
            records.pop(max(i for i, r in enumerate(records) if r.expected == zone))
            undone += 1
            println(paint(f"  last attempt undone - {count()}/{target}", C.YELLOW))
            accept_after = now + 0.35
        if now < accept_after:
            status(paint("  Get ready...", C.DIM))
            continue
        for tap in taps:
            if time.monotonic() < accept_after or count() >= target:
                continue
            decision = profile.classifier.predict(tap.feature)
            response = tap.response_ms()
            records.append(EvaluationRecord.from_decision(zone, decision, response, tap.feature))
            if decision.zone == zone:
                verdict = paint(f"✓ {zone.display_name:34s}", C.GREEN)
            elif decision.zone is not None:
                verdict = paint(f"✗ got {decision.zone.display_name:30s}", C.RED)
            else:
                verdict = paint(f"✗ rejected: {decision.rejection_reason.display_name:24s}", C.YELLOW)
            println(f"  {count():2d}/{target}  {verdict} {decision.confidence * 100:3.0f}%  {response:4.0f} ms")
            accept_after = time.monotonic() + 0.40
        status(f"  Tap {zone.display_name} - {count() + 1} of {target}   level {level_bar(listener.level, 16)}"
               f"   {paint('U undo  Q quit', C.DIM)}")
    println(paint(f"  {zone.display_name} done.", C.GREEN))
    return undone


def warn_strength_mismatch(profile: DeskProfile, records: list[EvaluationRecord]) -> None:
    """Flags a test whose taps were much softer or harder than the calibration taps."""
    calibration = [t.feature.quality.peak_amplitude for t in profile.classifier.positive_examples]
    tested = [r.feature.quality.peak_amplitude for r in records if r.feature is not None]
    if not calibration or not tested:
        return
    ratio = statistics.median(tested) / max(statistics.median(calibration), 1e-9)
    if ratio < 0.5 or ratio > 2.0:
        direction = "softer" if ratio < 1 else "harder"
        factor = 1 / ratio if ratio < 1 else ratio
        println(paint(f"  These taps were about {factor:.1f}x {direction} than your calibration taps. "
                      "Tap strength changes the sound, so recalibrate at the strength you normally use.", C.YELLOW))


def print_test_report(report: EvaluationReport) -> None:
    total = len(report.records)
    correct = sum(r.is_correct for r in report.records)
    println()
    println(paint("Accuracy test results", C.BOLD) + paint("  (every tap counts; rejections count as wrong)", C.DIM))
    acc_colour = C.GREEN if report.meets_accuracy_target else C.RED
    lat_colour = C.GREEN if report.meets_latency_target else C.RED
    println(f"  Overall   {paint(f'{report.overall_accuracy * 100:5.1f}%', C.BOLD, acc_colour)} ({correct}/{total})"
            f"   target >= {MINIMUM_ACCURACY * 100:.0f}%")
    println(f"  Response  {paint(f'{report.median_response_ms:5.0f} ms', lat_colour)} median"
            f"   target < {MAXIMUM_MEDIAN_RESPONSE_MS:.0f} ms")
    println(f"  Left vs right correct {report.axis_accuracy('side') * 100:5.1f}%    "
            f"rear vs front correct {report.axis_accuracy('depth') * 100:5.1f}%")
    rejections = report.rejection_counts
    if rejections:
        println("  Rejected: " + ", ".join(f"{name} {n}" for name, n in rejections.items()))
    if report.undone:
        println(paint(f"  {report.undone} attempt(s) were undone during the test.", C.DIM))
    println()
    print_confusion(report.confusion, {z: report.zone_accuracy(z) for z in DeskZone}, "accuracy")
    println()
    if not report.is_full_acceptance_session:
        println(paint(f"  Note: the original Holo's acceptance test is {TAPS_PER_ZONE} taps per zone; "
                      f"this run used {report.taps_per_zone}.", C.DIM))
    elif report.meets_accuracy_target and report.meets_latency_target:
        println(paint("  PASS: meets the original Holo's accuracy and response-time targets.", C.GREEN, C.BOLD))
    else:
        println(paint("  Below the original Holo's targets.", C.YELLOW, C.BOLD))


def cmd_test(args) -> int:
    profile, path = load_profile_with_path(args)
    if args.mode is None:
        args.mode = profile.audio_mode
    println(paint(f"Plaudit - accuracy test ({args.taps} taps per zone, profile '{profile.name}')", C.BOLD))
    println("Tap each zone when asked. Every tap counts, including rejected ones, so tap as you normally would.")
    println()
    listener = open_listener(args, Path(args.record) if args.record else None)
    if listener.sample_rate != profile.sample_rate or listener.channels != profile.channel_count:
        println(paint(f"Warning: profile was made at {profile.sample_rate:.0f} Hz / {profile.channel_count} ch.",
                      C.YELLOW))
    records: list[EvaluationRecord] = []
    undone = 0
    started = datetime.now().astimezone().isoformat()
    try:
        wait_for_warm_up(listener)
        done: set[DeskZone] = set()
        for zone in DeskZone:
            undone += run_test_zone(listener, profile, zone, args.taps, done, records)
            done.add(zone)
    except (CalibrationQuit, KeyboardInterrupt):
        println()
        println("Accuracy test cancelled; no report was saved.")
        return 1
    finally:
        listener.close()

    report = EvaluationReport(
        profile_name=profile.name,
        profile_path=str(path),
        records=records,
        taps_per_zone=args.taps,
        undone=undone,
        started_at=started,
        completed_at=datetime.now().astimezone().isoformat(),
    )
    print_test_report(report)
    warn_strength_mismatch(profile, records)
    json_path, csv_path = report.save()
    println(f"  Report saved: {json_path}")
    println(f"                {csv_path}")
    return 0


# -- replay (offline debugging) --------------------------------------------------------

def cmd_replay(args) -> int:
    audio, sample_rate = read_wav(Path(args.wav))
    profile = DeskProfile.load(Path(args.profile)) if args.profile else None
    detector = StreamingTapDetector(sample_rate, audio.shape[0])
    extractor = TapFeatureExtractor(sample_rate)
    println(f"{args.wav}: {audio.shape[1] / sample_rate:.1f} s, {sample_rate:.0f} Hz, {audio.shape[0]} ch")
    count = 0
    for start in range(0, audio.shape[1], 512):
        for event in detector.process(audio[:, start:start + 512]):
            count += 1
            feature = extractor.extract(event)
            q = feature.quality
            line = f"  tap {count:3d} {event.stream_sample_index / sample_rate:7.2f}s  " \
                   f"SNR {q.signal_to_noise_db:5.1f} dB  {q.summary}"
            if profile:
                decision = profile.classifier.predict(feature)
                line += "  -> " + (decision.zone.display_name if decision.was_accepted
                                   else f"rejected ({decision.rejection_reason.display_name})")
            println(line)
    println(f"{count} taps, {len(detector.gate_rejections)} recent sustained sounds ignored")
    return 0


# -- tray ---------------------------------------------------------------------------------

def cmd_tray(args) -> int:
    from .tray import main as tray_main  # pystray/Pillow load only when needed
    println("Plaudit is running in the system tray (look for the round icon near the clock). Quit it from its menu.")
    return tray_main(args.device, args.mode)


# -- entry point -------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="plaudit", description="Clap, snap or knock to control your PC.")
    parser.add_argument("--version", action="version", version=f"plaudit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def audio_options(p, mode_default="raw"):
        p.add_argument("--device", help="input device number or name (see 'plaudit devices'); default mic if omitted")
        p.add_argument("--mode", choices=MODES, default=mode_default,
                       help="raw (default) skips Windows mic effects; shared is what most apps get")

    sub.add_parser("devices", help="list microphones").set_defaults(func=cmd_devices)

    listen = sub.add_parser("listen", help="step 1: show detected taps and signal quality")
    audio_options(listen)
    listen.add_argument("--seconds", type=float, default=0, help="stop automatically after this many seconds")
    listen.add_argument("--record", nargs="?", const="auto", help="save the session to a WAV file")
    listen.set_defaults(func=cmd_listen)

    calibrate = sub.add_parser("calibrate", help="step 2: guided four-zone calibration")
    audio_options(calibrate)
    calibrate.add_argument("--name", default="My Desk")
    calibrate.add_argument("--taps", type=int, default=10, help="taps per zone (default 10)")
    calibrate.add_argument("--surface", default="", help="note about the desk surface")
    calibrate.add_argument("--position", default="", help="note about where the laptop sits")
    calibrate.add_argument("--output", help="profile path (default profiles/<name>.json)")
    calibrate.set_defaults(func=cmd_calibrate)

    live = sub.add_parser("live", help="recognise taps and run each zone's action")
    audio_options(live, mode_default=None)
    live.add_argument("--profile", help="profile path or name (default: most recent)")
    live.add_argument("--no-actions", action="store_true", help="only display taps; don't run actions")
    live.add_argument("--trigger", choices=("double", "single"), default="double",
                      help="double (default): tap a zone twice to run its action; single: every confident tap")
    live.add_argument("--min-confidence", type=float,
                      help="confidence needed to run an action (default 0.5 average for double, 0.8 for single)")
    live.add_argument("--double-window", type=float, default=DOUBLE_WINDOW_S,
                      help=f"max seconds between the two taps of a double tap (default {DOUBLE_WINDOW_S})")
    live.set_defaults(func=cmd_live)

    test = sub.add_parser("test", help="step 3: guided held-out accuracy test (15 taps per zone)")
    audio_options(test, mode_default=None)
    test.add_argument("--profile", help="profile path or name (default: most recent)")
    test.add_argument("--taps", type=int, default=TAPS_PER_ZONE, help=f"taps per zone (default {TAPS_PER_ZONE})")
    test.add_argument("--record", help="also save the session audio to this WAV path")
    test.set_defaults(func=cmd_test)

    actions = sub.add_parser("actions", help="step 4: choose what each zone does (interactive editor)")
    actions.add_argument("--profile", help="profile path or name (default: most recent)")
    action_sub = actions.add_subparsers(dest="action_command")
    action_sub.add_parser("list", help="show the zone actions")
    set_parser = action_sub.add_parser("set", help="e.g. plaudit actions set LR media play_pause")
    set_parser.add_argument("zone", help="LR, LF, RR or RF")
    set_parser.add_argument("kind", help=", ".join(KINDS))
    set_parser.add_argument("value", nargs="*", help="the action's value (key, path, URL, command, text...)")
    test_parser = action_sub.add_parser("test", help="run a zone's action once")
    test_parser.add_argument("zone")
    clear_parser = action_sub.add_parser("clear", help="set a zone back to feedback only")
    clear_parser.add_argument("zone")
    actions.set_defaults(func=cmd_actions)

    patterns = sub.add_parser("patterns", help="tap the desk N times anywhere to run an action (no calibration)")
    pattern_sub = patterns.add_subparsers(dest="pattern_command")
    pattern_run = pattern_sub.add_parser("run", help="listen for tap patterns and run their actions")
    audio_options(pattern_run)
    pattern_run.add_argument("--no-actions", action="store_true", help="only display the counts")
    pattern_test = pattern_sub.add_parser("test", help="guided check that tap counts are recognised")
    audio_options(pattern_test)
    pattern_test.add_argument("--rounds", type=int, default=5, help="prompts per count (default 5, so 15 total)")
    pattern_test.add_argument("--record", help="also save the session audio to this WAV path")
    pattern_sub.add_parser("list", help="show the pattern actions")
    pattern_set = pattern_sub.add_parser("set", help="e.g. plaudit patterns set 2 media play_pause")
    pattern_set.add_argument("count", help="2, 3, 4 or 5")
    pattern_set.add_argument("kind", help=", ".join(KINDS))
    pattern_set.add_argument("value", nargs="*")
    pattern_try = pattern_sub.add_parser("try", help="run a pattern's action once")
    pattern_try.add_argument("count")
    pattern_clear = pattern_sub.add_parser("clear", help="remove a pattern's action")
    pattern_clear.add_argument("count")
    patterns.set_defaults(func=cmd_patterns)

    tray = sub.add_parser("tray", help="run pattern listening from the system tray (plaudit-tray.bat starts it "
                                       "without a console window)")
    audio_options(tray)
    tray.set_defaults(func=cmd_tray)

    replay = sub.add_parser("replay", help="run a recorded WAV through the detector (and a profile)")
    replay.add_argument("wav")
    replay.add_argument("--profile")
    replay.set_defaults(func=cmd_replay)
    return parser


def main(argv=None) -> int:
    init_terminal()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (AudioSetupError, ClassifierTrainingError, ActionError) as error:
        println(paint(f"Error: {error}", C.RED))
        return 2


if __name__ == "__main__":
    sys.exit(main())
