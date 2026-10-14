# Plaudit

**Clap, snap or knock to control your Windows PC.** Clap twice to play or pause music, three times for the next
track, four times to take a screenshot. You choose what each count does. Plaudit uses your laptop's built-in
microphone, needs no calibration, and audio never leaves your PC.

Plaudit started as a Windows port of [Holo](https://github.com/JustinGamer191/Holo) (macOS, MIT) by JustinGamer191,
which recognises *where* you tap on a desk. That port is still included as an experimental mode (see
[Desk zones](#experimental-desk-zones)).

## Requirements

- Windows 10 or 11 with a built-in (or any) microphone
- Python 3.11+
- Microsoft Edge WebView2 Runtime for the settings window (preinstalled on Windows 11)

## Setup

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Using Plaudit

1. Double-click **`Plaudit Settings.bat`** to open the settings window. Under **Clap Patterns**, choose what 2, 3,
   4 and 5 claps do. Use **Try it** to test an action.
2. Go to **Listening** and click **Start**, or double-click **`plaudit-tray.bat`**. Plaudit runs in the system tray
   (a round icon near the clock) and listens in the background.
3. Clap. The action runs about 0.7 s after your last clap, once Plaudit knows how many there were.

The settings window is modelled on macOS System Settings and follows the Windows light/dark theme:

- **Clap Patterns:** the four counts and their actions; click one to edit it in a sheet.
- **Listening:** start, pause or resume (pausing turns the microphone off), how long to wait after the last clap,
  ignore claps while you type, start with Windows, notifications, and microphone details.
- **Activity:** every pattern Plaudit heard, whether its action ran, and why it was ignored if not.

Changes apply immediately; the tray app reloads them without a restart.

### Actions

| Action | Value |
| --- | --- |
| Media key | play/pause, next, previous, stop, volume up/down, mute |
| Keyboard shortcut | e.g. `win+d`, `ctrl+alt+t`, `alt+tab` |
| Open app, file or folder | an app name (`notepad`), a path, or a URI (`spotify:`, `ms-settings:sound`) |
| Open website | e.g. `example.com` |
| Screenshot | whole screen to the clipboard, or select an area with the Snipping overlay |
| Play Windows sound, speak text, copy text | a sound name or `.wav`, or the text |
| Run PowerShell command | runs hidden, with your permissions, every time |

### Tray icon

| Colour | Meaning |
| --- | --- |
| Green | Listening |
| Blue (brief flash) | An action just ran |
| Amber | Starting, learning the room's background noise |
| Grey | Paused; microphone off |
| Red | Not listening (e.g. mic unavailable); retries every 5 s |

The right-click menu has Pause/Resume, Open Plaudit..., Open logs folder, notifications, Start with Windows (a
per-user registry Run entry; no admin rights) and Quit. Only one copy runs at a time.

## How it works

- **Raw microphone audio.** Windows normally runs laptop mics through noise suppression and beamforming, which can
  smear short sounds. Plaudit asks WASAPI for *raw* audio (no effects), falling back to normal mode if unavailable.
- **Fast onset detection** (`plaudit/onsets.py`). It follows each sound's fading envelope (its echo only ever decays)
  and counts a new sound whenever the level jumps well above it and ~20 dB above the room. Claps about 0.1 s apart
  stay separate. A sound that stays loud, such as a syllable of speech, is ignored: a clap drops below 40% of its peak
  within the first half of its first 60 ms.
- **Counting and safeguards** (`plaudit/patterns.py`, `plaudit/engine.py`). Sounds less than 0.7 s apart form one
  burst. One sound does nothing, so stray noises don't trigger actions. A burst is ignored if the keyboard or mouse was
  used during it or just before it, so typing and clicking can't trigger actions. Applause or percussion from your
  own speakers still can.
- **Logs.** Every sound and burst is written to `logs\patterns-*.csv`; tray events and errors to `logs\tray.log`.

**Measured on one laptop** (Intel Smart Sound mic array), with the guided count test (`plaudit patterns test`):

| Input | Correct |
| --- | --- |
| Claps | 14/15 (93%) |
| Desk taps | 5/15 (with an older detector) |

There were no false triggers in a 2-minute real-world session of 28 bursts. Your room and microphone will differ, so
run the test yourself.

## Command line

Everything is also available from a terminal via `plaudit.bat` (or `python -m plaudit`):

```bash
.\plaudit.bat patterns run
```

| Command | What it does |
| --- | --- |
| `patterns` | Interactive editor for clap actions (`patterns set 2 media play_pause`, `list`, `try 2`, `clear 2`) |
| `patterns run` | Listen in the terminal (`--no-actions` to only watch the counts) |
| `patterns test` | Guided count test: 15 prompts, scored (`--record file.wav` keeps the audio) |
| `tray` | The tray app with a console, for debugging |
| `devices` | List microphones; select one with `--device` |
| `listen` | Show every detected sound with its level and signal-to-noise ratio (`--record`, `--mode shared/exclusive`) |
| `replay file.wav` | Run a recording back through the desk-tap detector |

## Experimental: desk zones

The original Holo idea: tap the desk to the left or right of the laptop, near the back or front, and the two mics
work out *where* you tapped. The full Holo pipeline is ported: the streaming detector, the impact gate, 30
acoustic/spatial features, and a ridge-regression plus nearest-example classifier with novelty and negative-example
rejection.

```bash
.\plaudit.bat calibrate --name "My Desk"
```

```bash
.\plaudit.bat test
```

```bash
.\plaudit.bat actions
```

```bash
.\plaudit.bat live
```

`calibrate` takes 10 taps per zone, with a consistency review. `test` is Holo's 60-tap acceptance test (it passes at
≥ 80% accuracy with a median response under 200 ms). `actions` assigns actions to zones. `live` runs them on double
taps by default.

**Honest results:** 88–92% within a single session, but 67–75% an hour later with nothing moved. Tap position, force
and style shift the sound more than the four zones differ. That's why claps are the recommended mode.

| Holo (Swift) | Plaudit |
| --- | --- |
| StreamingTapDetector, ImpactEventGate | `plaudit/detector.py` |
| TapFeatureExtractor (passive), FFT | `plaudit/features.py`, `plaudit/dsp.py` |
| TapClassifier, leave-one-out review | `plaudit/classifier.py` |
| Evaluation (60-tap test, JSON/CSV reports) | `plaudit/evaluation.py` |
| LocalActionPlanning, LocalActions | `plaudit/actions.py` (Windows APIs) |
| AVAudioEngine capture | `plaudit/audio.py` (WASAPI raw via sounddevice) |

Holo's active-probe and hybrid sensing modes (a quiet speaker chirp) aren't ported.

## Development

```bash
.venv\Scripts\python -m pytest tests
```

The settings UI is plain HTML/CSS/JS in `plaudit/gui/web/`, served in a native window by
[pywebview](https://pywebview.flowrl.com/). Opened in a normal browser, it runs against built-in sample data, which is
handy for design work.

## License

MIT. See [LICENSE](LICENSE). It includes the original Holo copyright notice, as the MIT licence requires.
