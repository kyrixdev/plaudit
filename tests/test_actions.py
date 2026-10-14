import ctypes
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plaudit.actions import (ActionDispatcher, ActionError, ZoneAction, capture_screen, get_clipboard_text,
                          normalize_url, parse_hotkey, plan, set_clipboard_text)
from plaudit.classifier import TrainedTapClassifier
from plaudit.models import DeskZone, LabeledTap, TapFeatureVector, good_quality
from plaudit.profile import DeskProfile

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows APIs")


# -- planning -----------------------------------------------------------------------------

def test_none_is_feedback_only():
    assert plan(ZoneAction()) is None


def test_media_keys():
    assert plan(ZoneAction("media", "play_pause")).args == (0xB3,)
    assert plan(ZoneAction("media", "volume_up")).args == (0xAF,)
    with pytest.raises(ActionError):
        plan(ZoneAction("media", "louder"))


def test_hotkeys():
    assert parse_hotkey("win+d") == (0x5B, 0x44)
    assert parse_hotkey("Ctrl + Shift + Esc") == (0x11, 0x10, 0x1B)
    assert parse_hotkey("alt+f4") == (0x12, 0x73)
    assert parse_hotkey("win") == (0x5B,)
    for bad in ["", "ctrl+", "ctrl+alt", "d+ctrl", "ctrl+banana"]:
        with pytest.raises(ActionError):
            parse_hotkey(bad)


def test_urls():
    assert normalize_url("example.com/path") == "https://example.com/path"
    assert normalize_url("http://localhost:3000") == "http://localhost:3000"
    for bad in ["", "ftp://example.com", "javascript:alert(1)", "https://"]:
        with pytest.raises(ActionError):
            normalize_url(bad)
    assert plan(ZoneAction("url", "example.com")).args == ("https://example.com",)


def test_open_targets(tmp_path):
    assert plan(ZoneAction("open", "notepad")).args == ("notepad",)
    assert plan(ZoneAction("open", "spotify:")).args == ("spotify:",)
    assert plan(ZoneAction("open", "ms-settings:sound")).args == ("ms-settings:sound",)
    assert plan(ZoneAction("open", str(tmp_path))).args == (str(tmp_path),)
    with pytest.raises(ActionError):
        plan(ZoneAction("open", str(tmp_path / "missing" / "file.txt")))
    with pytest.raises(ActionError):
        plan(ZoneAction("open", "C:\\definitely\\not\\here.exe"))


def test_other_kinds():
    assert plan(ZoneAction("screenshot", "full")).kind == "screenshot_full"
    assert plan(ZoneAction("screenshot", "selection")).args == ("ms-screenclip:",)
    assert plan(ZoneAction("sound", "asterisk")).args == ("SystemAsterisk",)
    assert plan(ZoneAction("command", "Get-Date")).args == ("Get-Date",)
    assert plan(ZoneAction("copy_text", "  hi  ")).args == ("  hi  ",)  # text is kept verbatim
    for bad in [ZoneAction("screenshot", "window"), ZoneAction("sound", "boing"), ZoneAction("command", ""),
                ZoneAction("speak_text", " "), ZoneAction("teleport", "x")]:
        with pytest.raises(ActionError):
            plan(bad)


def test_summaries():
    assert ZoneAction().summary == "Feedback only"
    assert ZoneAction("media", "next").summary == "Next track"
    assert ZoneAction("hotkey", "win+d").summary == "Keyboard shortcut: win+d"
    assert ZoneAction.from_dict({"kind": "bogus"}).kind == "none"


def test_profile_keeps_actions(tmp_path):
    feature = lambda z, j: TapFeatureVector(["a", "b"], [z.row + j, z.column - j], good_quality())
    taps = [LabeledTap(z, feature(z, j * 0.01)) for z in DeskZone for j in range(4)]
    profile = DeskProfile("Desk", TrainedTapClassifier.train(taps), 48_000, 2, "Mic", "raw")
    profile.actions[DeskZone.RIGHT_FRONT] = ZoneAction("media", "play_pause")
    loaded = DeskProfile.load(profile.save(tmp_path / "desk.json"))
    assert loaded.action(DeskZone.RIGHT_FRONT) == ZoneAction("media", "play_pause")
    assert loaded.action(DeskZone.LEFT_REAR) == ZoneAction()


# -- real Windows side effects (no key presses, no sound) ----------------------------------

@windows_only
def test_clipboard_text_round_trip():
    original = get_clipboard_text()
    try:
        set_clipboard_text("Plaudit tap ✓ é")
        assert get_clipboard_text() == "Plaudit tap ✓ é"
    finally:
        if original is not None:
            set_clipboard_text(original)


@windows_only
def test_screen_capture_produces_full_size_bitmap():
    bitmap = capture_screen()
    try:
        assert bitmap.width > 0 and bitmap.height > 0
    finally:
        gdi32 = ctypes.WinDLL("gdi32")
        gdi32.DeleteObject.argtypes = [ctypes.c_void_p]
        gdi32.DeleteObject(bitmap.handle)


@windows_only
def test_powershell_command_success_and_failure(tmp_path):
    marker = tmp_path / "ran.txt"
    dispatcher = ActionDispatcher()
    dispatcher.perform(ZoneAction("command", f"Set-Content -Path '{marker}' -Value tapped"))
    dispatcher.perform(ZoneAction("command", "Write-Error 'boom'; exit 3"))
    dispatcher.wait(30)
    errors = dispatcher.poll_errors()
    assert marker.read_text().strip() == "tapped"
    assert len(errors) == 1 and "exit code 3" in errors[0]
