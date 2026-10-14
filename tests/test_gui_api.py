import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plaudit.gui import api as api_module
from plaudit.gui.api import SettingsApi
from plaudit.livelog import COLUMNS
from plaudit.patterns import PatternSettings


def make_api(tmp_path):
    return SettingsApi(settings_path=tmp_path / "patterns.json")


def test_catalog_groups_every_action_kind():
    catalog = make_api(Path(".")).get_catalog()
    keys = [k["key"] for group in catalog["groups"] for k in group]
    assert keys[0] == "none" and len(keys) == len(set(keys)) == 10
    media = next(k for g in catalog["groups"] for k in g if k["key"] == "media")
    assert {"value": "play_pause", "label": "Play/pause"} in media["choices"]
    assert catalog["counts"] == [2, 3, 4, 5]


def test_save_pattern_validates_and_persists(tmp_path):
    api = make_api(tmp_path)
    result = api.save_pattern(3, "hotkey", " win+d ")
    assert result["ok"] and result["pattern"]["summary"] == "Keyboard shortcut: win+d"
    assert PatternSettings.load(api._settings_path).action(3).value == "win+d"

    bad = api.save_pattern(2, "hotkey", "ctrl+")
    assert not bad["ok"] and "complete shortcut" in bad["error"]
    assert PatternSettings.load(api._settings_path).action(2).kind == "none"  # unchanged

    assert api.save_pattern(2, "url", "javascript:alert(1)")["ok"] is False
    assert api.save_pattern(4, "none", "")["pattern"]["summary"] == "No action"


def test_state_lists_patterns_and_options(tmp_path):
    api = make_api(tmp_path)
    api.save_pattern(2, "media", "play_pause")
    state = api.get_state()
    assert [p["count"] for p in state["patterns"]] == [2, 3, 4, 5]
    assert state["patterns"][0]["summary"] == "Play/pause"
    assert set(state["options"]) == {"maxGap", "ignoreWhileTyping", "notifyActions", "startWithWindows", "paused"}
    assert set(state["tray"]) == {"running", "state", "text", "lastAction"}


def test_gap_option_is_clamped(tmp_path):
    api = make_api(tmp_path)
    assert api.set_option("maxGap", 5)["ok"]
    assert PatternSettings.load(api._settings_path).max_gap_s == 1.2
    api.set_option("maxGap", 0.55)
    assert PatternSettings.load(api._settings_path).max_gap_s == 0.55
    assert api.set_option("ignoreWhileTyping", False)["ok"]
    assert not PatternSettings.load(api._settings_path).ignore_while_typing
    assert not api.set_option("bogus", 1)["ok"]


def test_activity_reads_bursts_newest_first(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "LOGS_DIR", tmp_path)
    with (tmp_path / "patterns-20261014-120000.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerow(["2026-10-14T12:00:01.000", "1.0", "SOUND", "0", "1", "", "0.9", "30", "counted", ""])
        writer.writerow(["2026-10-14T12:00:02.000", "1.5", "BURST 2", "0", "700", "", "0.9", "0", "2×", "Play/pause"])
        writer.writerow(["2026-10-14T12:00:09.000", "9.0", "BURST 1", "0", "700", "", "0.9", "0",
                         "1× - ignored (single sounds do nothing)", ""])
        writer.writerow(["2026-10-14T12:00:20.000", "20.0", "BURST 3", "0", "700", "", "0.9", "0", "3×",
                         "failed: 'x' doesn't exist."])
    entries = make_api(tmp_path).get_activity()
    assert [e["count"] for e in entries] == [3, 1, 2]
    assert [e["status"] for e in entries] == ["failed", "ignored", "ran"]


def test_only_methods_are_exposed_to_javascript(tmp_path):
    """pywebview walks every public attribute; a reachable Window object froze the window."""
    api = make_api(tmp_path)
    api.attach(object())
    public = [name for name in dir(api) if not name.startswith("_")]
    assert public and all(callable(getattr(api, name)) for name in public)
