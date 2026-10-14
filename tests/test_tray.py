import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plaudit import engine as engine_module
from plaudit import startup
from plaudit.actions import ZoneAction
from plaudit.engine import PatternEngine
from plaudit.patterns import Burst, PatternSettings


class FakeDevice:
    index = 0
    name = "Fake Mic"


def make_engine(tmp_path, **actions):
    settings = PatternSettings()
    for count, action in actions.items():
        settings.actions[int(count[1:])] = action
    path = settings.save(tmp_path / "patterns.json")
    return PatternEngine(FakeDevice(), settings_path=path)


def burst(count, start=10.0):
    return Burst(count, start, start + 0.3 * (count - 1), [0.5] * count)


def test_burst_outcomes(tmp_path, monkeypatch):
    marker = tmp_path / "ran.txt"
    engine = make_engine(tmp_path, n2=ZoneAction("command", f"Set-Content -Path '{marker}' -Value ok"))
    monkeypatch.setattr(engine_module, "seconds_since_input", lambda: 60.0)
    assert "single sounds do nothing" in engine._handle_burst(burst(1), 12.0).outcome
    assert "too many" in engine._handle_burst(burst(6), 12.0).outcome
    assert "nothing assigned" in engine._handle_burst(burst(3), 12.0).outcome
    fired = engine._handle_burst(burst(2), 12.0)
    assert fired.fired and fired.outcome == "2×"
    engine.dispatcher.wait(30)
    assert marker.read_text().strip() == "ok"


def test_keyboard_guard_and_failures(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, n2=ZoneAction("media", "play_pause"), n3=ZoneAction("open", "notepad"))
    monkeypatch.setattr(engine_module, "seconds_since_input", lambda: 0.1)  # typing right now
    guarded = engine._handle_burst(burst(2), 12.0)
    assert not guarded.fired and "keyboard/mouse" in guarded.outcome
    monkeypatch.setattr(engine_module, "seconds_since_input", lambda: 60.0)
    engine.settings.actions[3] = ZoneAction("open", r"C:\nope\missing.exe")
    failed = engine._handle_burst(burst(3), 12.0)
    assert failed.failed and not failed.fired


def test_actions_reload_when_the_file_changes(tmp_path):
    engine = make_engine(tmp_path)
    assert not engine.reload_settings_if_changed()
    updated = PatternSettings()
    updated.actions[4] = ZoneAction("hotkey", "win+d")
    updated.save(engine.settings_path)
    engine._settings_mtime -= 1  # file systems with coarse timestamps
    assert engine.reload_settings_if_changed()
    assert engine.settings.action(4) == ZoneAction("hotkey", "win+d")


def test_paused_engine_does_nothing(tmp_path):
    engine = make_engine(tmp_path)
    update = engine.step()
    assert not engine.listening and update.bursts == [] and update.counting == 0


class FakeRegistry:
    """Stands in for winreg so tests never touch the real Run key."""
    HKEY_CURRENT_USER = "HKCU"
    KEY_SET_VALUE = 2
    REG_SZ = 1
    REG_DWORD = 4

    def __init__(self):
        self.values = {}

    class _Key:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def OpenKey(self, root, path, reserved=0, access=0):
        return self._Key(path)

    def CreateKey(self, root, path):
        return self._Key(path)

    def QueryValueEx(self, key, name):
        if (key.path, name) not in self.values:
            raise FileNotFoundError
        return self.values[(key.path, name)], self.REG_SZ

    def SetValueEx(self, key, name, reserved, kind, value):
        self.values[(key.path, name)] = value

    def DeleteKey(self, root, path):
        doomed = [k for k in self.values if k[0] == path]
        if not doomed:
            raise FileNotFoundError
        for k in doomed:
            del self.values[k]

    def DeleteValue(self, key, name):
        if (key.path, name) not in self.values:
            raise FileNotFoundError
        del self.values[(key.path, name)]


def test_start_with_windows(monkeypatch):
    fake = FakeRegistry()
    monkeypatch.setattr(startup, "winreg", fake)
    assert not startup.is_enabled()
    startup.enable()
    assert startup.is_enabled()
    command = fake.values[(startup.RUN_KEY, startup.VALUE_NAME)]
    assert "pythonw.exe" in command and command.rstrip('"').endswith("plaudit_tray.pyw")
    startup.disable()
    assert not startup.is_enabled()
    startup.disable()  # disabling twice is harmless


def test_tray_icon_images():
    pytest.importorskip("pystray")
    from plaudit.tray import COLOURS, icon_image
    for state in COLOURS:
        image = icon_image(state)
        assert image.size == (64, 64) and image.getpixel((60, 32))[:3] == COLOURS[state]
