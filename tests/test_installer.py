"""Installer and uninstaller logic, run against temporary folders and a fake registry."""

import sys
import time
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plaudit.installer import system, ui
from test_tray import FakeRegistry


def wait_until_done(api, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        progress = api.progress()
        if progress["done"]:
            return progress
        time.sleep(0.05)
    raise AssertionError("flow did not finish")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Redirects every location the installer touches into tmp_path."""
    registry = FakeRegistry()
    monkeypatch.setattr(system, "winreg", registry)
    monkeypatch.setattr(ui.startup, "winreg", registry)
    monkeypatch.setattr(system, "start_menu_shortcut", lambda: tmp_path / "start" / "Plaudit.lnk")
    monkeypatch.setattr(system, "desktop_shortcut", lambda: tmp_path / "desktop" / "Plaudit.lnk")
    monkeypatch.setattr(system, "stop_running", lambda exclude_pid=None: None)
    created = []
    monkeypatch.setattr(system, "create_shortcut",
                        lambda link, target, description: (link.parent.mkdir(parents=True, exist_ok=True),
                                                           link.write_text(str(target)), created.append(link)))
    payload = tmp_path / "payload.zip"
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("Plaudit.exe", b"MZ fake exe")
        archive.writestr("_internal/base_library.zip", b"x" * 5000)
    return {"root": tmp_path, "registry": registry, "payload": payload, "shortcuts": created}


def test_install_copies_files_and_registers(sandbox):
    target = sandbox["root"] / "Programs" / "Plaudit"
    api = ui.SetupApi(sandbox["payload"], install_dir=target)
    assert api.get_info()["existingVersion"] is None
    api.install({"desktopShortcut": True, "startWithWindows": True})
    result = wait_until_done(api)
    assert result["error"] is None and result["percent"] == 100
    assert (target / "Plaudit.exe").read_bytes() == b"MZ fake exe"
    assert (target / "_internal" / "base_library.zip").exists()
    assert len(sandbox["shortcuts"]) == 2
    values = sandbox["registry"].values
    assert values[(system.UNINSTALL_KEY, "UninstallString")] == f'"{target / "Plaudit.exe"}" --uninstall'
    assert values[(system.UNINSTALL_KEY, "DisplayName")] == "Plaudit"
    assert values[(ui.startup.RUN_KEY, ui.startup.VALUE_NAME)] == f'"{target / "Plaudit.exe"}" --tray'
    assert ui.SetupApi(sandbox["payload"], install_dir=target).get_info()["existingVersion"] == ui.__version__


def test_reinstall_replaces_the_previous_version_and_respects_options(sandbox):
    target = sandbox["root"] / "Programs" / "Plaudit"
    api = ui.SetupApi(sandbox["payload"], install_dir=target)
    api.install({})
    wait_until_done(api)
    (target / "stale-file.dll").write_text("old")
    api.install({"desktopShortcut": False, "startWithWindows": False})
    assert wait_until_done(api)["error"] is None
    assert not (target / "stale-file.dll").exists()
    assert (ui.startup.RUN_KEY, ui.startup.VALUE_NAME) not in sandbox["registry"].values


def test_install_refuses_a_folder_that_is_not_plaudit(sandbox):
    target = sandbox["root"] / "Programs" / "Plaudit"
    target.mkdir(parents=True)
    (target / "someone-elses-file.txt").write_text("keep me")
    api = ui.SetupApi(sandbox["payload"], install_dir=target)
    api.install({})
    result = wait_until_done(api)
    assert result["error"] and "doesn't look like a Plaudit install" in result["error"]
    assert (target / "someone-elses-file.txt").read_text() == "keep me"


def test_missing_payload_is_reported(sandbox):
    api = ui.SetupApi(sandbox["root"] / "nope.zip", install_dir=sandbox["root"] / "Plaudit")
    api.install({})
    assert "missing its app files" in wait_until_done(api)["error"]


def test_uninstall_refuses_to_run_from_source(sandbox):
    """From source, DATA_DIR is the project folder: nothing destructive may happen."""
    api = ui.UninstallApi()
    api.uninstall(True)
    result = wait_until_done(api)
    assert result["error"] and "isn't installed" in result["error"]
    assert Path(ui.DATA_DIR / "plaudit" / "__init__.py").exists()  # project untouched


def test_shortcut_creation_for_real(tmp_path):
    """Uses PowerShell's WScript.Shell, writing only into tmp_path."""
    link = tmp_path / "Plaudit.lnk"
    system.create_shortcut(link, Path(sys.executable), "test")
    assert link.exists() and link.stat().st_size > 0
