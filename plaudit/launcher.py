"""Entry point of the installed Plaudit.exe.

    Plaudit.exe              open the settings window; start listening in the tray if it isn't running
    Plaudit.exe --tray       listen in the system tray only (used by "Start with Windows")
    Plaudit.exe --settings   open the settings window only
    Plaudit.exe --uninstall  the uninstaller (Windows' Installed apps runs this)
    Plaudit.exe --diagnose   write an audio report to logs/diagnostics.txt
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if "--tray" in args:
        from .tray import main as tray_main
        return tray_main()
    if "--diagnose" in args:
        from .diagnose import main as diagnose_main
        return diagnose_main()
    if "--uninstall" in args:
        from .installer.ui import run_uninstaller
        return run_uninstaller()
    if "--settings" not in args:
        from .gui.api import tray_running
        from .paths import launch
        if not tray_running():
            launch("tray")
    from .gui.app import main as settings_main
    return settings_main()


if __name__ == "__main__":
    sys.exit(main())
