"""Starts the Plaudit tray app without a console window (run with pythonw.exe)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plaudit.tray import main

sys.exit(main())
