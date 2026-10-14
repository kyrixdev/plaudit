"""PyInstaller entry point for Plaudit-Setup.exe."""
import sys

from plaudit.installer.ui import setup_main

sys.exit(setup_main())
