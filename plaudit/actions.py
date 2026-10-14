"""Zone actions: what a recognised tap does. Windows counterpart of Holo's
LocalActionPlanning.swift and LocalActions.swift.

Planning (validating a saved action into a concrete command) is pure and
testable; dispatching performs the side effect through Windows APIs.
Actions only run from `plaudit live`, never during calibration or the accuracy test.
"""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
import time
from dataclasses import dataclass
from urllib.parse import urlparse


class ActionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ActionKind:
    key: str
    title: str
    prompt: str  # what the value means; empty when the kind takes no value
    choices: tuple[str, ...] = ()


MEDIA_KEYS = {
    "play_pause": (0xB3, "Play/pause"),
    "next": (0xB0, "Next track"),
    "previous": (0xB1, "Previous track"),
    "stop": (0xB2, "Stop media"),
    "volume_up": (0xAF, "Volume up"),
    "volume_down": (0xAE, "Volume down"),
    "mute": (0xAD, "Mute"),
}

SOUNDS = {
    "default": "SystemDefault",
    "asterisk": "SystemAsterisk",
    "exclamation": "SystemExclamation",
    "critical": "SystemHand",
    "question": "SystemQuestion",
    "notification": "Notification.Default",
}

KINDS = {k.key: k for k in [
    ActionKind("none", "Feedback only", ""),
    ActionKind("media", "Media key", "which key", tuple(MEDIA_KEYS)),
    ActionKind("hotkey", "Keyboard shortcut", "keys, e.g. win+d, ctrl+alt+t, alt+tab"),
    ActionKind("open", "Open app, file or folder", "path, app name or URI, e.g. notepad, C:\\Users\\me\\Music, spotify:"),
    ActionKind("url", "Open website", "web address, e.g. https://example.com"),
    ActionKind("command", "Run PowerShell command", "command (runs hidden, with your permissions)"),
    ActionKind("screenshot", "Screenshot", "full screen to clipboard, or selection with the snipping overlay",
               ("full", "selection")),
    ActionKind("sound", "Play Windows sound", "sound name or .wav path", tuple(SOUNDS)),
    ActionKind("copy_text", "Copy text to clipboard", "text"),
    ActionKind("speak_text", "Speak text aloud", "text"),
]}


@dataclass
class ZoneAction:
    kind: str = "none"
    value: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "value": self.value}

    @classmethod
    def from_dict(cls, data: dict | None) -> "ZoneAction":
        if not data or data.get("kind") not in KINDS:
            return cls()
        return cls(data["kind"], str(data.get("value", "")))

    @property
    def summary(self) -> str:
        kind = KINDS.get(self.kind, KINDS["none"])
        if self.kind == "none":
            return kind.title
        if self.kind == "media":
            return MEDIA_KEYS.get(self.value, (0, self.value))[1]
        if self.kind == "screenshot":
            return "Screenshot to clipboard" if self.value == "full" else "Snip a selection"
        value = self.value if len(self.value) <= 48 else self.value[:45] + "..."
        return f"{kind.title}: {value}"


# -- planning -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Command:
    kind: str
    args: tuple


_KEY_NAMES = {
    **{chr(c): c for c in range(ord("A"), ord("Z") + 1)},
    **{str(d): 0x30 + d for d in range(10)},
    **{f"F{n}": 0x6F + n for n in range(1, 25)},
    "ESC": 0x1B, "ESCAPE": 0x1B, "ENTER": 0x0D, "RETURN": 0x0D, "TAB": 0x09, "SPACE": 0x20,
    "BACKSPACE": 0x08, "DELETE": 0x2E, "DEL": 0x2E, "INSERT": 0x2D, "HOME": 0x24, "END": 0x23,
    "PAGEUP": 0x21, "PGUP": 0x21, "PAGEDOWN": 0x22, "PGDN": 0x22,
    "UP": 0x26, "DOWN": 0x28, "LEFT": 0x25, "RIGHT": 0x27,
    "PRTSC": 0x2C, "PRINTSCREEN": 0x2C, "CAPSLOCK": 0x14,
    "MINUS": 0xBD, "PLUS": 0xBB, "EQUALS": 0xBB, "COMMA": 0xBC, "PERIOD": 0xBE,
    **{name.upper(): vk for name, (vk, _) in MEDIA_KEYS.items()},
}
_MODIFIERS = {"CTRL": 0x11, "CONTROL": 0x11, "ALT": 0x12, "SHIFT": 0x10, "WIN": 0x5B, "WINDOWS": 0x5B}
# Keys that need KEYEVENTF_EXTENDEDKEY so they aren't read as numpad keys.
_EXTENDED = {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E, 0x5B, 0x2C,
             *(vk for vk, _ in MEDIA_KEYS.values())}


def parse_hotkey(text: str) -> tuple[int, ...]:
    """'ctrl+shift+esc' -> virtual-key codes, modifiers first."""
    parts = [p.strip().upper() for p in text.split("+")]
    if not text.strip() or any(not p for p in parts):
        raise ActionError(f"'{text}' isn't a complete shortcut. Use keys such as win+d or ctrl+alt+t.")
    *modifiers, key = parts
    codes = []
    for modifier in modifiers:
        if modifier not in _MODIFIERS:
            raise ActionError(f"'{modifier.lower()}' in '{text}' isn't a modifier (ctrl, alt, shift, win).")
        codes.append(_MODIFIERS[modifier])
    if key in _KEY_NAMES:
        codes.append(_KEY_NAMES[key])
    elif key in _MODIFIERS and not modifiers:
        codes.append(_MODIFIERS[key])  # a lone modifier, e.g. "win" opens Start
    elif key in _MODIFIERS:
        raise ActionError(f"'{text}' ends with a modifier; add the key to press, e.g. {text}+d.")
    else:
        raise ActionError(f"Unknown key '{key.lower()}' in '{text}'.")
    return tuple(codes)


def normalize_url(text: str) -> str:
    candidate = text.strip()
    if not candidate:
        raise ActionError("Enter a web address.")
    if "://" not in candidate:
        candidate = "https://" + candidate
    parsed = urlparse(candidate)
    try:
        parsed.port  # raises on a malformed port, e.g. "javascript:alert(1)" read as host:port
    except ValueError:
        parsed = None
    if parsed is None or parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
        raise ActionError(f"'{text}' is not an http(s) web address.")
    return candidate


def plan(action: ZoneAction) -> Command | None:
    """Validates a saved action. None means feedback only; raises ActionError if misconfigured."""
    kind, value = action.kind, action.value.strip()
    if kind == "none":
        return None
    if kind not in KINDS:
        raise ActionError(f"Unknown action type '{kind}'.")
    if kind == "media":
        if value not in MEDIA_KEYS:
            raise ActionError(f"Media key must be one of: {', '.join(MEDIA_KEYS)}.")
        return Command("keys", (MEDIA_KEYS[value][0],))
    if kind == "hotkey":
        return Command("keys", parse_hotkey(value))
    if kind == "url":
        return Command("open", (normalize_url(value),))
    if kind == "open":
        if not value:
            raise ActionError("Enter an app name, path or URI to open.")
        expanded = os.path.expandvars(os.path.expanduser(value.strip('"')))
        # A URI scheme has 2+ letters ("spotify:", "ms-settings:"); "C:" is a drive.
        is_uri = re.match(r"^[a-z][a-z0-9+.-]+:", expanded, re.I) is not None
        if not is_uri and ("\\" in expanded or "/" in expanded) and not os.path.exists(expanded):
            raise ActionError(f"'{value}' doesn't exist.")
        return Command("open", (expanded,))
    if kind == "command":
        if not value or "\0" in value:
            raise ActionError("Enter a PowerShell command.")
        return Command("command", (value,))
    if kind == "screenshot":
        if value not in ("full", "selection"):
            raise ActionError("Screenshot must be 'full' or 'selection'.")
        return Command("screenshot_full", ()) if value == "full" else Command("open", ("ms-screenclip:",))
    if kind == "sound":
        if value.lower().endswith(".wav"):
            path = os.path.expandvars(os.path.expanduser(value.strip('"')))
            if not os.path.isfile(path):
                raise ActionError(f"'{value}' doesn't exist.")
            return Command("sound_file", (path,))
        if value not in SOUNDS:
            raise ActionError(f"Sound must be a .wav path or one of: {', '.join(SOUNDS)}.")
        return Command("sound_alias", (SOUNDS[value],))
    if kind in ("copy_text", "speak_text"):
        if not value:
            raise ActionError("Enter the text.")
        return Command(kind, (action.value,))
    raise ActionError(f"Unhandled action type '{kind}'.")


# -- dispatch -----------------------------------------------------------------------------

CREATE_NO_WINDOW = 0x08000000
POWERSHELL = ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command"]


class ActionDispatcher:
    def __init__(self):
        self._running: list[tuple[str, subprocess.Popen]] = []

    def perform(self, action: ZoneAction) -> str | None:
        """Runs the action. Returns a short description, or None for feedback-only zones."""
        command = plan(action)
        if command is None:
            return None
        handler = getattr(self, f"_do_{command.kind}")
        handler(*command.args)
        return action.summary

    def poll_errors(self) -> list[str]:
        """Reports background commands that finished with an error."""
        errors, still_running = [], []
        for name, process in self._running:
            code = process.poll()
            if code is None:
                still_running.append((name, process))
            elif code != 0:
                detail = ""
                if process.stderr:
                    detail = process.stderr.read().decode("utf-8", "replace").strip().splitlines()
                    detail = f": {detail[-1]}" if detail else ""
                errors.append(f"{name} failed (exit code {code}){detail}")
        self._running = still_running
        return errors

    def wait(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        for _, process in self._running:
            process.wait(max(0.0, deadline - time.monotonic()))

    # -- handlers ---------------------------------------------------------------------------

    def _launch(self, name: str, arguments: list[str], env: dict | None = None) -> None:
        process = subprocess.Popen(arguments, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, creationflags=CREATE_NO_WINDOW, env=env)
        self._running.append((name, process))

    def _do_keys(self, *codes: int) -> None:
        press_keys(codes)

    def _do_open(self, target: str) -> None:
        try:
            os.startfile(target)
        except OSError as error:
            raise ActionError(f"Windows could not open '{target}': {error.strerror or error}") from error

    def _do_command(self, command: str) -> None:
        self._launch("PowerShell command", POWERSHELL + [command])

    def _do_speak_text(self, text: str) -> None:
        # The text travels in an environment variable so it is never parsed as code.
        env = dict(os.environ, HOLO_SPEAK_TEXT=text)
        script = ("Add-Type -AssemblyName System.Speech; "
                  "(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak($env:HOLO_SPEAK_TEXT)")
        self._launch("Speech", POWERSHELL + [script], env)

    def _do_copy_text(self, text: str) -> None:
        set_clipboard_text(text)

    def _do_screenshot_full(self) -> None:
        bitmap = capture_screen()
        set_clipboard_bitmap(bitmap)

    def _do_sound_alias(self, alias: str) -> None:
        import winsound
        try:
            winsound.PlaySound(alias, winsound.SND_ALIAS | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
        except RuntimeError as error:
            raise ActionError(f"The Windows sound '{alias}' is unavailable.") from error

    def _do_sound_file(self, path: str) -> None:
        import winsound
        try:
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
        except RuntimeError as error:
            raise ActionError(f"Could not play '{path}'.") from error


# -- Win32 helpers ------------------------------------------------------------------------

def _user32():
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.keybd_event.argtypes = [ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_uint32, ctypes.c_void_p]
    user32.MapVirtualKeyW.restype = ctypes.c_uint
    user32.OpenClipboard.argtypes = [ctypes.c_void_p]
    user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
    user32.SetClipboardData.restype = ctypes.c_void_p
    user32.GetDC.argtypes = [ctypes.c_void_p]
    user32.GetDC.restype = ctypes.c_void_p
    user32.ReleaseDC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return user32


def press_keys(codes) -> None:
    """Presses the keys in order, then releases them in reverse (a chord)."""
    user32 = _user32()
    keydown, keyup, extended = 0x0, 0x2, 0x1
    for vk in codes:
        flags = keydown | (extended if vk in _EXTENDED else 0)
        user32.keybd_event(vk, user32.MapVirtualKeyW(vk, 0) & 0xFF, flags, None)
    for vk in reversed(codes):
        flags = keyup | (extended if vk in _EXTENDED else 0)
        user32.keybd_event(vk, user32.MapVirtualKeyW(vk, 0) & 0xFF, flags, None)


class _Clipboard:
    def __enter__(self):
        user32 = _user32()
        for _ in range(20):  # another app may briefly hold the clipboard
            if user32.OpenClipboard(None):
                user32.EmptyClipboard()
                self.user32 = user32
                return user32
            time.sleep(0.025)
        raise ActionError("The clipboard is busy; try again.")

    def __exit__(self, *exc):
        self.user32.CloseClipboard()


def set_clipboard_text(text: str) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    data = (text + "\0").encode("utf-16-le")
    handle = kernel32.GlobalAlloc(0x0002, len(data))  # GMEM_MOVEABLE
    if not handle:
        raise ActionError("Could not allocate clipboard memory.")
    pointer = kernel32.GlobalLock(handle)
    ctypes.memmove(pointer, data, len(data))
    kernel32.GlobalUnlock(handle)
    with _Clipboard() as user32:
        if not user32.SetClipboardData(13, handle):  # CF_UNICODETEXT; the clipboard now owns handle
            raise ActionError("Could not write to the clipboard.")


def get_clipboard_text() -> str | None:
    user32 = _user32()
    user32.GetClipboardData.restype = ctypes.c_void_p
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    if not user32.OpenClipboard(None):
        return None
    try:
        handle = user32.GetClipboardData(13)
        if not handle:
            return None
        pointer = kernel32.GlobalLock(handle)
        try:
            return ctypes.wstring_at(pointer)
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


@dataclass
class ScreenBitmap:
    handle: int
    width: int
    height: int


def capture_screen() -> ScreenBitmap:
    """Captures every monitor (the virtual screen) into a GDI bitmap at full resolution."""
    user32 = _user32()
    gdi32 = ctypes.WinDLL("gdi32")
    gdi32.CreateCompatibleDC.argtypes = [ctypes.c_void_p]
    gdi32.CreateCompatibleDC.restype = ctypes.c_void_p
    gdi32.CreateCompatibleBitmap.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    gdi32.CreateCompatibleBitmap.restype = ctypes.c_void_p
    gdi32.SelectObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    gdi32.SelectObject.restype = ctypes.c_void_p
    gdi32.BitBlt.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint32]
    gdi32.DeleteDC.argtypes = [ctypes.c_void_p]
    try:  # physical pixels on scaled displays
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except AttributeError:
        user32.SetProcessDPIAware()
    x, y = user32.GetSystemMetrics(76), user32.GetSystemMetrics(77)
    width, height = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79)
    screen = user32.GetDC(None)
    memory = gdi32.CreateCompatibleDC(screen)
    bitmap = gdi32.CreateCompatibleBitmap(screen, width, height)
    previous = gdi32.SelectObject(memory, bitmap)
    ok = gdi32.BitBlt(memory, 0, 0, width, height, screen, x, y, 0x00CC0020 | 0x40000000)  # SRCCOPY|CAPTUREBLT
    gdi32.SelectObject(memory, previous)
    gdi32.DeleteDC(memory)
    user32.ReleaseDC(None, screen)
    if not ok or not bitmap:
        raise ActionError("Screen capture failed.")
    return ScreenBitmap(bitmap, width, height)


def set_clipboard_bitmap(bitmap: ScreenBitmap) -> None:
    with _Clipboard() as user32:
        if not user32.SetClipboardData(2, bitmap.handle):  # CF_BITMAP; the clipboard now owns it
            raise ActionError("Could not put the screenshot on the clipboard.")
