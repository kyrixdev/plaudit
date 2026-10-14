"""Microphone capture on Windows through WASAPI (via PortAudio / sounddevice).

Laptop mic arrays usually run noise suppression, echo cancellation and AGC
before apps see the audio. Those effects can smear or erase a short desk tap,
so the default here is WASAPI "raw" mode, which asks Windows to skip them.
"""

from __future__ import annotations

import queue
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd

MODES = ("raw", "shared", "exclusive")
EXTERNAL_HINTS = ("headset", "usb", "bluetooth", "hands-free", "airpods", "webcam", "camera")


class AudioSetupError(RuntimeError):
    pass


@dataclass
class InputDevice:
    index: int
    name: str
    channels: int
    default_sample_rate: float

    @property
    def looks_external(self) -> bool:
        lowered = self.name.lower()
        return any(hint in lowered for hint in EXTERNAL_HINTS)


def _wasapi_index() -> int:
    for index, api in enumerate(sd.query_hostapis()):
        if "WASAPI" in api["name"]:
            return index
    raise AudioSetupError("Windows WASAPI audio is not available.")


def wasapi_input_devices() -> list[InputDevice]:
    api = _wasapi_index()
    devices = []
    for index, info in enumerate(sd.query_devices()):
        if info["hostapi"] == api and info["max_input_channels"] > 0:
            devices.append(InputDevice(index, info["name"], info["max_input_channels"], info["default_samplerate"]))
    return devices


def default_input_device() -> InputDevice:
    api = sd.query_hostapis(_wasapi_index())
    index = api["default_input_device"]
    if index < 0:
        raise AudioSetupError("No default microphone. Check Settings > System > Sound > Input.")
    return resolve_device(index)


def resolve_device(selector: int | str | None) -> InputDevice:
    if selector is None:
        return default_input_device()
    devices = wasapi_input_devices()
    if isinstance(selector, int) or str(selector).isdigit():
        for device in devices:
            if device.index == int(selector):
                return device
        raise AudioSetupError(f"No WASAPI input device with index {selector}. Run 'plaudit devices'.")
    matches = [d for d in devices if str(selector).lower() in d.name.lower()]
    if not matches:
        raise AudioSetupError(f"No WASAPI input device matching '{selector}'. Run 'plaudit devices'.")
    return matches[0]


def _wasapi_settings(mode: str) -> sd.WasapiSettings:
    settings = sd.WasapiSettings(exclusive=(mode == "exclusive"))
    if mode == "raw":
        # AUDCLNT_STREAMOPTIONS_RAW: bypass the driver's signal-processing effects.
        settings._streaminfo.streamOption = sd._lib.eStreamOptionRaw
    return settings


class AudioInput:
    """Delivers (channels, frames) float32 blocks from a background PortAudio thread."""

    def __init__(self, device: InputDevice, mode: str = "raw", blocksize: int = 512):
        if mode not in MODES:
            raise AudioSetupError(f"Unknown audio mode '{mode}'. Use one of: {', '.join(MODES)}.")
        self.device = device
        self.mode = mode
        self.blocksize = blocksize
        self.sample_rate = float(device.default_sample_rate)
        self.channels = device.channels
        self.overflows = 0
        self.input_latency = 0.0  # seconds, as reported by the driver
        self.last_block_time = 0.0  # perf_counter() when the last read block arrived
        self._queue: queue.Queue[tuple[float, np.ndarray]] = queue.Queue(maxsize=400)
        self._stream: sd.InputStream | None = None

    def _callback(self, indata, frames, time_info, status) -> None:
        if status.input_overflow:
            self.overflows += 1
        try:
            self._queue.put_nowait((time.perf_counter(), indata.T.copy()))
        except queue.Full:
            self.overflows += 1

    def start(self) -> "AudioInput":
        try:
            self._stream = sd.InputStream(
                device=self.device.index,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.blocksize,
                dtype="float32",
                latency="low",
                extra_settings=_wasapi_settings(self.mode),
                callback=self._callback,
            )
            self._stream.start()
            self.input_latency = float(self._stream.latency)
        except sd.PortAudioError as error:
            self._stream = None
            hint = ""
            if self.mode == "exclusive":
                hint = " Exclusive mode needs 'Allow applications to take exclusive control' enabled for the mic."
            raise AudioSetupError(f"Could not open the microphone in {self.mode} mode: {error}.{hint}") from error
        return self

    def read(self, timeout: float = 0.1) -> np.ndarray | None:
        try:
            arrived, block = self._queue.get(timeout=timeout) if timeout > 0 else self._queue.get_nowait()
        except queue.Empty:
            return None
        self.last_block_time = arrived
        return block

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self) -> "AudioInput":
        return self.start() if self._stream is None else self

    def __exit__(self, *exc) -> None:
        self.stop()


def open_input(device: InputDevice, mode: str, blocksize: int = 512) -> tuple[AudioInput, str | None]:
    """Opens the mic, falling back from raw to shared mode. Returns (input, warning)."""
    try:
        return AudioInput(device, mode, blocksize).start(), None
    except AudioSetupError as error:
        if mode != "raw":
            raise
        warning = f"Raw mode unavailable ({error}); using shared mode, which may include noise suppression."
        return AudioInput(device, "shared", blocksize).start(), warning


class WavWriter:
    """Streams 32-bit float WAV, so quiet taps keep full precision for offline replay."""

    def __init__(self, path: Path, sample_rate: float, channels: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_rate = int(sample_rate)
        self.channels = channels
        self.frames = 0
        self._file = open(self.path, "wb")
        self._write_header()

    def _write_header(self) -> None:
        data_bytes = self.frames * self.channels * 4
        block_align = self.channels * 4
        self._file.seek(0)
        self._file.write(b"RIFF" + struct.pack("<I", 36 + data_bytes) + b"WAVE")
        self._file.write(b"fmt " + struct.pack("<IHHIIHH", 16, 3, self.channels, self.sample_rate,
                                                self.sample_rate * block_align, block_align, 32))
        self._file.write(b"data" + struct.pack("<I", data_bytes))

    def write(self, block: np.ndarray) -> None:
        """block is (channels, frames)."""
        self._file.write(np.ascontiguousarray(block.T, dtype="<f4").tobytes())
        self.frames += block.shape[1]

    def close(self) -> None:
        if self._file.closed:
            return
        self._write_header()
        self._file.close()


def read_wav(path: Path) -> tuple[np.ndarray, float]:
    """Reads float32 or int16 PCM WAV. Returns ((channels, frames) float32, sample_rate)."""
    data = Path(path).read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Not a WAV file")
    offset, fmt, payload = 12, None, None
    while offset + 8 <= len(data):
        chunk_id, size = data[offset:offset + 4], struct.unpack("<I", data[offset + 4:offset + 8])[0]
        body = data[offset + 8:offset + 8 + size]
        if chunk_id == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif chunk_id == b"data":
            payload = body
        offset += 8 + size + (size & 1)
    if fmt is None or payload is None:
        raise ValueError("WAV is missing fmt or data")
    audio_format, channels, sample_rate, _, _, bits = fmt
    if audio_format == 3 and bits == 32:
        samples = np.frombuffer(payload, dtype="<f4")
    elif audio_format == 1 and bits == 16:
        samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768
    else:
        raise ValueError(f"Unsupported WAV format {audio_format}/{bits}-bit")
    frames = len(samples) // channels
    return samples[: frames * channels].reshape(frames, channels).T.astype(np.float32), float(sample_rate)
