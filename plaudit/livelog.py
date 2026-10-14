"""CSV log of every live-mode decision, so whole sessions can be reviewed afterwards."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
COLUMNS = ["timestamp", "stream_seconds", "recognised", "confidence", "response_ms", "rejection_reason",
           "peak", "snr_db", "trigger", "action"]


class LiveLog:
    def __init__(self, directory: Path = LOGS_DIR, prefix: str = "live"):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{prefix}-{datetime.now():%Y%m%d-%H%M%S}.csv"
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(COLUMNS)

    def write(self, *, stream_seconds: float, recognised: str, confidence: float, response_ms: float,
              rejection_reason: str, peak: float, snr_db: float, trigger: str, action: str) -> None:
        self._writer.writerow([
            datetime.now().isoformat(timespec="milliseconds"), f"{stream_seconds:.3f}", recognised,
            f"{confidence:.3f}", f"{response_ms:.1f}", rejection_reason, f"{peak:.4f}", f"{snr_db:.1f}",
            trigger, action,
        ])
        self._file.flush()

    def close(self) -> None:
        self._file.close()
