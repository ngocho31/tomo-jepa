"""Low-overhead background GPU telemetry for ToMo-JEPA training."""

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
from threading import Event, Thread
from typing import Any


class HardwareTelemetryMonitor:
    """Periodically log passive GPU telemetry without running model workloads."""

    def __init__(self, log_path: str | Path, interval_seconds: float = 60.0) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.log_path = Path(log_path).expanduser()
        self.interval_seconds = interval_seconds
        self._stop_event = Event()
        self._thread = Thread(target=self._run, name="hardware-telemetry", daemon=True)

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join()

    def _run(self) -> None:
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is None:
            record = {
                "timestamp": datetime.now().astimezone().isoformat(),
                "source": "background_telemetry",
                "gpu_metrics_error": "nvidia-smi is unavailable",
            }
            self._write_record(record)
            return

        while not self._stop_event.is_set():
            record: dict[str, Any] = {
                "timestamp": datetime.now().astimezone().isoformat(),
                "source": "background_telemetry",
            }
            try:
                result = subprocess.run(
                    [
                        nvidia_smi,
                        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                record["gpus"] = [
                    {
                        "index": int(row[0].strip()),
                        "name": row[1].strip(),
                        "utilization_percent": _parse_optional_float(row[2]),
                        "memory_used_mb": _parse_optional_float(row[3]),
                        "memory_total_mb": _parse_optional_float(row[4]),
                        "temperature_c": _parse_optional_float(row[5]),
                        "power_w": _parse_optional_float(row[6]),
                    }
                    for row in csv.reader(result.stdout.splitlines())
                    if len(row) == 7
                ]
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                record["gpu_metrics_error"] = str(error)
                self._write_record(record)
                return

            self._write_record(record)
            self._stop_event.wait(self.interval_seconds)

    def _write_record(self, record: dict[str, Any]) -> None:
        try:
            with self.log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(record) + "\n")
        except OSError as error:
            print(f"[WARN] Hardware telemetry could not write {self.log_path}: {error}", flush=True)


def _parse_optional_float(value: str) -> float | None:
    normalized = value.strip()
    if normalized in {"", "N/A", "[Not Supported]"}:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None
