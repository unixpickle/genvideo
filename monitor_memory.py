"""Stop a generation process before system memory pressure becomes dangerous."""

from __future__ import annotations

import argparse
import csv
import os
import signal
import time
from pathlib import Path

import psutil


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pid", type=int)
    parser.add_argument("--minimum-available-gib", type=float, default=10.0)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--log", type=Path, default=Path("outputs/memory.csv"))
    args = parser.parse_args()

    process = psutil.Process(args.pid)
    minimum_available = int(args.minimum_available_gib * 1024**3)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    with args.log.open("w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(("unix_time", "process_rss_gib", "system_available_gib"))
        while process.is_running():
            memory = psutil.virtual_memory()
            rss = process.memory_info().rss
            writer.writerow((time.time(), rss / 1024**3, memory.available / 1024**3))
            log_file.flush()
            if memory.available < minimum_available:
                print(
                    f"Stopping PID {args.pid}: only {memory.available / 1024**3:.1f} GiB "
                    f"available (floor: {args.minimum_available_gib:.1f} GiB).",
                    flush=True,
                )
                os.kill(args.pid, signal.SIGINT)
                raise SystemExit(2)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
