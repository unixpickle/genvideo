"""One disposable process group per web job; the queue owns its lifetime."""
from pathlib import Path
import json
import os
import signal
import sys
import threading

from generation import ComfySession


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    os.replace(temporary, path)


def run(directory, parent_pid):
    def watch_parent():
        # Also release ComfyUI if the web server crashes or receives SIGKILL.
        while True:
            if os.getppid() != parent_pid:
                os.killpg(os.getpgrp(), signal.SIGKILL)
            threading.Event().wait(0.5)

    threading.Thread(target=watch_parent, daemon=True).start()
    spec = json.loads((directory / "request.json").read_text())
    lock = threading.Lock()

    def progress(value):
        with lock:
            write_json(directory / "progress.json", value)

    session = ComfySession(spec["memory_limit"], spec["model"],
                           work_directory=directory / "runtime",
                           own_process_group=False)
    try:
        progress({"stage": "Starting engine", "step": None, "total": None})
        session.generate(
            spec["prompt"], Path(spec["output_path"]),
            image=Path(spec["upload_path"]) if spec["upload_path"] else None,
            seed=spec["seed"], duration_seconds=spec["duration_seconds"],
            model=spec["model"], resolution=spec["resolution"],
            aspect_ratio=spec["aspect_ratio"], checkpoint_directory=directory,
            progress_callback=progress,
        )
        write_json(directory / "result.json", {"ok": True})
    except Exception as exc:
        write_json(directory / "result.json", {"ok": False, "error": str(exc)[-3000:]})
    finally:
        session.close()


if __name__ == "__main__":
    run(Path(sys.argv[1]), int(sys.argv[2]))
