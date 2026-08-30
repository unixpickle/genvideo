"""Single-user mobile web queue for LTX-2.5 video generation."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aiohttp import web

from main import (
    DEFAULT_DURATION_SECONDS,
    SUPPORTED_DURATION_SECONDS,
    ComfySession,
    ROOT,
    _random_seed,
)


WEB_DIR = ROOT / "web"
DEFAULT_OUTPUT_DIR = ROOT / "web_outputs"
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
STATE_FILENAME = "queue-state.json"


@dataclass(slots=True)
class Job:
    id: str
    prompt: str
    mode: str
    seed: int
    duration_seconds: int
    output_path: Path
    upload_path: Path | None = None
    image_content_type: str | None = None
    status: str = "queued"
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    def as_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "mode": self.mode,
            "seed": self.seed,
            "duration_seconds": self.duration_seconds,
            "output_filename": self.output_path.name,
            "upload_filename": (
                self.upload_path.name if self.upload_path is not None else None
            ),
            "image_content_type": self.image_content_type,
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    def as_dict(self, queue_position: int | None = None) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "mode": self.mode,
            "seed": self.seed,
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "queue_position": queue_position,
            "image_prompt_url": (
                f"/prompt-images/{self.id}"
                if self.mode == "image"
                and self.upload_path is not None
                and self.upload_path.is_file()
                else None
            ),
            "media_url": (
                f"/media/{self.output_path.name}"
                if self.status == "completed" and self.output_path.is_file()
                else None
            ),
        }


class QueueManager:
    def __init__(self, output_directory: Path, memory_limit: float):
        self.output_directory = output_directory
        self.upload_directory = output_directory / ".uploads"
        self.state_path = output_directory / STATE_FILENAME
        self.memory_limit = memory_limit
        self.jobs: dict[str, Job] = {}
        self.queue: asyncio.Queue[Job] = asyncio.Queue()
        self.session: ComfySession | None = None
        self.worker_task: asyncio.Task[None] | None = None
        self.engine_state = "idle"

    def start(self) -> None:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.upload_directory.mkdir(parents=True, exist_ok=True)
        self._load_state()
        self.worker_task = asyncio.create_task(self._worker(), name="video-queue-worker")

    async def close(self) -> None:
        if self.worker_task is not None:
            self.worker_task.cancel()
        session, self.session = self.session, None
        if session is not None:
            await asyncio.to_thread(session.close)
        if self.worker_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self.worker_task
        self.engine_state = "idle"
        self._save_state()

    def _load_state(self) -> None:
        if not self.state_path.is_file():
            return
        state = json.loads(self.state_path.read_text())
        if not isinstance(state, dict) or state.get("version") != 1:
            raise ValueError(f"unsupported queue state in {self.state_path}")
        records = state.get("jobs")
        if not isinstance(records, list):
            raise ValueError(f"invalid queue state in {self.state_path}")

        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"invalid job in {self.state_path}")
            output_filename = str(record["output_filename"])
            upload_filename = record.get("upload_filename")
            if Path(output_filename).name != output_filename:
                raise ValueError(f"invalid output filename in {self.state_path}")
            if upload_filename is not None:
                upload_filename = str(upload_filename)
                if Path(upload_filename).name != upload_filename:
                    raise ValueError(f"invalid upload filename in {self.state_path}")
            job = Job(
                id=str(record["id"]),
                prompt=str(record["prompt"]),
                mode=str(record["mode"]),
                seed=int(record["seed"]),
                duration_seconds=int(
                    record.get("duration_seconds", DEFAULT_DURATION_SECONDS)
                ),
                output_path=self.output_directory / output_filename,
                upload_path=(
                    self.upload_directory / upload_filename
                    if upload_filename is not None
                    else None
                ),
                image_content_type=record.get("image_content_type"),
                status=str(record["status"]),
                error=record.get("error"),
                created_at=float(record["created_at"]),
                started_at=(
                    float(record["started_at"])
                    if record.get("started_at") is not None
                    else None
                ),
                finished_at=(
                    float(record["finished_at"])
                    if record.get("finished_at") is not None
                    else None
                ),
            )
            if job.status == "cancelled":
                continue
            if job.status in {"queued", "running"}:
                if job.mode == "image" and (
                    job.upload_path is None or not job.upload_path.is_file()
                ):
                    job.status = "failed"
                    job.error = (
                        "The starting image was lost before this job could resume."
                    )
                    job.finished_at = time.time()
                else:
                    job.status = "queued"
                    job.started_at = None
                    job.finished_at = None
                    self.queue.put_nowait(job)
            self.jobs[job.id] = job
        self._save_state()

    def _save_state(self) -> None:
        state = {
            "version": 1,
            "jobs": [job.as_record() for job in self.jobs.values()],
        }
        temporary_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        try:
            temporary_path.write_text(json.dumps(state, indent=2) + "\n")
            temporary_path.replace(self.state_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def add(
        self,
        prompt: str,
        mode: str,
        seed: int,
        duration_seconds: int,
        upload_path: Path | None,
        image_content_type: str | None = None,
    ) -> Job:
        job_id = uuid.uuid4().hex[:12]
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        job = Job(
            id=job_id,
            prompt=prompt,
            mode=mode,
            seed=seed,
            duration_seconds=duration_seconds,
            upload_path=upload_path,
            image_content_type=image_content_type,
            output_path=self.output_directory / f"{stamp}-{job_id}.mp4",
        )
        self.jobs[job.id] = job
        try:
            self._save_state()
        except BaseException:
            self.jobs.pop(job.id, None)
            raise
        self.queue.put_nowait(job)
        return job

    def regenerate(self, job_id: str) -> Job:
        original = self.jobs.get(job_id)
        if original is None:
            raise web.HTTPNotFound(text="job not found")

        copied_upload: Path | None = None
        try:
            if original.mode == "image":
                if original.upload_path is None or not original.upload_path.is_file():
                    raise web.HTTPConflict(
                        text="the starting image for this job is no longer available"
                    )
                copied_upload = self.upload_directory / f"{uuid.uuid4().hex}.upload"
                shutil.copyfile(original.upload_path, copied_upload)
            return self.add(
                original.prompt,
                original.mode,
                original.seed,
                original.duration_seconds,
                copied_upload,
                original.image_content_type,
            )
        except BaseException:
            if copied_upload is not None:
                copied_upload.unlink(missing_ok=True)
            raise

    async def remove(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if job is None:
            raise web.HTTPNotFound(text="job not found")
        if job.status == "running":
            raise web.HTTPConflict(text="a running job cannot be removed")
        if job.status == "queued":
            job.status = "cancelled"
            job.finished_at = time.time()
        if job.upload_path is not None:
            job.upload_path.unlink(missing_ok=True)
        if job.output_path.is_file():
            job.output_path.unlink()
        if job.status != "queued":
            self.jobs.pop(job_id, None)
        self._save_state()

    def snapshot(self) -> dict[str, Any]:
        queued = [job for job in self.jobs.values() if job.status == "queued"]
        positions = {job.id: index + 1 for index, job in enumerate(queued)}
        priority = {"running": 0, "queued": 1, "failed": 2, "completed": 2}
        jobs = sorted(
            self.jobs.values(),
            key=lambda job: (priority.get(job.status, 3), -job.created_at),
        )
        return {
            "engine_state": self.engine_state,
            "queued_count": len(queued),
            "jobs": [job.as_dict(positions.get(job.id)) for job in jobs],
        }

    async def _close_session(self) -> None:
        session, self.session = self.session, None
        if session is not None:
            self.engine_state = "unloading"
            await asyncio.to_thread(session.close)
        self.engine_state = "idle"

    async def _worker(self) -> None:
        try:
            while True:
                job = await self.queue.get()
                if job.status == "cancelled":
                    job.finished_at = time.time()
                    if job.upload_path is not None:
                        job.upload_path.unlink(missing_ok=True)
                        job.upload_path = None
                    self.queue.task_done()
                    if self.queue.empty():
                        await self._close_session()
                    continue
                try:
                    job.status = "running"
                    job.started_at = time.time()
                    self._save_state()
                    if self.session is None:
                        self.engine_state = "starting"
                        self.session = ComfySession(self.memory_limit)
                        await asyncio.to_thread(self.session.start)
                    self.engine_state = "working"
                    await asyncio.to_thread(
                        self.session.generate,
                        job.prompt,
                        job.output_path,
                        image=job.upload_path,
                        seed=job.seed,
                        duration_seconds=job.duration_seconds,
                    )
                    job.status = "completed"
                except asyncio.CancelledError:
                    job.status = "queued"
                    job.started_at = None
                    job.finished_at = None
                    raise
                except Exception as exc:
                    job.status = "failed"
                    job.error = str(exc)[-3000:]
                    await self._close_session()
                finally:
                    if job.status != "queued":
                        job.finished_at = time.time()
                    self.queue.task_done()
                    self._save_state()
                if self.queue.empty():
                    await self._close_session()
        except asyncio.CancelledError:
            raise


MANAGER_KEY = web.AppKey("manager", QueueManager)


def _manager(request: web.Request) -> QueueManager:
    return request.app[MANAGER_KEY]


async def index(_: web.Request) -> web.FileResponse:
    return web.FileResponse(
        WEB_DIR / "index.html", headers={"Cache-Control": "no-store"}
    )


async def get_jobs(request: web.Request) -> web.Response:
    response = web.json_response(_manager(request).snapshot())
    response.headers["Cache-Control"] = "no-store"
    return response


async def create_job(request: web.Request) -> web.Response:
    manager = _manager(request)
    job_token = uuid.uuid4().hex
    staged_upload = manager.upload_directory / f"{job_token}.upload"
    prompt = ""
    mode = "text"
    seed_text = ""
    duration_text = str(DEFAULT_DURATION_SECONDS)
    image_received = False
    image_content_type = None
    try:
        if request.content_type.startswith("multipart/"):
            reader = await request.multipart()
            async for part in reader:
                if part.name == "image" and part.filename:
                    image_content_type = part.headers.get("Content-Type") or ""
                    if not image_content_type.startswith("image/"):
                        raise web.HTTPBadRequest(text="the uploaded file must be an image")
                    size = 0
                    with staged_upload.open("wb") as upload_file:
                        while chunk := await part.read_chunk(1024 * 1024):
                            size += len(chunk)
                            if size > MAX_UPLOAD_BYTES:
                                raise web.HTTPRequestEntityTooLarge(
                                    max_size=MAX_UPLOAD_BYTES, actual_size=size
                                )
                            upload_file.write(chunk)
                    image_received = size > 0
                elif part.name in {"prompt", "mode", "seed", "duration"}:
                    value = await part.text()
                    if part.name == "prompt":
                        prompt = value.strip()
                    elif part.name == "mode":
                        mode = value
                    elif part.name == "seed":
                        seed_text = value.strip()
                    else:
                        duration_text = value.strip()
        else:
            fields = await request.post()
            prompt = str(fields.get("prompt", "")).strip()
            mode = str(fields.get("mode", "text"))
            seed_text = str(fields.get("seed", "")).strip()
            duration_text = str(
                fields.get("duration", DEFAULT_DURATION_SECONDS)
            ).strip()

        if mode not in {"text", "image"}:
            raise web.HTTPBadRequest(text="mode must be text or image")
        if not prompt:
            raise web.HTTPBadRequest(text="prompt is required")
        if len(prompt) > 8000:
            raise web.HTTPBadRequest(text="prompt must be 8,000 characters or fewer")
        if mode == "image" and not image_received:
            raise web.HTTPBadRequest(text="an initial image is required in image mode")
        if mode == "text" and staged_upload.exists():
            staged_upload.unlink()
            image_received = False
        try:
            seed = int(seed_text) if seed_text else _random_seed()
        except ValueError as exc:
            raise web.HTTPBadRequest(text="seed must be an integer") from exc
        if not 0 <= seed < 2**63:
            raise web.HTTPBadRequest(text="seed must be between 0 and 2^63-1")
        try:
            duration_seconds = int(duration_text)
        except ValueError as exc:
            raise web.HTTPBadRequest(text="duration must be an integer") from exc
        if duration_seconds not in SUPPORTED_DURATION_SECONDS:
            choices = " or ".join(str(value) for value in SUPPORTED_DURATION_SECONDS)
            raise web.HTTPBadRequest(text=f"duration must be {choices} seconds")

        job = manager.add(
            prompt,
            mode,
            seed,
            duration_seconds,
            staged_upload if image_received else None,
            image_content_type if image_received else None,
        )
        return web.json_response(job.as_dict(), status=201)
    except BaseException:
        if staged_upload.exists():
            staged_upload.unlink()
        raise


async def delete_job(request: web.Request) -> web.Response:
    await _manager(request).remove(request.match_info["job_id"])
    return web.json_response({"ok": True})


async def regenerate_job(request: web.Request) -> web.Response:
    job = _manager(request).regenerate(request.match_info["job_id"])
    return web.json_response(job.as_dict(), status=201)


async def prompt_image(request: web.Request) -> web.FileResponse:
    job = _manager(request).jobs.get(request.match_info["job_id"])
    if (
        job is None
        or job.mode != "image"
        or job.upload_path is None
        or not job.upload_path.is_file()
    ):
        raise web.HTTPNotFound()
    headers = {
        "Cache-Control": "private, max-age=86400",
        "Content-Security-Policy": "default-src 'none'; sandbox",
        "X-Content-Type-Options": "nosniff",
    }
    if job.image_content_type:
        headers["Content-Type"] = job.image_content_type
    return web.FileResponse(job.upload_path, headers=headers)


async def media(request: web.Request) -> web.FileResponse:
    filename = request.match_info["filename"]
    if Path(filename).name != filename or not filename.endswith(".mp4"):
        raise web.HTTPNotFound()
    path = _manager(request).output_directory / filename
    if not path.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(path, headers={"Cache-Control": "private, max-age=86400"})


def make_app(output_directory: Path, memory_limit: float = 56.0) -> web.Application:
    manager = QueueManager(output_directory.resolve(), memory_limit)
    app = web.Application(client_max_size=MAX_UPLOAD_BYTES)
    app[MANAGER_KEY] = manager

    async def startup(_: web.Application) -> None:
        manager.start()

    async def cleanup(_: web.Application) -> None:
        await manager.close()

    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)
    app.router.add_get("/", index)
    app.router.add_get("/api/jobs", get_jobs)
    app.router.add_post("/api/jobs", create_job)
    app.router.add_post("/api/jobs/{job_id}/regenerate", regenerate_job)
    app.router.add_delete("/api/jobs/{job_id}", delete_job)
    app.router.add_get("/prompt-images/{job_id}", prompt_image)
    app.router.add_get("/media/{filename}", media)
    app.router.add_static("/assets", WEB_DIR, append_version=True)
    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the genvideo queue website")
    parser.add_argument("--host", default="10.9.0.8")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--memory-limit-gib", type=float, default=56.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    app = make_app(args.output_directory, args.memory_limit_gib)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
