"""Single-user mobile web queue for MiniMax H3 audio-video generation."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import sys
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aiohttp import web

from generation import (
    DEFAULT_ASPECT_RATIO,
    DEFAULT_MODEL,
    DEFAULT_DURATION_SECONDS,
    DEFAULT_RESOLUTION,
    MODEL_LABELS,
    SUPPORTED_ASPECT_RATIOS,
    SUPPORTED_MODELS,
    SUPPORTED_DURATION_SECONDS,
    SUPPORTED_RESOLUTIONS,
    GenerationError,
    ROOT,
    build_h3_prompt,
    canvas_dimensions,
    _random_seed,
)


WEB_DIR = ROOT / "web"
DEFAULT_OUTPUT_DIR = ROOT / "web_outputs"
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
STATE_FILENAME = "queue-state.json"
PRIORITIES = {"high": 0, "medium": 1, "low": 2}
LEGACY_MODEL_LABELS = {"ltx-2.5": "LTX-2.5 (legacy)"}


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(model, LEGACY_MODEL_LABELS.get(model, model))


@dataclass(slots=True)
class Job:
    id: str
    prompt: str
    mode: str
    seed: int
    duration_seconds: int
    model: str
    resolution: int
    aspect_ratio: str
    output_path: Path
    structured_prompt: dict[str, str] = field(default_factory=dict)
    upload_path: Path | None = None
    image_content_type: str | None = None
    priority: str = "medium"
    queue_order: int = 0
    progress: dict[str, Any] = field(default_factory=dict)
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
            "seed_text": str(self.seed),
            "duration_seconds": self.duration_seconds,
            "model": self.model,
            "resolution": self.resolution,
            "aspect_ratio": self.aspect_ratio,
            "structured_prompt": self.structured_prompt,
            "output_filename": self.output_path.name,
            "upload_filename": (
                self.upload_path.name if self.upload_path is not None else None
            ),
            "image_content_type": self.image_content_type,
            "priority": self.priority,
            "queue_order": self.queue_order,
            "progress": self.progress,
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    def as_dict(self, queue_position: int | None = None) -> dict[str, Any]:
        canvas_width, canvas_height = canvas_dimensions(
            self.resolution, self.aspect_ratio
        )
        return {
            "id": self.id,
            "prompt": self.prompt,
            "mode": self.mode,
            "seed": self.seed,
            "seed_text": str(self.seed),
            "duration_seconds": self.duration_seconds,
            "model": self.model,
            "model_label": _model_label(self.model),
            "resolution": self.resolution,
            "aspect_ratio": self.aspect_ratio,
            "canvas_width": canvas_width,
            "canvas_height": canvas_height,
            "structured_prompt": self.structured_prompt,
            "priority": self.priority,
            "queue_order": self.queue_order,
            "progress": self.progress,
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
        self.work_directory = output_directory / ".jobs"
        self.wake = asyncio.Event()
        self.process: asyncio.subprocess.Process | None = None
        self.stopping_process: asyncio.subprocess.Process | None = None
        self.stop_task: asyncio.Task[int] | None = None
        self.active_job: Job | None = None
        self.active_stopped = asyncio.Event()
        self.active_stopped.set()
        self.worker_task: asyncio.Task[None] | None = None
        self.engine_state = "idle"
        self.engine_model: str | None = None

    def start(self) -> None:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.upload_directory.mkdir(parents=True, exist_ok=True)
        self.work_directory.mkdir(parents=True, exist_ok=True)
        self._load_state()
        self.worker_task = asyncio.create_task(self._worker(), name="video-queue-worker")

    async def close(self) -> None:
        if self.worker_task is not None:
            self.worker_task.cancel()
        await self._stop_process()
        if self.worker_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self.worker_task
        self.engine_state = "idle"
        self.engine_model = None
        self._save_state()

    def _load_state(self) -> None:
        if not self.state_path.is_file():
            return
        state = json.loads(self.state_path.read_text())
        if not isinstance(state, dict) or state.get("version") not in {1, 2}:
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
            # Queue records from before per-job model selection were LTX jobs.
            model = str(record.get("model", "ltx-2.5"))
            if model not in {*SUPPORTED_MODELS, *LEGACY_MODEL_LABELS}:
                raise ValueError(f"invalid model in {self.state_path}: {model}")
            resolution = int(record.get("resolution", DEFAULT_RESOLUTION))
            aspect_ratio = str(record.get("aspect_ratio", DEFAULT_ASPECT_RATIO))
            if resolution not in SUPPORTED_RESOLUTIONS:
                raise ValueError(f"invalid resolution in {self.state_path}: {resolution}")
            if aspect_ratio not in SUPPORTED_ASPECT_RATIOS:
                raise ValueError(
                    f"invalid aspect ratio in {self.state_path}: {aspect_ratio}"
                )
            structured_prompt = record.get("structured_prompt", {})
            if not isinstance(structured_prompt, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in structured_prompt.items()
            ):
                raise ValueError(f"invalid structured prompt in {self.state_path}")
            job = Job(
                id=str(record["id"]),
                prompt=str(record["prompt"]),
                mode=str(record["mode"]),
                seed=int(record["seed"]),
                duration_seconds=int(
                    record.get("duration_seconds", DEFAULT_DURATION_SECONDS)
                ),
                model=model,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                structured_prompt=structured_prompt,
                output_path=self.output_directory / output_filename,
                upload_path=(
                    self.upload_directory / upload_filename
                    if upload_filename is not None
                    else None
                ),
                image_content_type=record.get("image_content_type"),
                status=str(record["status"]),
                priority=str(record.get("priority", "medium")),
                queue_order=int(record.get("queue_order", len(self.jobs))),
                progress=record.get("progress", {}),
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
            if job.priority not in PRIORITIES:
                raise ValueError(f"invalid priority in {self.state_path}")
            if Path(job.id).name != job.id or job.id in {".", ".."}:
                raise ValueError(f"invalid job id in {self.state_path}")
            if job.status == "cancelled":
                continue
            if job.status in {"queued", "running", "paused"}:
                if job.model not in SUPPORTED_MODELS:
                    job.status = "failed"
                    job.error = "LTX-2.5 support has been removed."
                    job.finished_at = time.time()
                elif job.mode == "image" and (
                    job.upload_path is None or not job.upload_path.is_file()
                ):
                    job.status = "failed"
                    job.error = (
                        "The starting image was lost before this job could resume."
                    )
                    job.finished_at = time.time()
                else:
                    if job.status != "paused":
                        job.status = "queued"
                    job.finished_at = None
            self.jobs[job.id] = job
        self._save_state()

    def _save_state(self) -> None:
        state = {
            "version": 2,
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
        structured_prompt: dict[str, str],
        mode: str,
        seed: int,
        duration_seconds: int,
        model: str,
        resolution: int,
        aspect_ratio: str,
        upload_path: Path | None,
        image_content_type: str | None = None,
        priority: str = "medium",
    ) -> Job:
        if priority not in PRIORITIES:
            raise ValueError(f"unsupported priority: {priority}")
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"unsupported model: {model}")
        if duration_seconds not in SUPPORTED_DURATION_SECONDS:
            raise ValueError(f"unsupported duration: {duration_seconds}")
        canvas_dimensions(resolution, aspect_ratio)
        job_id = uuid.uuid4().hex[:12]
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        job = Job(
            id=job_id,
            priority=priority,
            queue_order=self._next_order(),
            prompt=prompt,
            mode=mode,
            seed=seed,
            duration_seconds=duration_seconds,
            model=model,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            structured_prompt=structured_prompt,
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
        self.wake.set()
        return job

    def regenerate(self, job_id: str) -> Job:
        original = self.jobs.get(job_id)
        if original is None:
            raise web.HTTPNotFound(text="job not found")
        if original.model not in SUPPORTED_MODELS:
            raise web.HTTPConflict(text="this legacy LTX-2.5 job cannot be regenerated")

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
                original.structured_prompt.copy(),
                original.mode,
                _random_seed(),
                original.duration_seconds,
                original.model,
                original.resolution,
                original.aspect_ratio,
                copied_upload,
                original.image_content_type,
                original.priority,
            )
        except BaseException:
            if copied_upload is not None:
                copied_upload.unlink(missing_ok=True)
            raise

    def _next_order(self) -> int:
        return max((job.queue_order for job in self.jobs.values()), default=-1) + 1

    def _queued(self) -> list[Job]:
        return sorted((job for job in self.jobs.values() if job.status == "queued"),
                      key=lambda job: (PRIORITIES[job.priority], job.queue_order, job.created_at))

    def _job(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise web.HTTPNotFound(text="job not found")
        return job

    async def _stop_process(self) -> None:
        process = self.process
        if process is not None:
            # ComfyUI and ffmpeg inherit the worker's group. Kill the whole group,
            # even if the wrapper has exited but a descendant is still alive.
            if self.stopping_process is not process:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                self.stopping_process = process
                self.stop_task = asyncio.create_task(process.wait())
            await asyncio.shield(self.stop_task)

    async def pause(self, job_id: str) -> Job:
        job = self._job(job_id)
        if job.status not in {"queued", "running", "paused"}:
            raise web.HTTPConflict(text="only pending jobs can be paused")
        job.status = "paused"
        self._save_state()
        if self.active_job is job:
            await self._stop_process()
            await self.active_stopped.wait()
        self.wake.set()
        return job

    def resume(self, job_id: str) -> Job:
        job = self._job(job_id)
        if job.status != "paused":
            raise web.HTTPConflict(text="only paused jobs can be resumed")
        job.status = "queued"
        job.queue_order = self._next_order()
        self._save_state()
        self.wake.set()
        return job

    def edit(self, job_id: str, changes: dict[str, Any]) -> Job:
        job = self._job(job_id)
        if job.status not in {"queued", "running", "paused"}:
            raise web.HTTPConflict(text="only pending jobs can be reordered")
        if not changes or set(changes) - {"priority", "move"}:
            raise web.HTTPBadRequest(text="provide priority and/or move")
        priority = changes.get("priority", job.priority)
        move = changes.get("move")
        if not isinstance(priority, str) or priority not in PRIORITIES:
            raise web.HTTPBadRequest(text="priority must be low, medium, or high")
        if move is not None and move not in ("up", "down"):
            raise web.HTTPBadRequest(text="move must be up or down")
        if move is not None and job.status != "queued":
            raise web.HTTPConflict(text="only queued jobs can be moved")
        if priority != job.priority:
            job.priority = priority
            job.queue_order = self._next_order()
        if move:
            peers = [peer for peer in self._queued() if peer.priority == job.priority]
            index = peers.index(job)
            target = index + (-1 if move == "up" else 1)
            if 0 <= target < len(peers):
                peers[index], peers[target] = peers[target], peers[index]
                # Normalize old records as well, which can contain equal orders.
                for order, peer in enumerate(peers):
                    peer.queue_order = order
        self._save_state()
        self.wake.set()
        return job

    async def remove(self, job_id: str) -> None:
        job = self._job(job_id)
        job.status = "cancelled"
        self.jobs.pop(job_id)
        self._save_state()
        if self.active_job is job:
            await self._stop_process()
            await self.active_stopped.wait()
        if job.upload_path is not None:
            job.upload_path.unlink(missing_ok=True)
        job.output_path.unlink(missing_ok=True)
        for staging in self.output_directory.glob(f".{job.output_path.name}.*.tmp"):
            staging.unlink(missing_ok=True)
        await asyncio.to_thread(shutil.rmtree, self.work_directory / job.id, True)
        self.wake.set()

    def snapshot(self) -> dict[str, Any]:
        queued = self._queued()
        positions = {job.id: index + 1 for index, job in enumerate(queued)}
        running = [job for job in self.jobs.values() if job.status == "running"]
        paused = sorted((job for job in self.jobs.values() if job.status == "paused"),
                        key=lambda job: (PRIORITIES[job.priority], job.queue_order))
        history = sorted((job for job in self.jobs.values()
                          if job.status in {"completed", "failed"}),
                         key=lambda job: -(job.finished_at or job.created_at))
        return {
            "engine_state": self.engine_state,
            "engine_model_label": _model_label(self.engine_model) if self.engine_model else None,
            "queued_count": len(queued),
            "jobs": [job.as_dict(positions.get(job.id))
                     for job in running + queued + paused + history],
        }

    async def _run_job(self, job: Job) -> None:
        directory = self.work_directory / job.id
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("result.json", "progress.json"):
            (directory / name).unlink(missing_ok=True)
        spec = job.as_record() | {
            "output_path": str(job.output_path),
            "upload_path": str(job.upload_path) if job.upload_path else None,
            "memory_limit": self.memory_limit,
        }
        (directory / "request.json").write_text(json.dumps(spec))
        self.active_job = job
        self.active_stopped.clear()
        job.status = "running"
        job.started_at = job.started_at or time.time()
        job.progress = {"stage": "Starting engine", "step": None, "total": None}
        self.engine_state, self.engine_model = "starting", job.model
        self._save_state()
        try:
            with (directory / "worker.log").open("wb") as log:
                # Shield creation so shutdown cannot orphan a newly spawned child.
                spawn = asyncio.create_task(asyncio.create_subprocess_exec(
                    sys.executable, str(ROOT / "generation_worker.py"), str(directory), str(os.getpid()),
                    stdout=log, stderr=log, start_new_session=True))
                try:
                    self.process = await asyncio.shield(spawn)
                except asyncio.CancelledError:
                    self.process = await spawn
                    raise
            while True:
                progress_path = directory / "progress.json"
                if progress_path.is_file():
                    progress = json.loads(progress_path.read_text())
                    if job.progress != progress:
                        job.progress = progress
                        self.engine_state = "working"
                        self._save_state()
                if job.status != "running":
                    break
                result_path = directory / "result.json"
                if result_path.is_file():
                    result = json.loads(result_path.read_text())
                    if not result.get("ok"):
                        raise GenerationError(result.get("error", "Generation failed"))
                    if not job.output_path.is_file():
                        raise GenerationError("Worker completed without an output video")
                    job.status = "completed"
                    job.progress = {"stage": "Complete", "step": None, "total": None}
                    break
                if self.process.returncode is not None:
                    tail = (directory / "worker.log").read_text(errors="replace")[-3000:]
                    raise GenerationError(f"Generation worker exited ({self.process.returncode})\n{tail}")
                pending = self._queued()
                if pending and PRIORITIES[pending[0].priority] < PRIORITIES[job.priority]:
                    job.status = "queued"
                    # Preemption preserves its position within its own priority.
                    break
                await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            if job.status == "running":
                job.status = "queued"
            raise
        except Exception as exc:
            if job.status == "running":
                job.status = "failed"
                job.error = str(exc)[-3000:]
        finally:
            await self._stop_process()
            self.process = None
            self.engine_state, self.engine_model = "idle", None
            if job.status in {"completed", "failed"}:
                job.finished_at = time.time()
            self._save_state()
            if job.status in {"completed", "cancelled"}:
                await asyncio.to_thread(shutil.rmtree, directory, True)
            else:
                # All resumable data is outside runtime; discard partial decodes.
                await asyncio.to_thread(shutil.rmtree, directory / "runtime", True)
            self.active_job = None
            self.active_stopped.set()

    async def _worker(self) -> None:
        while True:
            self.wake.clear()
            queued = self._queued()
            if not queued:
                await self.wake.wait()
                continue
            await self._run_job(queued[0])


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
    text_fields: dict[str, str] = {}
    accepted_fields = {
        "prompt",
        "integrated_multimodal_description",
        "overall_soundscape",
        "non_diegetic_music",
        "mode",
        "model",
        "seed",
        "duration",
        "resolution",
        "aspect_ratio",
        "priority",
    }
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
                elif part.name in accepted_fields:
                    text_fields[part.name] = (await part.text()).strip()
        else:
            fields = await request.post()
            text_fields = {
                name: str(fields[name]).strip()
                for name in accepted_fields
                if name in fields
            }

        integrated_description = text_fields.get(
            "integrated_multimodal_description",
            text_fields.get("prompt", ""),
        )
        overall_soundscape = text_fields.get("overall_soundscape", "")
        non_diegetic_music = text_fields.get("non_diegetic_music", "")
        mode = text_fields.get("mode", "text")
        model = text_fields.get("model", DEFAULT_MODEL)
        seed_text = text_fields.get("seed", "")
        duration_text = text_fields.get("duration", str(DEFAULT_DURATION_SECONDS))
        resolution_text = text_fields.get("resolution", str(DEFAULT_RESOLUTION))
        aspect_ratio = text_fields.get("aspect_ratio", DEFAULT_ASPECT_RATIO)
        priority = text_fields.get("priority", "medium")
        if priority not in PRIORITIES:
            raise web.HTTPBadRequest(text="priority must be low, medium, or high")

        if mode not in {"text", "image"}:
            raise web.HTTPBadRequest(text="mode must be text or image")
        if model not in SUPPORTED_MODELS:
            raise web.HTTPBadRequest(text="unsupported model")
        try:
            prompt = build_h3_prompt(
                integrated_description,
                overall_soundscape,
                non_diegetic_music,
                image_mode=mode == "image",
            )
        except GenerationError as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
        if len(prompt) > 8000:
            raise web.HTTPBadRequest(
                text="combined H3 prompt must be 8,000 characters or fewer"
            )
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
            choices = ", ".join(str(value) for value in SUPPORTED_DURATION_SECONDS)
            raise web.HTTPBadRequest(text=f"duration must be {choices} seconds")
        try:
            resolution = int(resolution_text)
        except ValueError as exc:
            raise web.HTTPBadRequest(text="resolution must be an integer") from exc
        try:
            canvas_dimensions(resolution, aspect_ratio)
        except GenerationError as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc

        structured_prompt = {
            "integrated_multimodal_description": integrated_description,
            "overall_soundscape": overall_soundscape,
            "non_diegetic_music": non_diegetic_music,
        }

        job = manager.add(
            prompt,
            structured_prompt,
            mode,
            seed,
            duration_seconds,
            model,
            resolution,
            aspect_ratio,
            staged_upload if image_received else None,
            image_content_type if image_received else None,
            priority,
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


async def pause_job(request: web.Request) -> web.Response:
    job = await _manager(request).pause(request.match_info["job_id"])
    return web.json_response(job.as_dict())


async def resume_job(request: web.Request) -> web.Response:
    job = _manager(request).resume(request.match_info["job_id"])
    return web.json_response(job.as_dict())


async def edit_job(request: web.Request) -> web.Response:
    try:
        changes = await request.json()
    except (ValueError, TypeError) as exc:
        raise web.HTTPBadRequest(text="expected a JSON object") from exc
    if not isinstance(changes, dict):
        raise web.HTTPBadRequest(text="expected a JSON object")
    job = _manager(request).edit(request.match_info["job_id"], changes)
    return web.json_response(job.as_dict())


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
    app.router.add_post("/api/jobs/{job_id}/pause", pause_job)
    app.router.add_post("/api/jobs/{job_id}/resume", resume_job)
    app.router.add_patch("/api/jobs/{job_id}", edit_job)
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
