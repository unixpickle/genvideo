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
from ref2va import (REF2VA_MODEL, REF_FIELDS, MAX_REFERENCE_BYTES, MAX_REFERENCES,
                    validate_references, probe_reference, validate_durations,
                    reference_labels, build_ref2va_prompt)

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
    last_upload_path: Path | None = None
    last_image_content_type: str | None = None
    references: list[dict] = field(default_factory=list)
    reference_builder: dict = field(default_factory=dict)
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
            "references": self.references,
            "reference_builder": self.reference_builder,
            "output_filename": self.output_path.name,
            "upload_filename": (
                self.upload_path.name if self.upload_path is not None else None
            ),
            "image_content_type": self.image_content_type,
            "last_upload_filename": self.last_upload_path.name if self.last_upload_path else None,
            "last_image_content_type": self.last_image_content_type,
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
            "reference_builder": self.reference_builder,
            "references": [dict(ref, url=f"/reference-media/{self.id}/{ref['id']}",
                                labels=reference_labels(self.references)[ref["id"]])
                           for ref in self.references],
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
            "has_first_frame": self.upload_path is not None,
            "has_last_frame": self.last_upload_path is not None,
            "last_image_prompt_url": (
                f"/prompt-images/{self.id}/last"
                if self.mode == "image" and self.last_upload_path is not None
                and self.last_upload_path.is_file() else None
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
            last_upload_filename = record.get("last_upload_filename")
            if Path(output_filename).name != output_filename:
                raise ValueError(f"invalid output filename in {self.state_path}")
            if upload_filename is not None:
                upload_filename = str(upload_filename)
                if Path(upload_filename).name != upload_filename:
                    raise ValueError(f"invalid upload filename in {self.state_path}")
            if last_upload_filename is not None:
                last_upload_filename = str(last_upload_filename)
                if Path(last_upload_filename).name != last_upload_filename:
                    raise ValueError(f"invalid last upload filename in {self.state_path}")
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
            references = record.get("references", [])
            for ref in references:
                filename = ref.get("filename", "")
                if not filename or Path(filename).name != filename or filename in {".", ".."}:
                    raise ValueError("invalid reference filename")
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
                references=references,
                reference_builder=record.get("reference_builder", {}),
                output_path=self.output_directory / output_filename,
                upload_path=(
                    self.upload_directory / upload_filename
                    if upload_filename is not None
                    else None
                ),
                image_content_type=record.get("image_content_type"),
                last_upload_path=(self.upload_directory / last_upload_filename
                                  if last_upload_filename is not None else None),
                last_image_content_type=record.get("last_image_content_type"),
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
                    not any((job.upload_path, job.last_upload_path))
                    or any(path is not None and not path.is_file()
                           for path in (job.upload_path, job.last_upload_path))
                ):
                    job.status = "failed"
                    job.error = (
                        "A conditioning image was lost before this job could resume."
                    )
                    job.finished_at = time.time()
                elif job.mode == "ref2va" and not self._references_available(job):
                    job.status = "failed"
                    job.error = "Reference media was lost before this job could resume."
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
        last_upload_path: Path | None = None,
        last_image_content_type: str | None = None,
        references: list[dict] | None = None,
        reference_builder: dict | None = None,
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
            references=references or [],
            reference_builder=reference_builder or {},
            upload_path=upload_path,
            image_content_type=image_content_type,
            last_upload_path=last_upload_path,
            last_image_content_type=last_image_content_type,
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

        copied_uploads: list[Path | None] = [None, None]
        references = []
        try:
            if original.mode == "image":
                paths = (original.upload_path, original.last_upload_path)
                if not any(paths) or any(path is not None and not path.is_file() for path in paths):
                    raise web.HTTPConflict(
                        text="a conditioning image for this job is no longer available"
                    )
                for index, path in enumerate(paths):
                    if path is not None:
                        copied_uploads[index] = self.upload_directory / f"{uuid.uuid4().hex}.upload"
                        shutil.copyfile(path, copied_uploads[index])
            if original.mode == "ref2va" and not self._references_available(original):
                raise web.HTTPConflict(text="reference media is no longer available")
            for ref in original.references:
                destination = self.upload_directory / f"{uuid.uuid4().hex}.upload"
                copied_uploads.append(destination)
                shutil.copyfile(self.upload_directory / ref["filename"], destination)
                references.append(dict(ref, filename=destination.name))
            return self.add(
                original.prompt,
                original.structured_prompt.copy(),
                original.mode,
                _random_seed(),
                original.duration_seconds,
                original.model,
                original.resolution,
                original.aspect_ratio,
                copied_uploads[0],
                original.image_content_type,
                original.priority,
                copied_uploads[1],
                original.last_image_content_type,
                references,
                original.reference_builder.copy(),
            )
        except BaseException:
            for path in copied_uploads:
                if path is not None:
                    path.unlink(missing_ok=True)
            raise

    def _references_available(self, job):
        return bool(job.references) and all((self.upload_directory / r["filename"]).is_file()
                                            for r in job.references)

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

    def retry(self, job_id: str) -> Job:
        job = self._job(job_id)
        if job.status != "failed":
            raise web.HTTPConflict(text="only failed jobs can be retried")
        if job.model not in SUPPORTED_MODELS:
            raise web.HTTPConflict(text="this legacy LTX-2.5 job cannot be retried")
        if job.mode == "image":
            paths = (job.upload_path, job.last_upload_path)
            if not any(paths) or any(path is not None and not path.is_file() for path in paths):
                raise web.HTTPConflict(
                    text="a conditioning image for this job is no longer available"
                )
        if job.mode == "ref2va" and not self._references_available(job):
            raise web.HTTPConflict(text="reference media is no longer available")
        job.status = "queued"
        job.queue_order = self._next_order()
        job.error = None
        job.progress = {}
        job.started_at = None
        job.finished_at = None
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
        for path in (job.upload_path, job.last_upload_path):
            if path is not None:
                path.unlink(missing_ok=True)
        for ref in job.references:
            (self.upload_directory / ref["filename"]).unlink(missing_ok=True)
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
            "last_upload_path": str(job.last_upload_path) if job.last_upload_path else None,
            "memory_limit": self.memory_limit,
            "references": [dict(ref, path=str(self.upload_directory / ref["filename"]))
                           for ref in job.references],
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
    staged_uploads = {name: manager.upload_directory / f"{job_token}-{name}.upload"
                      for name in ("image", "last_image")}
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
        "priority", "references", "reference_builder", *REF_FIELDS,
    }
    received: dict[str, str] = {}
    try:
        if request.content_type.startswith("multipart/"):
            reader = await request.multipart()
            async for part in reader:
                if part.name and part.name.startswith("reference_") and part.filename:
                    if part.name in staged_uploads or sum(k.startswith("reference_") for k in staged_uploads) >= MAX_REFERENCES:
                        raise web.HTTPBadRequest(text="duplicate or too many reference uploads")
                    path = manager.upload_directory / f"{uuid.uuid4().hex}.upload"
                    staged_uploads[part.name] = path
                    size = 0
                    with path.open("wb") as output:
                        while chunk := await part.read_chunk(1024 * 1024):
                            size += len(chunk)
                            if size > MAX_REFERENCE_BYTES:
                                raise web.HTTPRequestEntityTooLarge(max_size=MAX_REFERENCE_BYTES, actual_size=size)
                            output.write(chunk)
                    if not size:
                        raise web.HTTPBadRequest(text="reference file is empty")
                    received[part.name] = part.headers.get("Content-Type", "application/octet-stream")
                elif part.name in staged_uploads and part.filename:
                    if part.name in received:
                        raise web.HTTPBadRequest(text="provide only one file per frame")
                    image_content_type = part.headers.get("Content-Type") or ""
                    if not image_content_type.startswith("image/"):
                        raise web.HTTPBadRequest(text="the uploaded file must be an image")
                    size = 0
                    with staged_uploads[part.name].open("wb") as upload_file:
                        while chunk := await part.read_chunk(1024 * 1024):
                            size += len(chunk)
                            if size > MAX_UPLOAD_BYTES:
                                raise web.HTTPRequestEntityTooLarge(
                                    max_size=MAX_UPLOAD_BYTES, actual_size=size
                                )
                            upload_file.write(chunk)
                    if size:
                        received[part.name] = image_content_type
                    else:
                        staged_uploads[part.name].unlink(missing_ok=True)
                elif part.name in accepted_fields:
                    raw = await part.read()
                    if len(raw) > 100_000:
                        raise web.HTTPBadRequest(text="form field is too large")
                    text_fields[part.name] = raw.decode("utf-8").strip()
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

        if mode not in {"text", "image", "ref2va"}:
            raise web.HTTPBadRequest(text="mode must be text, image, or ref2va")
        if (mode == "ref2va") != (model == REF2VA_MODEL):
            raise web.HTTPBadRequest(text="Ref2VA mode requires a Ref2VA model, and vice versa")
        if model not in SUPPORTED_MODELS:
            raise web.HTTPBadRequest(text="unsupported model")
        if mode != "ref2va" and any(k.startswith("reference_") for k in received):
            raise web.HTTPBadRequest(text="reference files require Ref2VA mode")
        if mode == "image" and not received:
            raise web.HTTPBadRequest(text="a start or end image is required in image mode")
        if mode == "text":
            for path in staged_uploads.values():
                path.unlink(missing_ok=True)
            received.clear()
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

        references, reference_builder = [], {}
        if mode == "ref2va":
            if "image" in received or "last_image" in received:
                raise web.HTTPBadRequest(text="Use reference attachments in Ref2VA mode")
            try:
                references = validate_references(json.loads(text_fields.get("references", "[]")))
                expected = {f"reference_{r['id']}" for r in references}
                if expected != set(received):
                    raise ValueError("Every reference must have exactly one matching upload.")
                for ref in references:
                    key = f"reference_{ref['id']}"
                    ref["filename"] = staged_uploads[key].name
                    ref["content_type"] = received[key]
                    await asyncio.to_thread(probe_reference, staged_uploads[key], ref)
                validate_durations(references)
                reference_builder = json.loads(text_fields.get("reference_builder", "{}"))
                if not isinstance(reference_builder, dict):
                    raise ValueError("Invalid prompt builder state")
                structured_prompt = {name: text_fields.get(name, "") for name in REF_FIELDS}
                prompt = build_ref2va_prompt(structured_prompt, references)
            except (ValueError, TypeError) as exc:
                raise web.HTTPBadRequest(text=str(exc)) from exc
        else:
            structured_prompt = {
                "integrated_multimodal_description": integrated_description,
                "overall_soundscape": overall_soundscape,
                "non_diegetic_music": non_diegetic_music,
            }
            try:
                prompt = build_h3_prompt(**structured_prompt, image_mode="image" in received,
                                         last_frame="last_image" in received, duration_seconds=duration_seconds)
            except GenerationError as exc:
                raise web.HTTPBadRequest(text=str(exc)) from exc
            if len(prompt) > 8000:
                raise web.HTTPBadRequest(text="combined H3 prompt must be 8,000 characters or fewer")

        job = manager.add(
            prompt,
            structured_prompt,
            mode,
            seed,
            duration_seconds,
            model,
            resolution,
            aspect_ratio,
            staged_uploads["image"] if "image" in received else None,
            received.get("image"),
            priority,
            staged_uploads["last_image"] if "last_image" in received else None,
            received.get("last_image"),
            references,
            reference_builder,
        )
        return web.json_response(job.as_dict(), status=201)
    except BaseException:
        for path in staged_uploads.values():
            path.unlink(missing_ok=True)
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


async def retry_job(request: web.Request) -> web.Response:
    job = _manager(request).retry(request.match_info["job_id"])
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
    if job is None or job.mode != "image":
        raise web.HTTPNotFound()
    last = request.match_info.get("frame") == "last"
    path = job.last_upload_path if last else job.upload_path
    content_type = job.last_image_content_type if last else job.image_content_type
    if path is None or not path.is_file():
        raise web.HTTPNotFound()
    headers = {
        "Cache-Control": "private, max-age=86400",
        "Content-Security-Policy": "default-src 'none'; sandbox",
        "X-Content-Type-Options": "nosniff",
    }
    if content_type:
        headers["Content-Type"] = content_type
    return web.FileResponse(path, headers=headers)


async def reference_media(request: web.Request) -> web.FileResponse:
    manager = _manager(request)
    job = manager.jobs.get(request.match_info["job_id"])
    ref = next((r for r in job.references if r["id"] == request.match_info["ref_id"]), None) if job else None
    if ref is None:
        raise web.HTTPNotFound()
    path = manager.upload_directory / ref["filename"]
    if not path.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(path, headers={"Content-Type": ref["content_type"],
        "Content-Security-Policy": "default-src 'none'; sandbox", "X-Content-Type-Options": "nosniff"})


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
    app = web.Application(client_max_size=MAX_REFERENCES * MAX_REFERENCE_BYTES + 1024 * 1024)
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
    app.router.add_post("/api/jobs/{job_id}/retry", retry_job)
    app.router.add_patch("/api/jobs/{job_id}", edit_job)
    app.router.add_get("/prompt-images/{job_id}", prompt_image)
    app.router.add_get("/prompt-images/{job_id}/{frame:last}", prompt_image)
    app.router.add_get("/reference-media/{job_id}/{ref_id}", reference_media)
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
