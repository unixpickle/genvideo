"""Memory-bounded MiniMax H3 generation engine for the web app."""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable

import aiohttp
import psutil
from ref2va import REF2VA_MODEL, REF2VA_WEIGHTS


ROOT = Path(__file__).resolve().parent
COMFY_DIR = ROOT / "ComfyUI"
COMFY_PYTHON = ROOT / ".venv" / "bin" / "python"
DEFAULT_MODEL = "minimax-h3"
SUPPORTED_MODELS = ("minimax-h3", "minimax-h3-larry-v4", "minimax-h3-base", REF2VA_MODEL)
MODEL_LABELS = {
    REF2VA_MODEL: "MiniMax H3 Ref2VA Q4 (20 steps)",
    "minimax-h3": "MiniMax H3 Turbo",
    "minimax-h3-larry-v4": "MiniMax H3 Turbo v4 (larryvrh, 6 steps)",
    "minimax-h3-base": "MiniMax H3 Regular",
}
WORKFLOW_PATHS = {
    REF2VA_MODEL: ROOT / "workflows" / "minimax_h3_video_api.json",
    "minimax-h3": ROOT / "workflows" / "minimax_h3_video_api.json",
    "minimax-h3-larry-v4": ROOT / "workflows" / "minimax_h3_video_api.json",
    "minimax-h3-base": ROOT / "workflows" / "minimax_h3_video_api.json",
}
HARD_MEMORY_CEILING_GIB = 64.0
MAX_SWAP_GROWTH_GIB = 4.0
MIN_SYSTEM_AVAILABLE_GIB = 2.0
DEFAULT_DURATION_SECONDS = 5
SUPPORTED_DURATION_SECONDS = tuple(range(3, 16))
DEFAULT_RESOLUTION = 512
SUPPORTED_RESOLUTIONS = (512, 768)
DEFAULT_ASPECT_RATIO = "1:1"
SUPPORTED_ASPECT_RATIOS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
CANVAS_MULTIPLE = 32
MAX_CANVAS_ASPECT = 7 / 4
PROGRESS_NODE_CLASSES = {
    "CLIPLoader",
    "CreateVideo",
    "LoraLoaderModelOnly",
    "MiniMaxH3TurboLoRA",
    "GenVideoMiniMaxH3Conditioning",
    "GenVideoMiniMaxH3ReferenceConditioning",
    "GenVideoLoadConditioning",
    "GenVideoLoadLatent",
    "GenVideoResumableSampler",
    "SamplerCustomAdvanced",
    "SaveVideo",
    "UnetLoaderGGUF",
    "VAEDecode",
    "VAEDecodeAudio",
    "VAELoader",
}


class GenerationError(RuntimeError):
    pass


def canvas_dimensions(resolution: int, aspect_ratio: str) -> tuple[int, int]:
    """Return an H3 canvas aligned to 32 pixels and its local area cap."""
    if resolution not in SUPPORTED_RESOLUTIONS:
        raise GenerationError(f"unsupported resolution: {resolution}p")
    if aspect_ratio not in SUPPORTED_ASPECT_RATIOS:
        raise GenerationError(f"unsupported aspect ratio: {aspect_ratio}")
    ratio_width, ratio_height = (int(value) for value in aspect_ratio.split(":"))
    ratio = ratio_width / ratio_height
    if ratio >= 1:
        nominal_width, nominal_height = resolution * ratio, float(resolution)
    else:
        nominal_width, nominal_height = float(resolution), resolution / ratio
    max_pixels = resolution * resolution * MAX_CANVAS_ASPECT
    if nominal_width * nominal_height > max_pixels:
        scale = math.sqrt(max_pixels / (nominal_width * nominal_height))
        nominal_width *= scale
        nominal_height *= scale
    width = max(
        CANVAS_MULTIPLE,
        round(nominal_width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
    )
    height = max(
        CANVAS_MULTIPLE,
        round(nominal_height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
    )
    return width, height


def build_h3_prompt(
    integrated_multimodal_description: str = "",
    overall_soundscape: str = "",
    non_diegetic_music: str = "",
    *,
    image_mode: bool = False,
    last_frame: bool = False,
    duration_seconds: int = DEFAULT_DURATION_SECONDS,
) -> str:
    """Serialize the optional H3 base prompt fields in their official order."""
    fields = (
        ("integrated_multimodal_description", integrated_multimodal_description),
        ("overall_soundscape", overall_soundscape),
        ("non_diegetic_music", non_diegetic_music),
    )
    sections = [f"{name}: {value.strip()}" for name, value in fields if value.strip()]
    if not sections:
        raise GenerationError("at least one H3 prompt field is required")
    if image_mode:
        sections.insert(
            0,
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced.",
        )
    if last_frame:
        frames = max(5, round(duration_seconds * 24))
        frames += (5 - frames % 17) % 17
        picture = 2 if image_mode else 1
        sections.insert(1 if image_mode else 0,
            f"For the target video, at {(frames - 1) / 24:.2f} seconds into the target video, "
            f"<Picture {picture}> is fully referenced as the last frame.")
    return "\n\n".join(sections)


def _prepare_image(source: Path, destination: Path, width: int, height: int) -> None:
    """Scale to cover, center-crop, and discard alpha in one ffmpeg pass."""
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vf",
        (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1,format=rgb24"
        ),
        "-frames:v",
        "1",
        str(destination),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise GenerationError(f"could not preprocess image: {source}") from exc


def _text_only_graph(workflow: dict[str, Any]) -> dict[str, Any]:
    """Bypass image conditioning and retain only ancestors of output nodes."""
    conditioners = [
        node
        for node in workflow.values()
        if node["class_type"] == "GenVideoMiniMaxH3Conditioning"
    ]
    if len(conditioners) != 1:
        raise GenerationError("workflow has no H3 image-conditioning stage to bypass")
    conditioners[0]["inputs"].pop("first_frame", None)
    conditioners[0]["inputs"].pop("last_frame", None)

    roots = [
        node_id
        for node_id, node in workflow.items()
        if node["class_type"] == "SaveVideo"
    ]
    reachable: set[str] = set()
    pending = roots[:]
    while pending:
        node_id = pending.pop()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        for value in workflow[node_id]["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and value[0] in workflow:
                pending.append(value[0])
    return {node_id: node for node_id, node in workflow.items() if node_id in reachable}


def _workflow(
    image_name: str | None,
    prompt: str,
    seed: int,
    duration_seconds: int = DEFAULT_DURATION_SECONDS,
    filename_prefix: str = "genvideo",
    model: str = DEFAULT_MODEL,
    width: int = 512,
    height: int = 512,
    last_image_name: str | None = None,
    references: list[dict] | None = None,
) -> dict[str, Any]:
    if model not in SUPPORTED_MODELS:
        raise GenerationError(f"unsupported model: {model}")
    with WORKFLOW_PATHS[model].open() as workflow_file:
        workflow: dict[str, Any] = json.load(workflow_file)

    if model in {"minimax-h3-base", "minimax-h3-larry-v4", REF2VA_MODEL}:
        turbo_loras = [
            (node_id, node)
            for node_id, node in workflow.items()
            if node["class_type"] == "LoraLoaderModelOnly"
            and "turbo" in node["inputs"].get("lora_name", "").lower()
        ]
        schedulers = [
            node
            for node in workflow.values()
            if node["class_type"] == "BasicScheduler"
        ]
        if len(turbo_loras) != 1 or len(schedulers) != 1:
            raise GenerationError("MiniMax H3 workflow has an unexpected Turbo structure")
        turbo_id, turbo_lora = turbo_loras[0]
        if model in {"minimax-h3-base", REF2VA_MODEL}:
            base_model = turbo_lora["inputs"]["model"]
            for node in workflow.values():
                for name, value in node["inputs"].items():
                    if value == [turbo_id, 0]:
                        node["inputs"][name] = list(base_model)
            del workflow[turbo_id]
            schedulers[0]["inputs"]["steps"] = 20
        else:
            turbo_lora["class_type"] = "MiniMaxH3TurboLoRA"
            turbo_lora["inputs"] = {
                "model": turbo_lora["inputs"]["model"],
                "lora_name": "minimax_h3_turbo_v4_step600_ema.safetensors",
                "strength": 1.0,
                "low_vram": False,
            }
            turbo_lora["_meta"] = {"title": "Load larryvrh H3 Turbo v4 LoRA"}
            schedulers[0]["inputs"]["steps"] = 6
            for node in workflow.values():
                if node["class_type"] == "KSamplerSelect":
                    # Installed ComfyUI's ModelSamplingAV handles both clocks;
                    # the author's Turbo sampler reduces to Euler on this stack.
                    node["inputs"]["sampler_name"] = "euler"

    load_images = [node for node in workflow.values() if node["class_type"] == "LoadImage"]
    prompts = [
        node
        for node in workflow.values()
        if node["class_type"] == "PrimitiveStringMultiline"
    ]
    save_videos = [node for node in workflow.values() if node["class_type"] == "SaveVideo"]
    noise_nodes = [node for node in workflow.values() if node["class_type"] == "RandomNoise"]
    durations = [
        node
        for node in workflow.values()
        if node.get("_meta", {}).get("title") == "Duration"
    ]
    conditioners = [
        node
        for node in workflow.values()
        if node["class_type"] == "GenVideoMiniMaxH3Conditioning"
    ]
    if (
        len(load_images) != 1
        or len(prompts) != 1
        or len(save_videos) != 1
        or len(durations) != 1
        or len(conditioners) != 1
    ):
        raise GenerationError("workflow template has an unexpected structure")
    if duration_seconds not in SUPPORTED_DURATION_SECONDS:
        supported = ", ".join(str(value) for value in SUPPORTED_DURATION_SECONDS)
        raise GenerationError(f"duration must be one of: {supported} seconds")

    if image_name is not None:
        load_images[0]["inputs"]["image"] = image_name
    else:
        first_link = conditioners[0]["inputs"].pop("first_frame")
        del workflow[first_link[0]]
    if last_image_name is not None:
        workflow["last_image"] = {
            "class_type": "LoadImage",
            "inputs": {"image": last_image_name},
            "_meta": {"title": "Load Last Frame"},
        }
        conditioners[0]["inputs"]["last_frame"] = ["last_image", 0]
    prompts[0]["inputs"]["value"] = prompt
    durations[0]["inputs"]["value"] = duration_seconds
    conditioners[0]["inputs"]["width"] = width
    conditioners[0]["inputs"]["height"] = height
    save_videos[0]["inputs"]["filename_prefix"] = filename_prefix
    for offset, node in enumerate(noise_nodes):
        node["inputs"]["noise_seed"] = seed + offset
    if model == REF2VA_MODEL:
        if image_name or last_image_name:
            raise GenerationError("Ref2VA uses reference attachments, not start/end frame inputs")
        if not references:
            raise GenerationError("Ref2VA requires reference attachments")
        workflow["1"]["inputs"]["unet_name"] = REF2VA_WEIGHTS
        conditioners[0]["class_type"] = "GenVideoMiniMaxH3ReferenceConditioning"
        conditioners[0]["inputs"].update(audio_vae=["4", 0], references_json=json.dumps(references))
        conditioners[0]["_meta"] = {"title": "Encode Ref2VA references"}
        return workflow
    return workflow if image_name is not None or last_image_name is not None else _text_only_graph(workflow)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request_json(
    url: str, *, payload: dict[str, Any] | None = None, timeout: float = 10
) -> dict[str, Any]:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        details = exc.read().decode(errors="replace")
        raise GenerationError(f"ComfyUI rejected the request: {details}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise GenerationError(f"could not communicate with ComfyUI: {exc}") from exc


def _resumable_workflow(workflow: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Replace completed stages and prune their expensive model dependencies."""
    directory.mkdir(parents=True, exist_ok=True)
    for node in workflow.values():
        kind = node["class_type"]
        if kind in {"GenVideoMiniMaxH3Conditioning", "GenVideoMiniMaxH3ReferenceConditioning"}:
            if (directory / "conditioning.pt").is_file():
                node["class_type"] = "GenVideoLoadConditioning"
                node["inputs"] = {}
                node["_meta"] = {"title": "Restore saved conditioning"}
            node["inputs"]["checkpoint_directory"] = str(directory)
        elif kind == "SamplerCustomAdvanced":
            if (directory / "latent.pt").is_file():
                node["class_type"] = "GenVideoLoadLatent"
                node["inputs"] = {}
                node["_meta"] = {"title": "Restore generated latents"}
            else:
                node["class_type"] = "GenVideoResumableSampler"
                sampler_id, _ = node["inputs"].pop("sampler")
                node["inputs"]["sampler_name"] = workflow[sampler_id]["inputs"]["sampler_name"]
            node["inputs"]["checkpoint_directory"] = str(directory)
    reachable = set()
    pending = [key for key, node in workflow.items() if node["class_type"] == "SaveVideo"]
    while pending:
        key = pending.pop()
        if key in reachable:
            continue
        reachable.add(key)
        for value in workflow[key]["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and value[0] in workflow:
                pending.append(value[0])
    return {key: node for key, node in workflow.items() if key in reachable}


def _process_tree_rss(process: subprocess.Popen[bytes]) -> int:
    try:
        root = psutil.Process(process.pid)
        processes = [root, *root.children(recursive=True)]
    except psutil.Error:
        return 0
    total = 0
    for child in processes:
        try:
            total += child.memory_info().rss
        except psutil.Error:
            pass
    return total


def _check_process(process: subprocess.Popen[bytes], memory_limit: int) -> int:
    return_code = process.poll()
    if return_code is not None:
        raise GenerationError(f"ComfyUI exited unexpectedly with status {return_code}")
    rss = _process_tree_rss(process)
    if rss > memory_limit:
        raise GenerationError(
            f"memory limit exceeded ({rss / 1024**3:.1f} GiB > "
            f"{memory_limit / 1024**3:.1f} GiB); generation stopped"
        )
    return rss


def _wait_for_server(base_url: str, process: subprocess.Popen[bytes], memory_limit: int) -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        _check_process(process, memory_limit)
        try:
            _request_json(f"{base_url}/system_stats", timeout=1)
            return
        except (OSError, GenerationError):
            time.sleep(1)
    raise GenerationError("timed out waiting for ComfyUI to start")


def _error_from_history(record: dict[str, Any]) -> str | None:
    messages = record.get("status", {}).get("messages", [])
    for message in messages:
        if isinstance(message, list) and message and message[0] == "execution_error":
            details = message[1] if len(message) > 1 else "unknown execution error"
            if isinstance(details, dict):
                return str(details.get("exception_message") or details)
            return str(details)
    return None


def _find_video(record: dict[str, Any], output_directory: Path) -> Path | None:
    for node_output in record.get("outputs", {}).values():
        for key in ("videos", "gifs", "images"):
            for item in node_output.get(key, []):
                filename = item.get("filename")
                if not filename or Path(filename).name != filename:
                    continue
                subfolder = item.get("subfolder", "")
                candidate = output_directory / subfolder / filename
                if candidate.suffix.lower() == ".mp4" and candidate.is_file():
                    return candidate
    return None


def _elapsed_label(started_at: float) -> str:
    return _duration_label(time.monotonic() - started_at)


def _duration_label(duration: float) -> str:
    elapsed = max(0, int(duration))
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


async def _progress_listener(
    base_url: str,
    client_id: str,
    node_titles: dict[str, str],
    logged_node_ids: set[str],
    stop_event: threading.Event,
    ready_event: threading.Event,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    websocket_url = base_url.replace("http://", "ws://", 1) + f"/ws?clientId={client_id}"
    started_at = time.monotonic()
    node_started_at: dict[str, float] = {}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(websocket_url, max_msg_size=0) as websocket:
                ready_event.set()
                while not stop_event.is_set():
                    try:
                        message = await asyncio.wait_for(websocket.receive(), timeout=1)
                    except asyncio.TimeoutError:
                        continue
                    if message.type != aiohttp.WSMsgType.TEXT:
                        if message.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            return
                        continue
                    try:
                        event = json.loads(message.data)
                    except (TypeError, json.JSONDecodeError):
                        continue
                    event_type = event.get("type")
                    data = event.get("data", {})
                    node_id = data.get("node")
                    if event_type == "executing" and node_id:
                        node_id = str(node_id)
                        node_started_at[node_id] = time.monotonic()
                        if progress_callback:
                            progress_callback({"stage": node_titles.get(node_id, node_id), "step": None, "total": None})
                        if node_id in logged_node_ids:
                            title = node_titles.get(node_id, node_id)
                            print(
                                f"genvideo: [{_elapsed_label(started_at)}] "
                                f"{title} ({node_id})",
                                file=sys.stderr,
                                flush=True,
                            )
                    elif event_type == "progress":
                        value, maximum = data.get("value"), data.get("max")
                        if progress_callback:
                            progress_callback({"stage": node_titles.get(str(node_id), "Generating"), "step": value, "total": maximum})
                        if value is None or maximum is None:
                            continue
                        node_id = str(node_id)
                        title = node_titles.get(node_id, node_id)
                        eta = ""
                        node_start = node_started_at.get(node_id)
                        if (
                            node_start is not None
                            and isinstance(value, (int, float))
                            and isinstance(maximum, (int, float))
                            and value > 0
                            and maximum >= value
                        ):
                            step_time = (time.monotonic() - node_start) / value
                            eta = (
                                f", node ETA "
                                f"{_duration_label(step_time * (maximum - value))}"
                            )
                        print(
                            f"genvideo: [{_elapsed_label(started_at)}] "
                            f"{title} ({node_id}) {value}/{maximum}{eta}",
                            file=sys.stderr,
                            flush=True,
                        )
    except (aiohttp.ClientError, OSError) as exc:
        if not stop_event.is_set():
            print(f"genvideo: progress logging unavailable: {exc}", file=sys.stderr)
    finally:
        ready_event.set()


def _run_progress_listener(
    base_url: str,
    client_id: str,
    node_titles: dict[str, str],
    logged_node_ids: set[str],
    stop_event: threading.Event,
    ready_event: threading.Event,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    asyncio.run(
        _progress_listener(
            base_url,
            client_id,
            node_titles,
            logged_node_ids,
            stop_event,
            ready_event,
            progress_callback,
        )
    )


def _generate(
    process: subprocess.Popen[bytes],
    base_url: str,
    workflow: dict[str, Any],
    output_directory: Path,
    memory_limit: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    client_id = uuid.uuid4().hex
    node_titles = {
        node_id: node.get("_meta", {}).get("title", node.get("class_type", node_id))
        for node_id, node in workflow.items()
    }
    logged_node_ids = {
        node_id
        for node_id, node in workflow.items()
        if node.get("class_type") in PROGRESS_NODE_CLASSES
    }
    stop_progress = threading.Event()
    progress_ready = threading.Event()
    progress_thread = threading.Thread(
        target=_run_progress_listener,
        args=(
            base_url,
            client_id,
            node_titles,
            logged_node_ids,
            stop_progress,
            progress_ready,
            progress_callback,
        ),
        daemon=True,
    )
    progress_thread.start()
    progress_ready.wait(timeout=10)
    peak_rss = 0
    minimum_available = psutil.virtual_memory().available
    initial_swap = psutil.swap_memory().used
    peak_swap_growth = 0
    try:
        response = _request_json(
            f"{base_url}/prompt",
            payload={"prompt": workflow, "client_id": client_id},
            timeout=30,
        )
        prompt_id = response.get("prompt_id")
        if not prompt_id:
            raise GenerationError(f"ComfyUI did not return a prompt ID: {response}")

        while True:
            peak_rss = max(peak_rss, _check_process(process, memory_limit))
            minimum_available = min(
                minimum_available, psutil.virtual_memory().available
            )
            if minimum_available < MIN_SYSTEM_AVAILABLE_GIB * 1024**3:
                raise GenerationError(
                    f"system memory safety reserve breached "
                    f"({minimum_available / 1024**3:.1f} GiB available < "
                    f"{MIN_SYSTEM_AVAILABLE_GIB:g} GiB); generation stopped"
                )
            swap_growth = max(0, psutil.swap_memory().used - initial_swap)
            peak_swap_growth = max(peak_swap_growth, swap_growth)
            if swap_growth > MAX_SWAP_GROWTH_GIB * 1024**3:
                raise GenerationError(
                    f"swap safety limit exceeded ({swap_growth / 1024**3:.1f} GiB "
                    f"growth > {MAX_SWAP_GROWTH_GIB:g} GiB); generation stopped"
                )
            history = _request_json(f"{base_url}/history/{prompt_id}", timeout=10)
            if prompt_id in history:
                record = history[prompt_id]
                error = _error_from_history(record)
                if error:
                    raise GenerationError(f"generation failed: {error}")
                if record.get("status", {}).get("completed"):
                    video = _find_video(record, output_directory)
                    if video is None:
                        raise GenerationError("generation completed but produced no MP4")
                    return video
            time.sleep(2)
    finally:
        stop_progress.set()
        progress_thread.join(timeout=3)
        if peak_rss:
            print(
                f"genvideo: peak process-tree RSS {peak_rss / 1024**3:.1f} GiB; "
                f"minimum system-available memory "
                f"{minimum_available / 1024**3:.1f} GiB; "
                f"peak swap growth {peak_swap_growth / 1024**3:.1f} GiB",
                file=sys.stderr,
                flush=True,
            )


def _stop_server(process: subprocess.Popen[bytes], own_group: bool = True) -> None:
    if process.poll() is not None:
        return
    try:
        if own_group:
            os.killpg(process.pid, signal.SIGINT)
        else:
            process.send_signal(signal.SIGINT)
        process.wait(timeout=20)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            if own_group:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()


class ComfySession:
    """An isolated ComfyUI process reusable across sequential generations."""

    def __init__(
        self, memory_limit_gib: float = 56.0, model: str = DEFAULT_MODEL,
        *, work_directory: Path | None = None, own_process_group: bool = True,
    ):
        if not 0 < memory_limit_gib <= HARD_MEMORY_CEILING_GIB:
            raise GenerationError(
                f"memory limit must be greater than 0 and at most "
                f"{HARD_MEMORY_CEILING_GIB:g} GiB"
            )
        if model not in SUPPORTED_MODELS:
            raise GenerationError(f"unsupported model: {model}")
        self.memory_limit = int(memory_limit_gib * 1024**3)
        self.model = model
        self.work_directory = work_directory
        self.own_process_group = own_process_group
        self._temporary: Any | None = None
        self._log_file: Any | None = None
        self.process: subprocess.Popen[bytes] | None = None
        self.base_url: str | None = None
        self.input_directory: Path | None = None
        self.output_directory: Path | None = None
        self.log_path: Path | None = None

    def start(self) -> None:
        if self.process is not None:
            _check_process(self.process, self.memory_limit)
            return
        if not COMFY_PYTHON.is_file() or not (COMFY_DIR / "main.py").is_file():
            raise GenerationError(f"ComfyUI runtime is missing from {COMFY_DIR}")
        workflow_path = WORKFLOW_PATHS[self.model]
        if not workflow_path.is_file():
            raise GenerationError(f"workflow template is missing: {workflow_path}")

        if self.work_directory is None:
            self._temporary = tempfile.TemporaryDirectory(prefix="genvideo-")
            temporary = Path(self._temporary.name)
        else:
            temporary = self.work_directory
            temporary.mkdir(parents=True, exist_ok=True)
        self.input_directory = temporary / "input"
        self.output_directory = temporary / "output"
        comfy_temp = temporary / "temp"
        user_directory = temporary / "user"
        for directory in (
            self.input_directory,
            self.output_directory,
            comfy_temp,
            user_directory,
        ):
            directory.mkdir(exist_ok=True)
        port = _free_port()
        self.base_url = f"http://127.0.0.1:{port}"
        self.log_path = temporary / "comfyui.log"
        self._log_file = self.log_path.open("wb")
        comfy_environment = os.environ.copy()
        comfy_environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), comfy_environment.get("PYTHONPATH")]))
        if sys.platform == "darwin":
            comfy_environment.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        memory_arguments = ["--lowvram", "--fast-disk"]
        try:
            self.process = subprocess.Popen(
                [
                    str(COMFY_PYTHON),
                    str(COMFY_DIR / "main.py"),
                    "--listen",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--disable-auto-launch",
                    "--input-directory",
                    str(self.input_directory),
                    "--output-directory",
                    str(self.output_directory),
                    "--temp-directory",
                    str(comfy_temp),
                    "--user-directory",
                    str(user_directory),
                    "--extra-model-paths-config",
                    str(ROOT / "extra_model_paths.yaml"),
                    "--cache-none",
                    *memory_arguments,
                    "--disable-metadata",
                ],
                cwd=COMFY_DIR,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                start_new_session=self.own_process_group,
                env=comfy_environment,
            )
            _wait_for_server(self.base_url, self.process, self.memory_limit)
        except BaseException as exc:
            error = self._error_with_log(exc)
            self.close()
            if error is exc:
                raise
            raise error from exc

    def _error_with_log(self, exc: BaseException) -> BaseException:
        if not isinstance(exc, GenerationError) or self.log_path is None:
            return exc
        if self._log_file is not None:
            self._log_file.flush()
        try:
            tail = self.log_path.read_text(errors="replace").splitlines()[-20:]
        except OSError:
            tail = []
        if not tail:
            return exc
        return GenerationError(f"{exc}\nLast ComfyUI log lines:\n" + "\n".join(tail))

    def generate(
        self,
        prompt: str,
        output: Path,
        *,
        image: Path | None = None,
        last_image: Path | None = None,
        references: list[dict] | None = None,
        seed: int | None = None,
        duration_seconds: int = DEFAULT_DURATION_SECONDS,
        model: str = DEFAULT_MODEL,
        resolution: int = DEFAULT_RESOLUTION,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        checkpoint_directory: Path | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> Path:
        if model not in SUPPORTED_MODELS:
            raise GenerationError(f"unsupported model: {model}")
        if model != self.model:
            raise GenerationError(
                f"session was started for {self.model}, not {model}"
            )
        self.start()
        assert self.process is not None
        assert self.base_url is not None
        assert self.input_directory is not None
        assert self.output_directory is not None

        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise GenerationError("prompt must not be empty")
        output = output.expanduser().resolve()
        if output.suffix.lower() != ".mp4":
            raise GenerationError("output path must end in .mp4")
        sources = [path.expanduser().resolve() if path is not None else None
                   for path in (image, last_image)]
        for source in sources:
            if source is not None and not source.is_file():
                raise GenerationError(f"input image does not exist: {source}")
            if source == output:
                raise GenerationError("input image and output path must be different")
        if duration_seconds not in SUPPORTED_DURATION_SECONDS:
            supported = ", ".join(str(value) for value in SUPPORTED_DURATION_SECONDS)
            raise GenerationError(f"duration must be one of: {supported} seconds")
        width, height = canvas_dimensions(resolution, aspect_ratio)
        selected_seed = seed if seed is not None else _random_seed()

        job_token = uuid.uuid4().hex
        prepared_images: list[Path] = []
        try:
            if any(sources) and shutil.which("ffmpeg") is None:
                raise GenerationError("ffmpeg is required but was not found on PATH")
            image_names = []
            for index, source in enumerate(sources):
                if source is None:
                    image_names.append(None)
                    continue
                prepared = self.input_directory / f"{job_token}-{index}.png"
                prepared_images.append(prepared)
                _prepare_image(source, prepared, width, height)
                image_names.append(prepared.name)
            workflow = _workflow(
                image_name=image_names[0],
                last_image_name=image_names[1],
                prompt=cleaned_prompt,
                seed=selected_seed,
                duration_seconds=duration_seconds,
                filename_prefix=f"job-{job_token}",
                model=model,
                width=width,
                height=height,
                references=references,
            )
            if checkpoint_directory is not None:
                workflow = _resumable_workflow(workflow, checkpoint_directory)
            generated = _generate(
                self.process,
                self.base_url,
                workflow,
                self.output_directory,
                self.memory_limit,
                progress_callback,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
            try:
                shutil.copyfile(generated, staging)
                os.replace(staging, output)
            finally:
                staging.unlink(missing_ok=True)
            try:
                generated.unlink(missing_ok=True)
            except OSError:
                pass
            return output
        except BaseException as exc:
            error = self._error_with_log(exc)
            if error is exc:
                raise
            raise error from exc
        finally:
            for prepared in prepared_images:
                prepared.unlink(missing_ok=True)

    def close(self) -> None:
        process, self.process = self.process, None
        if process is not None:
            _stop_server(process, self.own_process_group)
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
        self.base_url = None
        self.input_directory = None
        self.output_directory = None
        self.log_path = None

    def __enter__(self) -> "ComfySession":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _random_seed() -> int:
    return random.SystemRandom().randrange(1, 2**63 - 1)
