"""One-shot, memory-bounded LTX-2.5 image-to-video generation."""

from __future__ import annotations

import argparse
import asyncio
import json
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
from typing import Any

import aiohttp
import psutil


ROOT = Path(__file__).resolve().parent
COMFY_DIR = ROOT / "ComfyUI"
COMFY_PYTHON = ROOT / ".venv" / "bin" / "python"
WORKFLOW_PATH = ROOT / "workflows" / "ltx2_5_video_api.json"
IMAGE_SIZE = 512
HARD_MEMORY_CEILING_GIB = 64.0
DEFAULT_DURATION_SECONDS = 3
SUPPORTED_DURATION_SECONDS = (3, 5)
PROGRESS_NODE_CLASSES = {
    "CLIPLoader",
    "CLIPTextEncode",
    "CreateVideo",
    "LatentUpscaleModelLoader",
    "LTXVAudioVAEDecode",
    "LTXVLatentUpsampler",
    "SamplerCustomAdvanced",
    "SaveVideo",
    "UNETLoader",
    "UnetLoaderGGUF",
    "VAEDecode",
    "VAEDecodeTiled",
    "VAELoader",
}


class GenerationError(RuntimeError):
    pass


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, help="generation seed (default: random)")
    parser.add_argument(
        "--duration",
        type=int,
        choices=SUPPORTED_DURATION_SECONDS,
        default=DEFAULT_DURATION_SECONDS,
        metavar="SECONDS",
        help="video duration in seconds (choices: 3 or 5; default: 3)",
    )
    parser.add_argument(
        "--memory-limit-gib",
        type=float,
        default=56.0,
        metavar="GIB",
        help="stop if process-tree RSS exceeds this value (default: 56, maximum: 64)",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] == "--text":
        parser = argparse.ArgumentParser(
            prog="genvideo --text",
            description=(
                "Generate one 512x512 LTX-2.5 video from a text prompt. All "
                "intermediate files are temporary."
            ),
        )
        parser.add_argument("prompt", help="description of the desired video")
        parser.add_argument("output", type=Path, help="final .mp4 path")
        _add_runtime_options(parser)
        parsed = parser.parse_args(arguments[1:])
        parsed.image = None
        parsed.text_only = True
        return parsed

    parser = argparse.ArgumentParser(
        prog="genvideo",
        description=(
            "Generate one 512x512 LTX-2.5 video from an image. The image is "
            "scaled to cover and center-cropped; all intermediate files are temporary. "
            "For generation without an image, use: genvideo --text PROMPT OUTPUT"
        ),
    )
    parser.add_argument("image", type=Path, help="initial image (any dimensions)")
    parser.add_argument("prompt", help="description of the desired motion/video")
    parser.add_argument("output", type=Path, help="final .mp4 path")
    _add_runtime_options(parser)
    parsed = parser.parse_args(arguments)
    parsed.text_only = False
    return parsed


def _validate_args(args: argparse.Namespace) -> tuple[Path | None, Path]:
    image = args.image.expanduser().resolve() if args.image is not None else None
    output = args.output.expanduser().resolve()
    if image is not None and not image.is_file():
        raise GenerationError(f"input image does not exist: {image}")
    if output.suffix.lower() != ".mp4":
        raise GenerationError("output path must end in .mp4")
    if image is not None and image == output:
        raise GenerationError("input image and output path must be different")
    if not args.prompt.strip():
        raise GenerationError("prompt must not be empty")
    if not 0 < args.memory_limit_gib <= HARD_MEMORY_CEILING_GIB:
        raise GenerationError(
            f"--memory-limit-gib must be greater than 0 and at most "
            f"{HARD_MEMORY_CEILING_GIB:g}"
        )
    if not COMFY_PYTHON.is_file() or not (COMFY_DIR / "main.py").is_file():
        raise GenerationError(f"ComfyUI runtime is missing from {COMFY_DIR}")
    if not WORKFLOW_PATH.is_file():
        raise GenerationError(f"workflow template is missing: {WORKFLOW_PATH}")
    if shutil.which("ffmpeg") is None:
        raise GenerationError("ffmpeg is required but was not found on PATH")
    return image, output


def _prepare_image(source: Path, destination: Path) -> None:
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
            f"scale={IMAGE_SIZE}:{IMAGE_SIZE}:force_original_aspect_ratio=increase,"
            f"crop={IMAGE_SIZE}:{IMAGE_SIZE},setsar=1,format=rgb24"
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
    image_conditioners = {
        node_id: node["inputs"]["latent"]
        for node_id, node in workflow.items()
        if node["class_type"] == "LTXVImgToVideoInplace"
    }
    if not image_conditioners:
        raise GenerationError("workflow has no image-conditioning stages to bypass")

    for node in workflow.values():
        for name, value in node["inputs"].items():
            if (
                isinstance(value, list)
                and len(value) == 2
                and value[0] in image_conditioners
            ):
                node["inputs"][name] = list(image_conditioners[value[0]])
    for node_id in image_conditioners:
        del workflow[node_id]

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
) -> dict[str, Any]:
    with WORKFLOW_PATH.open() as workflow_file:
        workflow: dict[str, Any] = json.load(workflow_file)

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
    if (
        len(load_images) != 1
        or len(prompts) != 1
        or len(save_videos) != 1
        or len(durations) != 1
    ):
        raise GenerationError("workflow template has an unexpected structure")
    if duration_seconds not in SUPPORTED_DURATION_SECONDS:
        supported = ", ".join(str(value) for value in SUPPORTED_DURATION_SECONDS)
        raise GenerationError(f"duration must be one of: {supported} seconds")

    if image_name is not None:
        load_images[0]["inputs"]["image"] = image_name
    prompts[0]["inputs"]["value"] = prompt
    durations[0]["inputs"]["value"] = duration_seconds
    save_videos[0]["inputs"]["filename_prefix"] = filename_prefix
    for offset, node in enumerate(noise_nodes):
        node["inputs"]["noise_seed"] = seed + offset
    return workflow if image_name is not None else _text_only_graph(workflow)


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
) -> None:
    asyncio.run(
        _progress_listener(
            base_url,
            client_id,
            node_titles,
            logged_node_ids,
            stop_event,
            ready_event,
        )
    )


def _generate(
    process: subprocess.Popen[bytes],
    base_url: str,
    workflow: dict[str, Any],
    output_directory: Path,
    memory_limit: int,
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
        ),
        daemon=True,
    )
    progress_thread.start()
    progress_ready.wait(timeout=10)
    peak_rss = 0
    minimum_available = psutil.virtual_memory().available
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
                f"{minimum_available / 1024**3:.1f} GiB",
                file=sys.stderr,
                flush=True,
            )


def _stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=20)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


class ComfySession:
    """An isolated ComfyUI process reusable across sequential generations."""

    def __init__(self, memory_limit_gib: float = 56.0):
        if not 0 < memory_limit_gib <= HARD_MEMORY_CEILING_GIB:
            raise GenerationError(
                f"memory limit must be greater than 0 and at most "
                f"{HARD_MEMORY_CEILING_GIB:g} GiB"
            )
        self.memory_limit = int(memory_limit_gib * 1024**3)
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
        if not WORKFLOW_PATH.is_file():
            raise GenerationError(f"workflow template is missing: {WORKFLOW_PATH}")

        self._temporary = tempfile.TemporaryDirectory(prefix="genvideo-")
        temporary = Path(self._temporary.name)
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
            directory.mkdir()
        port = _free_port()
        self.base_url = f"http://127.0.0.1:{port}"
        self.log_path = temporary / "comfyui.log"
        self._log_file = self.log_path.open("wb")
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
                    "--cache-none",
                    "--lowvram",
                    "--disable-metadata",
                ],
                cwd=COMFY_DIR,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
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
        seed: int | None = None,
        duration_seconds: int = DEFAULT_DURATION_SECONDS,
    ) -> Path:
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
        source = image.expanduser().resolve() if image is not None else None
        if source is not None and not source.is_file():
            raise GenerationError(f"input image does not exist: {source}")
        if source == output:
            raise GenerationError("input image and output path must be different")
        if duration_seconds not in SUPPORTED_DURATION_SECONDS:
            supported = ", ".join(str(value) for value in SUPPORTED_DURATION_SECONDS)
            raise GenerationError(f"duration must be one of: {supported} seconds")
        selected_seed = seed if seed is not None else _random_seed()

        job_token = uuid.uuid4().hex
        prepared_image: Path | None = None
        if source is None:
            workflow = _workflow(
                None,
                cleaned_prompt,
                selected_seed,
                duration_seconds,
                f"job-{job_token}",
            )
        else:
            if shutil.which("ffmpeg") is None:
                raise GenerationError("ffmpeg is required but was not found on PATH")
            prepared_image = self.input_directory / f"{job_token}.png"
            _prepare_image(source, prepared_image)
            workflow = _workflow(
                prepared_image.name,
                cleaned_prompt,
                selected_seed,
                duration_seconds,
                f"job-{job_token}",
            )
        try:
            generated = _generate(
                self.process,
                self.base_url,
                workflow,
                self.output_directory,
                self.memory_limit,
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
            if prepared_image is not None:
                prepared_image.unlink(missing_ok=True)

    def close(self) -> None:
        process, self.process = self.process, None
        if process is not None:
            _stop_server(process)
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


def run(args: argparse.Namespace) -> Path:
    image, output = _validate_args(args)
    seed = args.seed if args.seed is not None else _random_seed()
    with ComfySession(args.memory_limit_gib) as session:
        return session.generate(
            args.prompt,
            output,
            image=image,
            seed=seed,
            duration_seconds=args.duration,
        )


def main() -> None:
    try:
        output = run(_parse_args())
    except (GenerationError, OSError, KeyboardInterrupt) as exc:
        message = str(exc) if str(exc) else "interrupted"
        print(f"genvideo: error: {message}", file=sys.stderr)
        raise SystemExit(1) from None
    print(output)


if __name__ == "__main__":
    main()
