"""Ref2VA prompt and attachment validation shared by the web app and worker."""
from __future__ import annotations

import json
import math
import mimetypes
import re
import subprocess
from pathlib import Path

REF2VA_MODEL = "minimax-h3-ref2va"
REF2VA_WEIGHTS = "minimax_h3_ref2va_pruned-Q4_K.gguf"
REF_FIELDS = ("subject_definitions", "summary", "retention_analysis", "detailed_description",
              "overall_soundscape", "non_diegetic_music")
MAX_REFERENCE_BYTES = 256 * 1024 * 1024
MAX_REFERENCES = 12


def reference_labels(references):
    """Match MiniMaxH3ReferenceToVideo: pictures, videos + enabled tracks, audio."""
    counts = {"image": 0, "video": 0, "audio": 0}
    labels = {}
    for kind in counts:
        for ref in references:
            if ref["kind"] != kind:
                continue
            counts[kind] += 1
            label = f'<{dict(image="Picture", video="Video", audio="Audio")[kind]} {counts[kind]}>'
            labels[ref["id"]] = [label]
            if kind == "video" and ref.get("use_audio"):
                counts["audio"] += 1
                labels[ref["id"]].append(f'<Audio {counts["audio"]}>')
    return labels


def validate_references(value):
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_REFERENCES:
        raise ValueError("Attach between 1 and 12 reference files.")
    counts = {"image": 0, "video": 0, "audio": 0}
    seen = set()
    result = []
    for ref in value:
        if not isinstance(ref, dict):
            raise ValueError("Invalid reference metadata.")
        ident, kind = ref.get("id"), ref.get("kind")
        if not isinstance(ident, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", ident) or ident in seen:
            raise ValueError("Reference IDs must be unique letters, numbers, or underscores.")
        if not isinstance(kind, str) or kind not in counts:
            raise ValueError("References must be images, videos, or audio.")
        if not isinstance(ref.get("use_audio", False), bool):
            raise ValueError("use_audio must be a boolean.")
        seen.add(ident)
        counts[kind] += 1
        result.append({"id": ident, "kind": kind, "use_audio": kind == "video" and ref.get("use_audio", False),
                       "name": str(ref.get("name", kind))[:255]})
    if counts["image"] > 9 or counts["video"] > 3 or counts["audio"] + sum(r["use_audio"] for r in result) > 3:
        raise ValueError("Use at most 9 images, 3 videos, and 3 audio tracks (including enabled video soundtracks).")
    return result


def probe_reference(path: Path, ref: dict):
    if ref["kind"] == "image":
        from PIL import Image
        if path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("Reference images must be at most 32 MB.")
        try:
            with Image.open(path) as image:
                if image.width * image.height > 40_000_000:
                    raise ValueError("Reference images must be at most 40 megapixels.")
                ref["content_type"] = Image.MIME.get(image.format, "image/png")
                image.verify()
        except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
            raise ValueError(f'Cannot read reference image: {ref["name"]}') from exc
    try:
        proc = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                              capture_output=True, check=True, timeout=30)
        info = json.loads(proc.stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise ValueError(f'Cannot read reference media: {ref["name"]}') from exc
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if (ref["kind"] in {"image", "video"} and not video) or (ref["kind"] == "audio" and not audio):
        raise ValueError(f'Reference does not contain the selected media type: {ref["name"]}')
    if ref["use_audio"] and not audio:
        raise ValueError(f'No soundtrack in {ref["name"]}; disable its audio track.')
    if video and (video.get("width", 0) <= 0 or video.get("height", 0) <= 0
                  or video["width"] * video["height"] > 40_000_000):
        raise ValueError("Reference images/video must be at most 40 megapixels.")
    if ref["kind"] != "image":
        if not ref.get("content_type", "").startswith(ref["kind"] + "/"):
            guessed = mimetypes.guess_type(ref["name"])[0] or ""
            ref["content_type"] = guessed if guessed.startswith(ref["kind"] + "/") else (
                "video/mp4" if ref["kind"] == "video" else "audio/wav")
        try:
            duration = float(info.get("format", {}).get("duration", "nan"))
        except (ValueError, TypeError):
            duration = float("nan")
        if not math.isfinite(duration) or not 2 <= duration <= 15.05:
            raise ValueError("Each reference video/audio clip must be 2–15 seconds long.")
        ref["duration"] = duration
    if video:
        ref["width"], ref["height"] = video["width"], video["height"]
        rotation = next((s.get("rotation", 0) for s in video.get("side_data_list", [])
                         if "rotation" in s), 0)
        if abs(float(rotation)) % 180 == 90:
            ref["width"], ref["height"] = ref["height"], ref["width"]


def validate_durations(references):
    for kind in ("video", "audio"):
        total = sum(r.get("duration", 0) for r in references
                    if r["kind"] == kind or (kind == "audio" and r.get("use_audio")))
        if total > 15.05:
            raise ValueError(f"Total reference {kind} duration must not exceed 15 seconds.")


def build_ref2va_prompt(fields, references):
    missing = [name for name in REF_FIELDS if not fields.get(name, "").strip()]
    if missing:
        raise ValueError("Complete all six Ref2VA sections (use N/A for absent sound or music).")
    prompt = "\n\n".join(f"{name}:\n{fields[name].strip()}" for name in REF_FIELDS)
    if len(prompt) > 24000:
        raise ValueError("Combined Ref2VA prompt must be 24,000 characters or fewer.")
    labels = {label for group in reference_labels(references).values() for label in group}
    subjects = set(re.findall(r"<Subject [1-9][0-9]*>", fields["subject_definitions"]))
    used = set(re.findall(r"<(?:Picture|Video|Audio|Subject) [0-9]+>", prompt))
    unknown = used - labels - subjects
    if unknown:
        raise ValueError("Prompt refers to missing references: " + ", ".join(sorted(unknown)))
    return prompt
