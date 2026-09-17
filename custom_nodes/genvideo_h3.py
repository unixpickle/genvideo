"""Memory-bounded MiniMax H3 conditioning for genvideo."""

from __future__ import annotations

import gc
import logging
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import comfy.samplers
import comfy.nested_tensor
from comfy_extras import nodes_custom_sampler
from checkpoint_sampler import atomic_save, load_state, sample_resumable

import comfy.model_management
import node_helpers
from comfy_extras import nodes_minimax_h3


class GenVideoMiniMaxH3Conditioning:
    """Encode H3 conditioning, then destructively release the text encoder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "width": ("INT", {"default": 512, "min": 32, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 512, "min": 32, "max": 16384, "step": 32}),
                "length": ("INT", {"default": 73, "min": 5, "max": 3600, "step": 17}),
            },
            "optional": {
                "checkpoint_directory": ("STRING", {"default": ""}),
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "encode_and_release"
    CATEGORY = "genvideo"
    DESCRIPTION = (
        "MiniMax H3 conditioning that permanently releases its text encoder "
        "before diffusion, reducing peak unified-memory pressure."
    )

    @staticmethod
    def _release_clip(clip) -> None:
        patcher = clip.patcher
        comfy.model_management.unload_model_and_clones(
            patcher, unload_additional_models=True, all_devices=True
        )

        # This node is deliberately terminal for this CLIP instance. Merely
        # unloading it moves weights to CPU, which is still unified RAM on a
        # Mac; severing both owning references lets the 32B encoder be freed.
        clip.patcher = None
        clip.cond_stage_model = None
        del patcher
        gc.collect()
        comfy.model_management.cleanup_models_gc()
        comfy.model_management.soft_empty_cache(force=True)
        logging.info("genvideo released the MiniMax H3 text encoder after conditioning")

    def encode_and_release(
        self,
        clip,
        vae,
        prompt,
        width,
        height,
        length,
        first_frame=None,
        last_frame=None,
        checkpoint_directory="",
    ):
        latent, frame_count = nodes_minimax_h3._empty_av_latent(
            width, height, length
        )

        images = []
        keyframes = []
        if first_frame is not None:
            image = nodes_minimax_h3._resize(
                first_frame[:1], width, height, "disabled"
            )
            images.append(image)
            keyframes.append({"resolved_frame_index": 0, "image": image})
        if last_frame is not None:
            image = nodes_minimax_h3._resize(
                last_frame[:1], width, height, "center"
            )
            images.append(image)
            keyframes.append(
                {"resolved_frame_index": frame_count - 1, "image": image}
            )

        try:
            tokens = clip.tokenize(prompt, images=images)
            conditioning = clip.encode_from_tokens_scheduled(tokens)

            if keyframes:
                for keyframe in keyframes:
                    keyframe["latent"] = vae.encode(keyframe.pop("image"))
                conditioning = node_helpers.conditioning_set_values(
                    conditioning, {"minimax_keyframes": keyframes}
                )
        finally:
            self._release_clip(clip)

        if checkpoint_directory:
            atomic_save(to_cpu((conditioning, latent)),
                        Path(checkpoint_directory) / "conditioning.pt")
        return (conditioning, latent)


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, comfy.nested_tensor.NestedTensor):
        return comfy.nested_tensor.NestedTensor([to_cpu(x) for x in value.unbind()])
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(to_cpu(v) for v in value)
    return value


class GenVideoMiniMaxH3ReferenceConditioning(GenVideoMiniMaxH3Conditioning):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update(audio_vae=("VAE",), references_json=("STRING",))
        inputs["optional"].pop("first_frame")
        inputs["optional"].pop("last_frame")
        return inputs

    @staticmethod
    def _audio(path):
        data = subprocess.run([
            "ffmpeg", "-v", "error", "-i", str(path), "-t", "15", "-vn",
            "-ac", "2", "-ar", "32000", "-f", "f32le", "pipe:1",
        ], capture_output=True, check=True, timeout=90).stdout
        samples = np.frombuffer(data, dtype=np.float32).copy().reshape(-1, 2)
        return {"waveform": torch.from_numpy(samples.T).unsqueeze(0), "sample_rate": 32000}

    def encode_and_release(self, clip, vae, audio_vae, prompt, width, height,
                           length, references_json, checkpoint_directory=""):
        images, videos, soundtracks, audios = {}, {}, {}, {}
        try:
            for ref in json.loads(references_json):
                path = Path(ref["path"])
                if ref["kind"] == "image":
                    with Image.open(path) as source:
                        img = ImageOps.exif_transpose(source).convert("RGB")
                        scale = min(1, math.sqrt(width * height / (img.width * img.height)))
                        img = img.resize((max(32, round(img.width * scale / 32) * 32),
                                          max(32, round(img.height * scale / 32) * 32)))
                        images[f"ref_image_{len(images)}"] = torch.from_numpy(
                            np.asarray(img).copy()).float().unsqueeze(0) / 255
                elif ref["kind"] == "video":
                    scale = min(1, math.sqrt(width * height / (ref["width"] * ref["height"])))
                    w = max(32, round(ref["width"] * scale / 32) * 32)
                    h = max(32, round(ref["height"] * scale / 32) * 32)
                    data = subprocess.run([
                        "ffmpeg", "-v", "error", "-i", str(path), "-t", "15", "-an",
                        "-vf", f"fps=24,scale={w}:{h}", "-frames:v", str(length),
                        "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
                    ], capture_output=True, check=True, timeout=90).stdout
                    index = len(videos)
                    videos[f"ref_video_{index}"] = torch.from_numpy(
                        np.frombuffer(data, dtype=np.uint8).copy().reshape(-1, h, w, 3)).float() / 255
                    if ref.get("use_audio"):
                        soundtracks[f"ref_video_audio_{index}"] = self._audio(path)
                else:
                    audios[f"ref_audio_{len(audios)}"] = self._audio(path)
            result = nodes_minimax_h3.MiniMaxH3ReferenceToVideo.execute(
                clip, vae, audio_vae, prompt, width, height, length,
                ref_images=images, ref_videos=videos, ref_video_audios=soundtracks, ref_audios=audios,
            )
            conditioning, latent = result[0], result[1]
        finally:
            self._release_clip(clip)
        if checkpoint_directory:
            atomic_save(to_cpu((conditioning, latent)), Path(checkpoint_directory) / "conditioning.pt")
        return (conditioning, latent)


class GenVideoLoadConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"checkpoint_directory": ("STRING",)}}

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    FUNCTION = "load"
    CATEGORY = "genvideo"

    def load(self, checkpoint_directory):
        return tuple(load_state(Path(checkpoint_directory) / "conditioning.pt"))


class GenVideoLoadLatent(GenVideoLoadConditioning):
    RETURN_TYPES = ("LATENT", "LATENT")

    def load(self, checkpoint_directory):
        latent = load_state(Path(checkpoint_directory) / "latent.pt")
        return (latent, latent)


class GenVideoResumableSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "noise": ("NOISE",), "guider": ("GUIDER",),
            "sigmas": ("SIGMAS",), "latent_image": ("LATENT",),
            "checkpoint_directory": ("STRING",),
            "sampler_name": (["res_multistep", "euler"], {"default": "res_multistep"}),
        }}

    RETURN_TYPES = ("LATENT", "LATENT")
    FUNCTION = "sample"
    CATEGORY = "genvideo"

    def sample(self, noise, guider, sigmas, latent_image, checkpoint_directory,
               sampler_name="res_multistep"):
        sampler = comfy.samplers.KSAMPLER(sample_resumable, {
            "checkpoint_path": str(Path(checkpoint_directory) / "sampler.pt"),
            "sampler_name": sampler_name,
        })
        result = nodes_custom_sampler.SamplerCustomAdvanced.execute(
            noise, guider, sampler, sigmas, latent_image)
        atomic_save(to_cpu(result[0]), Path(checkpoint_directory) / "latent.pt")
        return (result[0], result[1])


NODE_CLASS_MAPPINGS = {
    "GenVideoMiniMaxH3ReferenceConditioning": GenVideoMiniMaxH3ReferenceConditioning,
    "GenVideoMiniMaxH3Conditioning": GenVideoMiniMaxH3Conditioning,
    "GenVideoLoadConditioning": GenVideoLoadConditioning,
    "GenVideoLoadLatent": GenVideoLoadLatent,
    "GenVideoResumableSampler": GenVideoResumableSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GenVideoMiniMaxH3Conditioning": "MiniMax H3 Conditioning (release encoder)",
}
