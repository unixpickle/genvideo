"""Memory-bounded MiniMax H3 conditioning for genvideo."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

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
        }}

    RETURN_TYPES = ("LATENT", "LATENT")
    FUNCTION = "sample"
    CATEGORY = "genvideo"

    def sample(self, noise, guider, sigmas, latent_image, checkpoint_directory):
        sampler = comfy.samplers.KSAMPLER(sample_resumable, {
            "checkpoint_path": str(Path(checkpoint_directory) / "sampler.pt"),
        })
        result = nodes_custom_sampler.SamplerCustomAdvanced.execute(
            noise, guider, sampler, sigmas, latent_image)
        atomic_save(to_cpu(result[0]), Path(checkpoint_directory) / "latent.pt")
        return (result[0], result[1])


NODE_CLASS_MAPPINGS = {
    "GenVideoMiniMaxH3Conditioning": GenVideoMiniMaxH3Conditioning,
    "GenVideoLoadConditioning": GenVideoLoadConditioning,
    "GenVideoLoadLatent": GenVideoLoadLatent,
    "GenVideoResumableSampler": GenVideoResumableSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GenVideoMiniMaxH3Conditioning": "MiniMax H3 Conditioning (release encoder)",
}
