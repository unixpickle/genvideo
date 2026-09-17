"""Durable state for H3's deterministic res_multistep and Euler samplers.

Keep the full schedule and multistep history: restarting a sliced schedule with
only the latent would silently change the second-order solver's trajectory.
"""
from pathlib import Path
import os

import torch


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_state(path):
    # These are private, locally generated checkpoints, never uploaded tensors.
    return torch.load(path, map_location="cpu", weights_only=False)


@torch.no_grad()
def sample_resumable(model, x, sigmas, extra_args=None, callback=None,
                     disable=None, *, checkpoint_path, sampler_name="res_multistep"):
    if sampler_name not in {"res_multistep", "euler"}:
        raise ValueError(f"Unsupported checkpoint sampler: {sampler_name}")
    path = Path(checkpoint_path)
    extra_args = extra_args or {}
    start = 0
    old_denoised = old_sigma_down = None
    if path.is_file():
        state = load_state(path)
        if state["version"] != 1 or not torch.equal(state["sigmas"], sigmas.cpu()):
            raise ValueError("Checkpoint sampler schedule does not match this job")
        if state.get("sampler_name", "res_multistep") != sampler_name:
            raise ValueError("Checkpoint sampler does not match this job")
        if state["x"].shape != x.shape:
            raise ValueError("Checkpoint latent shape does not match this job")
        start = state["step"]
        if not 0 <= start < len(sigmas):
            raise ValueError("Invalid checkpoint step")
        x = state["x"].to(x)
        old_denoised = state["old_denoised"].to(x)
        old_sigma_down = sigmas[start]
        del state
    s_in = x.new_ones([x.shape[0]])
    for i in range(start, len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down = sigmas[i + 1]  # res_multistep uses eta=0, no added noise.
        if sampler_name == "euler" or sigma_down == 0 or old_denoised is None:
            d = (x - denoised) / sigmas[i]
            x = x + d * (sigma_down - sigmas[i])
        else:
            t = -sigmas[i].log()
            t_old = -old_sigma_down.log()
            t_next = -sigma_down.log()
            t_prev = -sigmas[i - 1].log()
            h = t_next - t
            c2 = (t_prev - t_old) / h
            phi1 = torch.expm1(-h) / -h
            phi2 = (phi1 - 1.0) / -h
            b1 = torch.nan_to_num(phi1 - phi2 / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2 / c2, nan=0.0)
            x = (-h).exp() * x + h * (b1 * denoised + b2 * old_denoised)
        old_denoised, old_sigma_down = denoised, sigma_down
        atomic_save({"version": 1, "sampler_name": sampler_name,
                     "step": i + 1, "sigmas": sigmas.cpu(),
                     "x": x.cpu(), "old_denoised": denoised.cpu()}, path)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i],
                      "sigma_hat": sigmas[i], "denoised": denoised})
    return x
