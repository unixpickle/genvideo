import ast
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from checkpoint_sampler import atomic_save, load_state, sample_resumable
import main


class SamplerTests(unittest.TestCase):
    def test_resume_every_boundary_matches_comfy_multistep(self):
        # Run the installed upstream solver as the reference without importing
        # ComfyUI's model/GPU initialization into the web test process.
        source = ast.parse((main.COMFY_DIR / "comfy/k_diffusion/sampling.py").read_text())
        functions = []
        for node in source.body:
            if isinstance(node, ast.FunctionDef) and node.name in {
                "res_multistep", "get_ancestral_step", "to_d",
            }:
                node.decorator_list = []
                functions.append(node)
        namespace = {
            "torch": torch, "trange": lambda n, **kwargs: range(n),
            "default_noise_sampler": lambda *args, **kwargs: None,
            "utils": mock.Mock(append_dims=lambda value, ndim: value.reshape([1] * ndim)),
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), "upstream", "exec"), namespace)

        class Model:
            inner_model = mock.Mock()
            inner_model.model_patcher.get_model_object.return_value.noise_scale = 1.0

            def __init__(self):
                self.calls = 0

            def __call__(self, x, sigma, **kwargs):
                self.calls += 1
                return torch.sin(x) * 0.3 + sigma.reshape(-1, 1) * 0.2

        for steps in (4, 20):
            sigmas = torch.linspace(1.0, 0, steps + 1)
            noise = torch.randn((1, 32), generator=torch.Generator().manual_seed(7))
            reference = namespace["res_multistep"](Model(), noise.clone(), sigmas, eta=0)
            for boundary in range(1, steps + 1):
                with self.subTest(steps=steps, boundary=boundary), tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "sampler.pt"

                    def interrupt(event):
                        if event["i"] + 1 == boundary:
                            raise InterruptedError()

                    with self.assertRaises(InterruptedError):
                        sample_resumable(Model(), noise.clone(), sigmas,
                                         callback=interrupt, checkpoint_path=path)
                    resumed_model = Model()
                    resumed = sample_resumable(resumed_model, noise.clone(), sigmas,
                                               checkpoint_path=path)
                    self.assertTrue(torch.equal(reference, resumed))
                    self.assertEqual(resumed_model.calls, steps - boundary)

    def test_interrupted_checkpoint_write_keeps_previous_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pt"
            atomic_save({"step": 2}, path)
            with mock.patch("checkpoint_sampler.torch.save", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    atomic_save({"step": 3}, path)
            self.assertEqual(load_state(path), {"step": 2})
            self.assertFalse(path.with_suffix(".pt.tmp").exists())

    def test_schedule_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pt"
            sigmas = torch.tensor([1., .5, 0.])
            sample_resumable(lambda x, sigma: x * .5, torch.ones(1, 2), sigmas,
                             checkpoint_path=path)
            with self.assertRaisesRegex(ValueError, "schedule"):
                sample_resumable(lambda x, sigma: x, torch.ones(1, 2),
                                 torch.tensor([1., .6, 0.]), checkpoint_path=path)


class ResumeWorkflowTests(unittest.TestCase):
    def workflow(self, directory):
        return main._resumable_workflow(main._workflow(
            image_name="input.png", prompt="A bird", seed=12), directory)

    def test_conditioning_restore_prunes_text_encoder_and_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "conditioning.pt").touch()
            kinds = {node["class_type"] for node in self.workflow(directory).values()}
            self.assertIn("GenVideoLoadConditioning", kinds)
            self.assertIn("GenVideoResumableSampler", kinds)
            self.assertNotIn("CLIPLoader", kinds)
            self.assertNotIn("LoadImage", kinds)

    def test_final_latents_skip_all_encoding_and_diffusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "latent.pt").touch()
            kinds = {node["class_type"] for node in self.workflow(directory).values()}
            self.assertIn("GenVideoLoadLatent", kinds)
            for kind in ("CLIPLoader", "UnetLoaderGGUF", "GenVideoMiniMaxH3Conditioning",
                         "GenVideoResumableSampler", "LoraLoaderModelOnly"):
                self.assertNotIn(kind, kinds)
