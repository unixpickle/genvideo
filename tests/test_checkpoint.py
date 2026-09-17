import ast
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from checkpoint_sampler import atomic_save, load_state, sample_resumable
import generation


class SamplerTests(unittest.TestCase):
    def test_resume_every_boundary_matches_comfy_samplers(self):
        # Run the installed upstream solver as the reference without importing
        # ComfyUI's model/GPU initialization into the web test process.
        source = ast.parse((generation.COMFY_DIR / "comfy/k_diffusion/sampling.py").read_text())
        functions = []
        for node in source.body:
            if isinstance(node, ast.FunctionDef) and node.name in {
                "res_multistep", "sample_euler", "get_ancestral_step", "to_d",
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

        for sampler_name, steps in (("res_multistep", 4), ("res_multistep", 20), ("euler", 6)):
            sigmas = torch.linspace(1.0, 0, steps + 1)
            noise = torch.randn((1, 32), generator=torch.Generator().manual_seed(7))
            reference_fn = namespace["sample_euler" if sampler_name == "euler" else "res_multistep"]
            kwargs = {} if sampler_name == "euler" else {"eta": 0}
            reference = reference_fn(Model(), noise.clone(), sigmas, **kwargs)
            for boundary in range(1, steps + 1):
                with self.subTest(sampler=sampler_name, steps=steps, boundary=boundary), tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "sampler.pt"

                    def interrupt(event):
                        if event["i"] + 1 == boundary:
                            raise InterruptedError()

                    with self.assertRaises(InterruptedError):
                        sample_resumable(Model(), noise.clone(), sigmas,
                                         callback=interrupt, checkpoint_path=path,
                                         sampler_name=sampler_name)
                    resumed_model = Model()
                    resumed = sample_resumable(resumed_model, noise.clone(), sigmas,
                                               checkpoint_path=path, sampler_name=sampler_name)
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

    def test_sampler_mismatch_rejected_and_legacy_checkpoint_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sampler.pt"
            sigmas = torch.tensor([1., .5, 0.])
            model = lambda x, sigma: x * .5
            noise = torch.ones(1, 2)
            expected = sample_resumable(model, noise, sigmas, checkpoint_path=path)
            state = load_state(path)
            del state["sampler_name"]  # Existing saved jobs predate this field.
            atomic_save(state, path)
            with self.assertRaisesRegex(ValueError, "sampler"):
                sample_resumable(model, noise, sigmas, checkpoint_path=path, sampler_name="euler")
            actual = sample_resumable(model, noise, sigmas, checkpoint_path=path)
            self.assertTrue(torch.equal(actual, expected))


class ResumeWorkflowTests(unittest.TestCase):
    def test_resumption_retains_selected_sampler(self):
        for model, sampler in (("minimax-h3", "res_multistep"),
                               ("minimax-h3-larry-v4", "euler"),
                               ("minimax-h3-base", "res_multistep")):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                workflow = generation._resumable_workflow(
                    generation._workflow(None, "A bird", 12, model=model), Path(tmp))
                inputs = next(n["inputs"] for n in workflow.values()
                              if n["class_type"] == "GenVideoResumableSampler")
                self.assertEqual(inputs["sampler_name"], sampler)

    def workflow(self, directory):
        return generation._resumable_workflow(generation._workflow(
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
