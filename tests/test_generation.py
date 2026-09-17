import unittest

import generation


class H3PromptTests(unittest.TestCase):
    def test_end_frame_alignment_uses_image_order_and_snapped_duration(self):
        for first in (False, True):
            with self.subTest(first=first):
                prompt = generation.build_h3_prompt(
                    "The subject turns.", image_mode=first, last_frame=True,
                    duration_seconds=5,
                )
                self.assertIn("at 5.12 seconds", prompt)  # Frame 123 of 124, at 24 fps.
                self.assertIn(f"<Picture {2 if first else 1}> is fully referenced as the last frame", prompt)
                self.assertEqual("at 0.00 seconds" in prompt, first)

    def test_serializes_only_present_fields_in_official_order(self):
        prompt = generation.build_h3_prompt(
            "[Shot 1] A glass bird takes flight.",
            non_diegetic_music="N/A",
        )
        self.assertEqual(
            prompt,
            "integrated_multimodal_description: [Shot 1] A glass bird takes flight.\n\n"
            "non_diegetic_music: N/A",
        )

    def test_image_mode_adds_first_frame_alignment(self):
        prompt = generation.build_h3_prompt("The subject turns.", image_mode=True)
        self.assertTrue(
            prompt.startswith(
                "For the target video, at 0.00 seconds into the target video, "
                "<Picture 1> (from [Shot 1]) is fully referenced.\n\n"
            )
        )

    def test_requires_one_field(self):
        with self.assertRaisesRegex(generation.GenerationError, "at least one"):
            generation.build_h3_prompt()


class CanvasTests(unittest.TestCase):
    def test_supported_canvas_presets(self):
        expected_512 = {
            "21:9": (1024, 448),
            "16:9": (896, 512),
            "4:3": (672, 512),
            "1:1": (512, 512),
            "3:4": (512, 672),
            "9:16": (512, 896),
        }
        expected_768 = {
            "21:9": (1536, 672),
            "16:9": (1344, 768),
            "4:3": (1024, 768),
            "1:1": (768, 768),
            "3:4": (768, 1024),
            "9:16": (768, 1344),
        }
        self.assertEqual(
            {ratio: generation.canvas_dimensions(512, ratio) for ratio in expected_512},
            expected_512,
        )
        self.assertEqual(
            {ratio: generation.canvas_dimensions(768, ratio) for ratio in expected_768},
            expected_768,
        )


class WorkflowTests(unittest.TestCase):
    def test_model_presets_select_lora_steps_and_sampler(self):
        presets = (
            ("minimax-h3", "LoraLoaderModelOnly", 4, "res_multistep",
             "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"),
            ("minimax-h3-larry-v4", "MiniMaxH3TurboLoRA", 6, "euler",
             "minimax_h3_turbo_v4_step600_ema.safetensors"),
            ("minimax-h3-base", None, 20, "res_multistep", None),
        )
        for model, loader, steps, sampler, filename in presets:
            with self.subTest(model=model):
                workflow = generation._workflow(None, "A bird", 123, model=model)
                schedule = next(n["inputs"] for n in workflow.values()
                                if n["class_type"] == "BasicScheduler")
                self.assertEqual(schedule["steps"], steps)
                self.assertEqual(schedule["scheduler"], "simple")
                selected = next(n["inputs"] for n in workflow.values()
                                if n["class_type"] == "KSamplerSelect")
                self.assertEqual(selected["sampler_name"], sampler)
                loras = [n for n in workflow.values() if "lora_name" in n["inputs"]]
                self.assertEqual(len(loras), int(loader is not None))
                if loader:
                    self.assertEqual(loras[0]["class_type"], loader)
                    self.assertEqual(loras[0]["inputs"]["lora_name"], filename)
                if model == "minimax-h3-larry-v4":
                    self.assertEqual(loras[0]["inputs"]["strength"], 1.0)
                    self.assertFalse(loras[0]["inputs"]["low_vram"])

    def test_each_frame_combination_links_only_selected_images(self):
        for model in (m for m in generation.SUPPORTED_MODELS if m != generation.REF2VA_MODEL):
            for first, last in ((True, False), (False, True), (True, True), (False, False)):
                with self.subTest(model=model, first=first, last=last):
                    workflow = generation._workflow(
                        image_name="start.png" if first else None,
                        last_image_name="end.png" if last else None,
                        prompt="A bird", seed=123, model=model,
                    )
                    inputs = next(node["inputs"] for node in workflow.values()
                                  if node["class_type"] == "GenVideoMiniMaxH3Conditioning")
                    for present, key, filename in ((first, "first_frame", "start.png"),
                                                   (last, "last_frame", "end.png")):
                        self.assertEqual(key in inputs, present)
                        if present:
                            self.assertEqual(workflow[inputs[key][0]]["inputs"]["image"], filename)
                    self.assertEqual(sum(node["class_type"] == "LoadImage" for node in workflow.values()),
                                     int(first) + int(last))
                    for node in workflow.values():
                        for value in node["inputs"].values():
                            if isinstance(value, list):
                                self.assertIn(value[0], workflow)

    def test_workflow_applies_duration_canvas_and_text_mode(self):
        prompt = generation.build_h3_prompt("[Shot 1] A glass bird takes flight.")
        workflow = generation._workflow(
            image_name=None,
            prompt=prompt,
            seed=123,
            duration_seconds=15,
            model="minimax-h3",
            width=768,
            height=1344,
        )
        conditioners = [
            node
            for node in workflow.values()
            if node["class_type"] == "GenVideoMiniMaxH3Conditioning"
        ]
        self.assertEqual(len(conditioners), 1)
        self.assertEqual(conditioners[0]["inputs"]["width"], 768)
        self.assertEqual(conditioners[0]["inputs"]["height"], 1344)
        self.assertNotIn("first_frame", conditioners[0]["inputs"])
        self.assertFalse(
            any(node["class_type"] == "LoadImage" for node in workflow.values())
        )
        duration = next(
            node
            for node in workflow.values()
            if node.get("_meta", {}).get("title") == "Duration"
        )
        self.assertEqual(duration["inputs"]["value"], 15)


if __name__ == "__main__":
    unittest.main()
