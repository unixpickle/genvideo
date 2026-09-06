import unittest

import generation


class H3PromptTests(unittest.TestCase):
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
