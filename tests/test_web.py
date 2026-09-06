import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aiohttp import web

import web as queue_web


class FakeRequest:
    content_type = "application/x-www-form-urlencoded"

    def __init__(self, manager, fields):
        self.app = {queue_web.MANAGER_KEY: manager}
        self._fields = fields

    async def post(self):
        return self._fields


class CreateJobTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        output = Path(self.temporary.name)
        self.manager = queue_web.QueueManager(output, 56.0)
        output.mkdir(exist_ok=True)
        self.manager.upload_directory.mkdir()

    async def asyncTearDown(self):
        self.temporary.cleanup()

    async def test_structured_job_compiles_prompt_and_output_options(self):
        request = FakeRequest(
            self.manager,
            {
                "mode": "text",
                "model": "minimax-h3",
                "duration": "15",
                "resolution": "768",
                "aspect_ratio": "9:16",
                "integrated_multimodal_description": "[Shot 1] Rain crosses a window.",
                "overall_soundscape": "Steady rain and low room tone.",
                "non_diegetic_music": "N/A",
            },
        )
        response = await queue_web.create_job(request)
        payload = json.loads(response.body)
        self.assertEqual(response.status, 201)
        self.assertEqual(payload["model"], "minimax-h3")
        self.assertEqual(payload["duration_seconds"], 15)
        self.assertEqual((payload["canvas_width"], payload["canvas_height"]), (768, 1344))
        self.assertEqual(
            list(payload["structured_prompt"]),
            [
                "integrated_multimodal_description",
                "overall_soundscape",
                "non_diegetic_music",
            ],
        )
        self.assertIn("overall_soundscape: Steady rain", payload["prompt"])

    async def test_legacy_prompt_field_still_submits_h3(self):
        request = FakeRequest(self.manager, {"prompt": "A tiny boat crosses a cup."})
        response = await queue_web.create_job(request)
        payload = json.loads(response.body)
        self.assertEqual(payload["model"], "minimax-h3")
        self.assertEqual(payload["duration_seconds"], 5)
        self.assertEqual(
            payload["prompt"],
            "integrated_multimodal_description: A tiny boat crosses a cup.",
        )

    async def test_rejects_ltx_submission(self):
        request = FakeRequest(
            self.manager,
            {"prompt": "A tiny boat crosses a cup.", "model": "ltx-2.5"},
        )
        with self.assertRaises(web.HTTPBadRequest):
            await queue_web.create_job(request)


class QueueStateTests(unittest.TestCase):
    def test_regenerate_uses_new_random_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manager = queue_web.QueueManager(output, 56.0)
            manager.upload_directory.mkdir()
            original = manager.add(
                "integrated_multimodal_description: [Shot 1] A paper bird flies.",
                {
                    "integrated_multimodal_description": (
                        "[Shot 1] A paper bird flies."
                    )
                },
                "text",
                123,
                10,
                "minimax-h3",
                768,
                "16:9",
                None,
            )

            with mock.patch.object(queue_web, "_random_seed", return_value=456):
                regenerated = manager.regenerate(original.id)

            self.assertEqual(regenerated.seed, 456)
            self.assertNotEqual(regenerated.seed, original.seed)
            self.assertEqual(regenerated.prompt, original.prompt)
            self.assertEqual(regenerated.structured_prompt, original.structured_prompt)
            self.assertEqual(regenerated.duration_seconds, original.duration_seconds)
            self.assertEqual(regenerated.resolution, original.resolution)
            self.assertEqual(regenerated.aspect_ratio, original.aspect_ratio)

    def test_legacy_ltx_history_loads_but_cannot_regenerate(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            state = {
                "version": 1,
                "jobs": [
                    {
                        "id": "legacy123456",
                        "prompt": "An old prompt",
                        "mode": "text",
                        "seed": 1,
                        "duration_seconds": 5,
                        "model": "ltx-2.5",
                        "output_filename": "legacy.mp4",
                        "upload_filename": None,
                        "image_content_type": None,
                        "status": "completed",
                        "error": None,
                        "created_at": 1.0,
                        "started_at": 2.0,
                        "finished_at": 3.0,
                    }
                ],
            }
            (output / queue_web.STATE_FILENAME).write_text(json.dumps(state))
            manager = queue_web.QueueManager(output, 56.0)
            manager.upload_directory.mkdir()
            manager._load_state()
            job = manager.jobs["legacy123456"]
            self.assertEqual(job.model, "ltx-2.5")
            self.assertEqual(job.as_dict()["model_label"], "LTX-2.5 (legacy)")
            with self.assertRaises(web.HTTPConflict):
                manager.regenerate(job.id)


if __name__ == "__main__":
    unittest.main()
