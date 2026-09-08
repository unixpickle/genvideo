import asyncio
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import generation
import web as queue_web


class APITests(unittest.IsolatedAsyncioTestCase):
    async def test_frame_combinations_survive_restart_regenerate_and_removal(self):
        for first, last in ((True, False), (False, True), (True, True)):
            with self.subTest(first=first, last=last):
                data = aiohttp.FormData()
                for key, value in {"mode": "image", "prompt": "A bird lands.",
                                   "model": "minimax-h3-base", "duration": "5"}.items():
                    data.add_field(key, value)
                if first:
                    data.add_field("image", b"start bytes", filename="start.png", content_type="image/png")
                if last:
                    data.add_field("last_image", b"end bytes", filename="end.jpg", content_type="image/jpeg")
                response = await self.client.post("/api/jobs", data=data)
                self.assertEqual(response.status, 201, await response.text())
                job = await response.json()
                self.assertEqual(job["has_first_frame"], first)
                self.assertEqual(job["has_last_frame"], last)
                self.assertEqual("at 0.00 seconds" in job["prompt"], first)
                if last:
                    self.assertIn(f"<Picture {2 if first else 1}> is fully referenced as the last frame", job["prompt"])

                reloaded = queue_web.QueueManager(Path(self.temporary.name), 56)
                reloaded._load_state()
                self.assertEqual(reloaded.jobs[job["id"]].status, "queued")
                self.assertEqual(reloaded.jobs[job["id"]].as_dict(), job)
                regenerated = await self.client.post(f'/api/jobs/{job["id"]}/regenerate')
                self.assertEqual(regenerated.status, 201)
                copy = await regenerated.json()
                self.assertEqual(copy["prompt"], job["prompt"])
                await self.client.delete(f'/api/jobs/{job["id"]}')
                for url_key, present, expected, mime in (
                    ("image_prompt_url", first, b"start bytes", "image/png"),
                    ("last_image_prompt_url", last, b"end bytes", "image/jpeg"),
                ):
                    if present:
                        image = await self.client.get(copy[url_key])
                        self.assertEqual(await image.read(), expected)
                        self.assertEqual(image.content_type, mime)
                        self.assertEqual((await self.client.get(job[url_key])).status, 404)
                    else:
                        self.assertIsNone(copy[url_key])
                await self.client.delete(f'/api/jobs/{copy["id"]}')
                self.assertEqual(list((Path(self.temporary.name) / ".uploads").iterdir()), [])

    async def test_missing_frames_and_invalid_second_upload(self):
        response = await self.client.post("/api/jobs", data={"mode": "image", "prompt": "A bird"})
        self.assertEqual(response.status, 400)
        data = aiohttp.FormData()
        data.add_field("mode", "image")
        data.add_field("prompt", "A bird")
        data.add_field("image", b"start", filename="start.png", content_type="image/png")
        data.add_field("last_image", b"invalid", filename="end.txt", content_type="text/plain")
        self.assertEqual((await self.client.post("/api/jobs", data=data)).status, 400)
        self.assertEqual(list((Path(self.temporary.name) / ".uploads").iterdir()), [])

    async def test_text_mode_discards_both_uploads(self):
        data = aiohttp.FormData()
        data.add_field("mode", "text")
        data.add_field("prompt", "A bird")
        for field in ("image", "last_image"):
            data.add_field(field, b"image", filename="frame.png", content_type="image/png")
        response = await self.client.post("/api/jobs", data=data)
        self.assertEqual(response.status, 201)
        job = await response.json()
        self.assertFalse(job["has_first_frame"])
        self.assertFalse(job["has_last_frame"])
        self.assertNotIn("<Picture", job["prompt"])
        self.assertEqual(list((Path(self.temporary.name) / ".uploads").iterdir()), [])

    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()

        async def idle_worker(manager):
            await asyncio.Event().wait()

        self.worker_patch = mock.patch.object(queue_web.QueueManager, "_worker", idle_worker)
        self.worker_patch.start()
        self.client = TestClient(TestServer(queue_web.make_app(Path(self.temporary.name))))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        self.worker_patch.stop()
        self.temporary.cleanup()

    async def test_submit_edit_pause_resume_cancel_image_job(self):
        data = aiohttp.FormData()
        for name, value in {"mode": "image", "prompt": "A bird flies.",
                            "priority": "low", "seed": "4611686018427388027"}.items():
            data.add_field(name, value)
        data.add_field("image", b"image bytes", filename="bird.png", content_type="image/png")
        response = await self.client.post("/api/jobs", data=data)
        self.assertEqual(response.status, 201)
        job = await response.json()
        self.assertEqual(job["seed_text"], "4611686018427388027")
        self.assertEqual(job["priority"], "low")
        endpoint = f'/api/jobs/{job["id"]}'
        image = await self.client.get(job["image_prompt_url"])
        self.assertEqual(await image.read(), b"image bytes")
        response = await self.client.patch(endpoint, json={"priority": "high"})
        self.assertEqual((await response.json())["priority"], "high")
        response = await self.client.post(endpoint + "/pause")
        self.assertEqual((await response.json())["status"], "paused")
        response = await self.client.post(endpoint + "/resume")
        self.assertEqual((await response.json())["status"], "queued")
        response = await self.client.get("/api/jobs")
        snapshot = await response.json()
        self.assertEqual(snapshot["jobs"][0]["queue_position"], 1)
        response = await self.client.delete(endpoint)
        self.assertEqual(response.status, 200)
        self.assertEqual((await self.client.get(job["image_prompt_url"])).status, 404)
        self.assertEqual((await (await self.client.get("/api/jobs")).json())["jobs"], [])

    async def test_invalid_priorities_and_edits_return_client_errors(self):
        response = await self.client.post("/api/jobs", data={"prompt": "A bird", "priority": "urgent"})
        self.assertEqual(response.status, 400)
        response = await self.client.post("/api/jobs", data={"prompt": "A bird"})
        endpoint = '/api/jobs/' + (await response.json())["id"]
        for value in (None, [], {"priority": []}, {"move": "first"}, {"move": []}):
            response = await self.client.patch(endpoint, data=json.dumps(value), headers={"Content-Type": "application/json"})
            self.assertEqual(response.status, 400)
        self.assertEqual((await self.client.post(endpoint + "/resume")).status, 409)
        self.assertEqual((await self.client.post("/api/jobs/missing/pause")).status, 404)


class ProgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_websocket_events_report_stage_and_step(self):
        async def websocket(request):
            socket = web.WebSocketResponse()
            await socket.prepare(request)
            for value in ({"type": "executing", "data": {"node": "11"}},
                          {"type": "progress", "data": {"node": "11", "value": 2, "max": 4}}):
                await socket.send_json(value)
            await socket.close()
            return socket

        app = web.Application()
        app.router.add_get("/ws", websocket)
        server = TestServer(app)
        await server.start_server()
        try:
            events = []
            await generation._progress_listener(str(server.make_url("/")).rstrip("/"), "test",
                                          {"11": "Diffusion"}, set(), threading.Event(),
                                          threading.Event(), events.append)
            self.assertEqual(events, [{"stage": "Diffusion", "step": None, "total": None},
                                      {"stage": "Diffusion", "step": 2, "total": 4}])
        finally:
            await server.close()
