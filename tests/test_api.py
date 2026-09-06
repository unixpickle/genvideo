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
