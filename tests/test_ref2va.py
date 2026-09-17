import asyncio
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import aiohttp
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

import generation
import ref2va
import web as queue_web


def fields():
    return dict(subject_definitions="<Subject 1> is the bird in <Picture 1>.",
                summary="[reference generation] <Subject 1> flies.",
                retention_analysis="<Subject 1>: fully_preserved - retain its feathers.",
                detailed_description="Naturalistic style. [Shot 1] <Subject 1> flies above a lake.",
                overall_soundscape="Wind.", non_diegetic_music="N/A")


def png():
    output = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(output, format="PNG")
    return output.getvalue()


class ReferenceTests(unittest.TestCase):
    def test_installed_conditioning_node_with_mixed_media(self):
        subprocess.run([sys.executable, str(Path(__file__).with_name("ref2va_conditioning_check.py"))],
                       check=True, capture_output=True, timeout=90)

    def test_labels_follow_runtime_order_and_soundtracks_are_opt_in(self):
        refs = [{"id": "a", "kind": "audio"}, {"id": "v", "kind": "video"},
                {"id": "i", "kind": "image"}, {"id": "v2", "kind": "video", "use_audio": True}]
        self.assertEqual(ref2va.reference_labels(refs), {
            "i": ["<Picture 1>"], "v": ["<Video 1>"], "v2": ["<Video 2>", "<Audio 1>"], "a": ["<Audio 2>"]})

    def test_prompt_requires_complete_sections_and_existing_labels(self):
        refs = [{"id": "i", "kind": "image"}]
        prompt = ref2va.build_ref2va_prompt(fields(), refs)
        self.assertEqual([block.split(":")[0] for block in prompt.split("\n\n")], list(ref2va.REF_FIELDS))
        for values in (fields() | {"summary": ""}, fields() | {"summary": "<Audio 1>"},
                       fields() | {"detailed_description": "<Subject 2> walks"}):
            with self.assertRaises(ValueError):
                ref2va.build_ref2va_prompt(values, refs)

    def test_reference_limits_and_duplicate_ids(self):
        for value in ([], [{"id": "../x", "kind": "image"}],
                      [{"id": "x", "kind": "image"}] * 2,
                      [{"id": str(i), "kind": "video", "use_audio": True} for i in range(3)] + [{"id": "a", "kind": "audio"}]):
            with self.assertRaises(ValueError):
                ref2va.validate_references(value)
        with self.assertRaises(ValueError):
            ref2va.validate_durations([{"kind": "audio", "duration": 8}] * 2)

    def test_workflow_uses_ref_model_without_fl2va_lora_and_resumes(self):
        refs = [{"id": "i", "kind": "image", "path": "/tmp/a.png"}]
        graph = generation._workflow(None, "prompt", 3, model=ref2va.REF2VA_MODEL, references=refs)
        self.assertEqual(graph["1"]["inputs"]["unet_name"], ref2va.REF2VA_WEIGHTS)
        self.assertFalse(any("lora_name" in n["inputs"] for n in graph.values()))
        self.assertEqual(graph["9"]["inputs"]["steps"], 20)
        self.assertEqual(json.loads(graph["6"]["inputs"]["references_json"]), refs)
        self.assertEqual(graph["6"]["inputs"]["audio_vae"], ["4", 0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "conditioning.pt").touch()
            restored = generation._resumable_workflow(graph, path)
            self.assertEqual(restored["6"]["class_type"], "GenVideoLoadConditioning")
            self.assertNotIn("2", restored)
        with self.assertRaises(generation.GenerationError):
            generation._workflow("start.png", "prompt", 3, model=ref2va.REF2VA_MODEL, references=refs)


class ReferenceAPITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        async def idle(manager):
            await asyncio.Event().wait()
        self.patch = mock.patch.object(queue_web.QueueManager, "_worker", idle)
        self.patch.start()
        self.client = TestClient(TestServer(queue_web.make_app(Path(self.tmp.name))))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        self.patch.stop()
        self.tmp.cleanup()

    def data(self, refs=None, payload=None, **changes):
        refs = refs if refs is not None else [{"id": "i", "kind": "image", "name": "bird.png"}]
        data = aiohttp.FormData()
        for name, value in (fields() | {"mode": "ref2va", "model": ref2va.REF2VA_MODEL,
                                        "references": json.dumps(refs), "reference_builder": '{"test": true}'} | changes).items():
            data.add_field(name, value)
        data.add_field("reference_i", png() if payload is None else payload, filename="bird.png", content_type="image/png")
        return data

    async def test_reference_survives_restart_copy_retry_and_removal(self):
        response = await self.client.post("/api/jobs", data=self.data())
        self.assertEqual(response.status, 201, await response.text())
        job = await response.json()
        self.assertEqual(job["references"][0]["labels"], ["<Picture 1>"])
        self.assertEqual(await (await self.client.get(job["references"][0]["url"])).read(), png())
        manager = self.client.server.app[queue_web.MANAGER_KEY]
        reloaded = queue_web.QueueManager(Path(self.tmp.name), 56)
        reloaded._load_state()
        self.assertEqual(reloaded.jobs[job["id"]].as_dict(), job)
        clone = await (await self.client.post(f'/api/jobs/{job["id"]}/regenerate')).json()
        await self.client.delete(f'/api/jobs/{job["id"]}')
        self.assertEqual(await (await self.client.get(clone["references"][0]["url"])).read(), png())
        record = manager.jobs[clone["id"]]
        record.status = "failed"
        self.assertEqual((await self.client.post(f'/api/jobs/{record.id}/retry')).status, 200)
        (manager.upload_directory / record.references[0]["filename"]).unlink()
        record.status = "failed"
        self.assertEqual((await self.client.post(f'/api/jobs/{record.id}/retry')).status, 409)
        await self.client.delete(f'/api/jobs/{record.id}')
        self.assertFalse(list(manager.upload_directory.iterdir()))

    async def test_invalid_requests_cleanup_uploads(self):
        broken = bytearray(png())
        broken[-5] ^= 255
        for data in (self.data(model="minimax-h3"), self.data(payload=b"not media"), self.data(payload=bytes(broken)),
                     self.data(summary="<Picture 2>"), self.data(refs=[]), self.data(reference_builder="[]")):
            response = await self.client.post("/api/jobs", data=data)
            self.assertEqual(response.status, 400, await response.text())
            self.assertFalse(list((Path(self.tmp.name) / ".uploads").iterdir()))
        self.assertEqual((await self.client.post("/api/jobs", data={"model": ref2va.REF2VA_MODEL, "prompt": "x"})).status, 400)

    async def test_video_and_audio_probe_and_numbering(self):
        video = Path(self.tmp.name) / "video.mp4"
        await asyncio.to_thread(subprocess.run, ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=blue:s=64x64:r=24",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=32000", "-t", "2", "-c:v", "libx264", "-c:a", "aac", str(video)], check=True)
        refs = [{"id": "i", "kind": "video", "use_audio": True, "name": "video.mp4"}]
        values = {k: v.replace("<Picture 1>", "<Video 1>") for k, v in fields().items()}
        response = await self.client.post("/api/jobs", data=self.data(refs=refs, payload=video.read_bytes(), **values))
        self.assertEqual(response.status, 201, await response.text())
        job = await response.json()
        self.assertEqual(job["references"][0]["labels"], ["<Video 1>", "<Audio 1>"])
        self.assertEqual(job["references"][0]["width"], 64)
