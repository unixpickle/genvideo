import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import psutil

from aiohttp import web
import web as queue_web


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.manager = queue_web.QueueManager(Path(self.temporary.name), 56)
        self.manager.upload_directory.mkdir()
        self.manager.work_directory.mkdir()

    async def asyncTearDown(self):
        await self.manager.close()
        self.temporary.cleanup()

    def add(self, priority="medium", image=None):
        return self.manager.add("A bird", {}, "image" if image else "text", 2**62 + 123,
                                5, "minimax-h3", 512, "1:1", image, priority=priority)

    async def until(self, predicate):
        async with asyncio.timeout(5):
            while not predicate():
                await asyncio.sleep(.01)

    def start_fake_worker(self, *, spawn_delay=0):
        original_spawn = asyncio.create_subprocess_exec

        async def spawn(*args, **kwargs):
            if spawn_delay:
                await asyncio.sleep(spawn_delay)
            # A real disposable subprocess with a descendant, exercising group kill.
            directory = Path(args[2])
            code = '''import json, pathlib, subprocess, sys, time
p = pathlib.Path(sys.argv[1])
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])
(p / 'child.pid').write_text(str(child.pid))
(p / 'sampler.pt').write_text('saved step 2')
(p / 'progress.tmp').write_text(json.dumps({'stage': 'Diffusion', 'step': 2, 'total': 4}))
(p / 'progress.tmp').replace(p / 'progress.json')
time.sleep(300)
'''
            return await original_spawn(sys.executable, "-c", code, str(directory), **kwargs)

        patcher = mock.patch.object(queue_web.asyncio, "create_subprocess_exec", side_effect=spawn)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.manager.worker_task = asyncio.create_task(self.manager._worker())

    async def test_priority_order_moves_resume_and_restart(self):
        low = self.add("low")
        a, b = self.add(), self.add()
        high = self.add("high")
        self.assertEqual(self.manager._queued(), [high, a, b, low])
        self.manager.edit(b.id, {"move": "up"})
        self.assertEqual(self.manager._queued(), [high, b, a, low])
        await self.manager.pause(b.id)
        self.manager.resume(b.id)
        self.assertEqual(self.manager._queued(), [high, a, b, low])
        await self.manager.pause(low.id)
        a.status = "running"
        self.manager._save_state()
        restored = queue_web.QueueManager(self.manager.output_directory, 56)
        restored._load_state()
        self.assertEqual([job.id for job in restored._queued()], [high.id, a.id, b.id])
        self.assertEqual(restored.jobs[low.id].status, "paused")
        self.assertEqual(restored.jobs[a.id].as_dict()["seed_text"], str(a.seed))

    async def test_high_priority_preempts_and_pause_resume_preserves_checkpoint(self):
        low = self.add("low")
        self.start_fake_worker()
        await self.until(lambda: low.progress.get("step") == 2)
        low_process = self.manager.process
        child_pid = int((self.manager.work_directory / low.id / "child.pid").read_text())
        high = self.add("high")
        await self.until(lambda: high.progress.get("step") == 2)
        self.assertEqual(low.status, "queued")
        self.assertIsNotNone(low_process.returncode)
        self.assertTrue(not psutil.pid_exists(child_pid)
                        or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE)
        saved = self.manager.work_directory / low.id / "sampler.pt"
        self.assertEqual(saved.read_text(), "saved step 2")
        await self.manager.pause(high.id)
        self.assertEqual(high.status, "paused")
        await self.until(lambda: low.status == "running")
        await self.manager.pause(low.id)
        self.assertEqual(saved.read_text(), "saved step 2")
        self.manager.resume(high.id)
        await self.until(lambda: high.status == "running")
        await self.manager.remove(high.id)
        self.assertNotIn(high.id, self.manager.jobs)
        self.assertFalse((self.manager.work_directory / high.id).exists())

    async def test_equal_priority_does_not_preempt_but_priority_edit_does(self):
        a = self.add()
        self.start_fake_worker()
        await self.until(lambda: a.progress.get("step") == 2)
        b = self.add()
        await asyncio.sleep(.3)
        self.assertEqual(a.status, "running")
        self.manager.edit(b.id, {"priority": "high"})
        await self.until(lambda: b.progress.get("step") == 2)
        self.assertEqual(a.status, "queued")

    async def test_cancel_during_spawn_deletes_everything_and_kills_process(self):
        image = self.manager.upload_directory / "image.upload"
        image.write_bytes(b"image")
        job = self.add(image=image)
        job.output_path.write_bytes(b"partial output")
        self.start_fake_worker(spawn_delay=.15)
        await self.until(lambda: self.manager.active_job is job)
        await self.manager.remove(job.id)
        self.assertIsNone(self.manager.process)
        self.assertFalse(image.exists())
        self.assertFalse(job.output_path.exists())
        self.assertFalse((self.manager.work_directory / job.id).exists())
        self.assertEqual(json.loads(self.manager.state_path.read_text())["jobs"], [])

    async def test_shutdown_during_spawn_requeues_job(self):
        job = self.add()
        self.start_fake_worker(spawn_delay=.1)
        await self.until(lambda: self.manager.active_job is job)
        await self.manager.close()
        self.assertEqual(job.status, "queued")
        self.assertIsNone(self.manager.process)

    async def test_completed_worker_moves_to_history_and_removes_checkpoint(self):
        job = self.add()
        self.start_fake_worker()
        await self.until(lambda: job.progress.get("step") == 2)
        job.output_path.write_bytes(b"mp4")
        directory = self.manager.work_directory / job.id
        (directory / "result.json").write_text(json.dumps({"ok": True}))
        await self.until(lambda: self.manager.active_job is None)
        self.assertEqual(job.status, "completed")
        self.assertFalse(directory.exists())
        self.assertIsNotNone(job.finished_at)
        self.assertTrue(job.as_dict()["media_url"])

    async def test_invalid_edit_does_not_change_job(self):
        job = self.add()
        for change in ({"priority": "urgent"}, {"move": "first"}, {"priority": []}, {}):
            with self.assertRaises(web.HTTPBadRequest):
                self.manager.edit(job.id, change)
        self.assertEqual(job.priority, "medium")
