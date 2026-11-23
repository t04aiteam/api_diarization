"""VibeVoice lifecycle checks using fake backends without model weights."""

import asyncio
import importlib
import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from models import registry
from models.base import DiarizeResult


class VibeVoiceIdleTests(unittest.IsolatedAsyncioTestCase):
    def setup_backends(self, default_model):
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.addCleanup(self.release.set)
        self.enterContext(mock.patch.dict(os.environ, {
            "DEFAULT_MODEL": default_model,
            "ENABLED_MODELS": "vibevoice,pyannote",
            "MODEL_IDLE_SECONDS": "1",
        }))
        importlib.reload(registry)
        self.enterContext(mock.patch.object(registry, "_build", side_effect=self.build))

    def build(self, name):
        test = self

        class Backend:
            model_id = name

            def load(self):
                test.calls.append((name, "load"))

            def unload(self):
                test.calls.append((name, "unload"))

            def run(self, path, _hotwords, _max_tokens):
                test.calls.append((name, path))
                if path == "blocked":
                    test.started.set()
                    if not test.release.wait(5):
                        raise RuntimeError("test did not release inference")
                return DiarizeResult()

        return Backend()

    async def wait_for_unload(self, count):
        async with asyncio.timeout(4):
            while self.calls.count(("vibevoice", "unload")) < count:
                await asyncio.sleep(0.025)

    async def test_inference_and_queued_work_prevent_idle_unload(self):
        self.setup_backends("vibevoice")
        async with registry.lifespan(None):
            active = asyncio.create_task(registry.run("vibevoice", "blocked", None, 10))
            try:
                self.assertTrue(await asyncio.to_thread(self.started.wait, 2))
                queued = asyncio.create_task(registry.run("vibevoice", "queued", None, 10))
                active.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await active
                await asyncio.sleep(1.1)
                self.assertEqual(self.calls, [("vibevoice", "load"), ("vibevoice", "blocked")])
            finally:
                self.release.set()
            self.assertEqual((await queued)[0], "vibevoice")
            await self.wait_for_unload(1)
            self.assertEqual((await registry.run("vibevoice", "new", None, 10))[0], "vibevoice")
            await self.wait_for_unload(2)
            self.assertEqual(self.calls, [
                ("vibevoice", "load"), ("vibevoice", "blocked"),
                ("vibevoice", "queued"), ("vibevoice", "unload"),
                ("vibevoice", "load"), ("vibevoice", "new"), ("vibevoice", "unload"),
            ])

    async def test_pyannote_does_not_load_or_keep_vibevoice_warm(self):
        self.setup_backends("pyannote")
        async with registry.lifespan(None):
            self.assertEqual((await registry.run("pyannote", "remote", None, 10))[0], "pyannote")
            self.assertEqual(self.calls, [("pyannote", "load"), ("pyannote", "remote")])
            await registry.run("vibevoice", "file", None, 10)
            await self.wait_for_unload(1)
            await registry.run("pyannote", "remote again", None, 10)
            self.assertEqual(self.calls.count(("vibevoice", "load")), 1)
            self.assertEqual(self.calls.count(("pyannote", "load")), 1)
            self.assertNotIn(("pyannote", "unload"), self.calls)

    async def test_queued_vibevoice_leaves_executor_capacity_for_pyannote(self):
        self.setup_backends("vibevoice")
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=2))
        async with registry.lifespan(None):
            active = asyncio.create_task(registry.run("vibevoice", "blocked", None, 10))
            queued = None
            try:
                self.assertTrue(await asyncio.to_thread(self.started.wait, 2))
                queued = asyncio.create_task(registry.run("vibevoice", "queued", None, 10))
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                remote = await asyncio.wait_for(registry.run("pyannote", "remote", None, 10), 1)
                self.assertEqual(remote[0], "pyannote")
                self.assertFalse(active.done())
                self.assertFalse(queued.done())
            finally:
                self.release.set()
                await active
                if queued is not None:
                    await queued
