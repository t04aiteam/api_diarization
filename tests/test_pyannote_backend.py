"""Gate tests: model=pyannote on POST /diarize answers through the pyannote service.

The pyannote backend is a client of ai-audio-jobs' pyannote_server (222:6426):
one pyannote setup (offline weights, threshold 0.55, speaker-count trim) for
both the job API and direct callers, and no pyannote packages in diar-env.

Failure modes pinned here:
- the upload does not reach the pyannote service as multipart ``file``;
- segments lose their times or speaker ids, or text is anything but null
  (pyannote does not transcribe);
- the answer says model "vibevoice" or the wrong model_id;
- the backend loads without PYANNOTE_URL, or with a dead service, instead of
  failing at load with a clear message;
- a pyannote service error becomes a 200 instead of a 500 with its detail;
- with pyannote enabled, a pyannote request loads VibeVoice (18 GB);
- a request for a backend not in ENABLED_MODELS gets past the 400;
- a pyannote request holds the VibeVoice GPU lock while the remote service
  works (up to an hour on a 60-min file), stalling VibeVoice callers.

Run with diar-env (needs fastapi + httpx, no GPU, no weights):
    diar-env/bin/python -m unittest discover -s tests -v
"""

import importlib
import json
import os
import sys
import threading
import unittest
import wave
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ANSWER = {
    "filename": "audio.wav",
    "model": "pyannote",
    "model_id": "pyannote/speaker-diarization-3.1",
    "duration_seconds": 2.0,
    "num_speakers": 2,
    "speakers": [0, 1],
    "segments": [
        {"start_time": 0.0, "end_time": 0.8, "speaker_id": 1, "text": None},
        {"start_time": 0.9, "end_time": 2.0, "speaker_id": 0, "text": None},
    ],
    "raw_text": None,
    "warning": None,
}


def wav_bytes(seconds: float = 2.0) -> bytes:
    """A silent 16 kHz mono wav.

    Args:
        seconds: Length.

    Returns:
        The file bytes.
    """
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * int(16000 * seconds))
    return buf.getvalue()


class FakePyannote:
    """A pyannote_server stand-in that records what it was sent."""

    def __init__(self, status: int = 200, body: dict | None = None):
        self.status, self.body, self.uploads = status, body or ANSWER, []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def reply(self, status, body):
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                self.reply(200, {"status": "ok"})

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                msg = BytesParser(policy=email_policy).parsebytes(
                    b"Content-Type: "
                    + self.headers["Content-Type"].encode()
                    + b"\r\n\r\n"
                    + body
                )
                for part in msg.iter_parts():
                    if part.get_param("name", header="content-disposition") == "file":
                        fake.uploads.append(
                            (
                                self.path,
                                part.get_filename(),
                                part.get_payload(decode=True),
                            )
                        )
                self.reply(fake.status, fake.body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class PyannoteBackendTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakePyannote()
        self.addCleanup(self.fake.close)

    def backend(self, url):
        from models.pyannote import PyannoteBackend

        with mock.patch.dict(os.environ, {"PYANNOTE_URL": url}):
            b = PyannoteBackend()
            b.load()
        return b

    def test_the_upload_reaches_the_service_and_segments_come_back(self):
        data = wav_bytes()
        path = os.path.join(self.enterContext(_tmpdir()), "talk.wav")
        with open(path, "wb") as fh:
            fh.write(data)
        result = self.backend(self.fake.url).run(path, "tên riêng", 32768)
        self.assertEqual(self.fake.uploads, [("/diarize", "talk.wav", data)])
        self.assertEqual([s.__dict__ for s in result.segments], ANSWER["segments"])
        self.assertIsNone(result.raw_text)

    def test_load_needs_a_live_service(self):
        from models.pyannote import PyannoteBackend

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PYANNOTE_URL", None)
            with self.assertRaisesRegex(RuntimeError, "PYANNOTE_URL"):
                PyannoteBackend().load()
        with self.assertRaisesRegex(RuntimeError, "not reachable"):
            self.backend("http://127.0.0.1:9")

    def test_a_service_error_raises_with_its_detail(self):
        bad = FakePyannote(500, {"detail": "Inference failed: boom"})
        self.addCleanup(bad.close)
        path = os.path.join(self.enterContext(_tmpdir()), "a.wav")
        with open(path, "wb") as fh:
            fh.write(wav_bytes())
        with self.assertRaisesRegex(RuntimeError, "500.*boom"):
            self.backend(bad.url).run(path, None, 1)


def client(enabled: str):
    """A test app with the diarize router, registry re-read for ENABLED_MODELS.

    Args:
        enabled: ENABLED_MODELS value.

    Returns:
        (TestClient, registry module).
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from models import registry
    from routes.diarize import router

    with mock.patch.dict(os.environ, {"ENABLED_MODELS": enabled}):
        importlib.reload(registry)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app), registry


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakePyannote()
        self.addCleanup(self.fake.close)
        self.enterContext(mock.patch.dict(os.environ, {"PYANNOTE_URL": self.fake.url}))
        self.client, registry = client("vibevoice,pyannote")
        self.registry = registry
        self.built = []
        real_build = registry._build
        self.enterContext(
            mock.patch.object(
                registry,
                "_build",
                side_effect=lambda name: self.built.append(name) or real_build(name),
            )
        )

    def post(self, model):
        return self.client.post(
            "/diarize",
            files={"file": ("talk.wav", wav_bytes(), "audio/wav")},
            data={"model": model},
        )

    def test_model_pyannote_answers_from_the_service_only(self):
        r = self.post("pyannote")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["model"], "pyannote")
        self.assertEqual(body["model_id"], "pyannote/speaker-diarization-3.1")
        self.assertEqual(body["segments"], ANSWER["segments"])
        self.assertEqual((body["num_speakers"], body["speakers"]), (2, [0, 1]))
        self.assertEqual(self.built, ["pyannote"])

    def test_models_lists_pyannote_enabled(self):
        body = self.client.get("/models").json()
        self.assertEqual(body["enabled"], ["pyannote", "vibevoice"])
        self.assertFalse(body["models"]["pyannote"]["transcribes"])

    def test_a_service_error_is_a_500_with_its_detail(self):
        self.fake.status, self.fake.body = 500, {"detail": "Inference failed: boom"}
        r = self.post("pyannote")
        self.assertEqual(r.status_code, 500)
        self.assertIn("boom", r.json()["detail"])

    def test_a_pyannote_request_leaves_the_vibevoice_lock_free(self):
        seen = []
        from models.pyannote import PyannoteBackend

        real_run = PyannoteBackend.run

        def run(backend, *args):
            lock = self.registry.inference_lock()
            seen.append(lock.acquire(blocking=False))
            if seen[-1]:
                lock.release()
            return real_run(backend, *args)

        with mock.patch("models.pyannote.PyannoteBackend.run", run):
            self.assertEqual(self.post("pyannote").status_code, 200)
        self.assertEqual(seen, [True])

    def test_a_disabled_backend_is_still_a_400(self):
        self.client, _ = client("vibevoice")
        r = self.post("pyannote")
        self.assertEqual(r.status_code, 400)
        self.assertIn("not enabled", r.json()["detail"])


def _tmpdir():
    import tempfile

    return tempfile.TemporaryDirectory()


if __name__ == "__main__":
    unittest.main()
