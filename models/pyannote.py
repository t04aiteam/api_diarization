"""
pyannote diarization backend (speaker segmentation only, no transcription:
`transcribes = False`, segment text is null).

A client of ai-audio-jobs' `pyannote_server.py` (222:6426) instead of loading
pyannote here: one pyannote setup for the job API and for direct callers
(offline weights without a token, clustering threshold 0.55, speaker counts
that never merge people), and no pyannote packages in diar-env. Enable with
ENABLED_MODELS=vibevoice,pyannote and PYANNOTE_URL=http://127.0.0.1:6426.
PYANNOTE_TIMEOUT (seconds, default 10800) covers a 60-min file on CPU.
"""

import json
import os
import urllib.error
import urllib.request
import uuid
from typing import Optional

from models.base import Backend, DiarizeResult, Segment

MODEL_ID = "pyannote/speaker-diarization-3.1"


class PyannoteBackend(Backend):
    name = "pyannote"
    model_id = MODEL_ID
    transcribes = False
    remote = True

    def __init__(self):
        self._url = None

    def load(self) -> None:
        url = os.getenv("PYANNOTE_URL", "").rstrip("/")
        if not url:
            raise RuntimeError(
                "pyannote backend needs PYANNOTE_URL (the pyannote service, e.g. "
                "http://127.0.0.1:6426)"
            )
        try:
            urllib.request.urlopen(f"{url}/health", timeout=10).close()
        except OSError as exc:
            raise RuntimeError(
                f"pyannote service {url} is not reachable: {exc}"
            ) from exc
        self._url = url

    def run(
        self, audio_path: str, hotwords: Optional[str], max_new_tokens: int
    ) -> DiarizeResult:
        # hotwords and max_new_tokens only steer transcription, which pyannote does not do.
        with open(audio_path, "rb") as fh:
            data = fh.read()
        boundary = uuid.uuid4().hex
        name = os.path.basename(audio_path)
        body = (
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                f'filename="{name}"\r\nContent-Type: application/octet-stream\r\n\r\n'
            ).encode()
            + data
            + f"\r\n--{boundary}--\r\n".encode()
        )
        req = urllib.request.Request(
            f"{self._url}/diarize",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        timeout = float(os.getenv("PYANNOTE_TIMEOUT", "10800"))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                answer = json.load(resp)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"pyannote service answered {exc.code}: {exc.read().decode(errors='replace')}"
            ) from exc
        return DiarizeResult(
            segments=[
                Segment(
                    start_time=s["start_time"],
                    end_time=s["end_time"],
                    speaker_id=s["speaker_id"],
                    text=None,
                )
                for s in answer["segments"]
            ],
            warning=answer.get("warning"),
        )
