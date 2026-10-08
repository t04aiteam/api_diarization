"""
Optional pyannote.audio diarization backend (speaker segmentation only, no
transcription — `transcribes = False`).

Disabled by default per the HDSD doc: `pyannote/speaker-diarization-3.1` is a
gated model not present in the offline cache. Enabling it needs internet + a
valid HF_TOKEN with the model's terms accepted on huggingface.co.
`pyannote.audio` is imported lazily in `load()` so the app can start fine
even when the package isn't installed while this backend stays disabled.
"""

import os
from typing import Optional

from models.base import Backend, DiarizeResult, Segment

MODEL_ID = "pyannote/speaker-diarization-3.1"


class PyannoteBackend(Backend):
    name = "pyannote"
    model_id = MODEL_ID
    transcribes = False

    def __init__(self):
        self._pipeline = None

    def load(self) -> None:
        hf_token = os.getenv("HF_TOKEN") or None
        if not hf_token:
            raise RuntimeError(
                "pyannote backend requires HF_TOKEN (gated model) and internet access "
                "to fetch it the first time; it cannot run fully offline."
            )
        from pyannote.audio import Pipeline

        self._pipeline = Pipeline.from_pretrained(self.model_id, use_auth_token=hf_token)

    def run(self, audio_path: str, hotwords: Optional[str], max_new_tokens: int) -> DiarizeResult:
        # pyannote.audio's Pipeline accepts a file path directly and does its
        # own decoding/resampling; no transcription support, so hotwords are
        # irrelevant here.
        diarization = self._pipeline(audio_path)

        segments = []
        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            digits = "".join(filter(str.isdigit, speaker_label))
            speaker_id = int(digits) if digits else 0
            segments.append(
                Segment(
                    start_time=round(turn.start, 2),
                    end_time=round(turn.end, 2),
                    speaker_id=speaker_id,
                    text=None,  # pyannote is diarization-only, no transcription
                )
            )
        return DiarizeResult(segments=segments)
