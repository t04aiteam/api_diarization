"""Shared backend interface for ASR + speaker-diarization models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Segment:
    start_time: float
    end_time: float
    speaker_id: int
    text: Optional[str] = None


@dataclass
class DiarizeResult:
    segments: List[Segment] = field(default_factory=list)
    raw_text: Optional[str] = None
    warning: Optional[str] = None


class Backend(ABC):
    """One ASR/diarization backend (e.g. vibevoice, pyannote)."""

    name: str
    model_id: str
    transcribes: bool

    @abstractmethod
    def load(self) -> None:
        """Load weights into memory/GPU. Called once, lazily, on first use."""

    @abstractmethod
    def run(self, audio_path: str, hotwords: Optional[str], max_new_tokens: int) -> DiarizeResult:
        """Run inference on the audio file at `audio_path`.

        `hotwords` is a free-text hint string (comma-separated terms/names,
        or None) — each backend decides whether/how to use it.
        """
