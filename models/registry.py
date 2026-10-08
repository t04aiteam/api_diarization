"""Backend registry: which ASR/diarization backends exist, which are enabled,
and which is the default — all driven by .env, per the HDSD doc.

Only the `vibevoice` backend is enabled by default. `pyannote` is a client of
the pyannote service (ai-audio-jobs pyannote_server.py): add it to
ENABLED_MODELS and set PYANNOTE_URL.
"""

import os
import threading

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "vibevoice")
_ENABLED = {b.strip() for b in os.getenv("ENABLED_MODELS", "vibevoice").split(",") if b.strip()}

ALL_BACKENDS = {
    "vibevoice": {"model_id": "microsoft/VibeVoice-ASR", "transcribes": True},
    "pyannote": {"model_id": "pyannote/speaker-diarization-3.1", "transcribes": False},
}

_INSTANCES = {}
_LOAD_LOCKS = {}
_INFERENCE_LOCK = threading.Lock()  # only one inference request on the GPU at a time


def _build(name: str):
    if name == "vibevoice":
        from models.vibevoice import VibeVoiceBackend

        return VibeVoiceBackend()
    if name == "pyannote":
        from models.pyannote import PyannoteBackend

        return PyannoteBackend()
    raise KeyError(name)


def is_enabled(name: str) -> bool:
    return name in _ENABLED and name in ALL_BACKENDS


def get_backend(name: str):
    """Lazily load (once) and return the backend instance for `name`."""
    if not is_enabled(name):
        raise KeyError(f"backend '{name}' is not enabled")
    lock = _LOAD_LOCKS.setdefault(name, threading.Lock())
    with lock:
        if name not in _INSTANCES:
            backend = _build(name)
            backend.load()
            _INSTANCES[name] = backend
    return _INSTANCES[name]


def preload_default() -> None:
    """Load the default backend at startup so the first request isn't slow."""
    get_backend(DEFAULT_MODEL)


def models_status() -> dict:
    return {
        name: {
            "model_id": meta["model_id"],
            "enabled": is_enabled(name),
            "transcribes": meta["transcribes"],
        }
        for name, meta in ALL_BACKENDS.items()
    }


def enabled_models_simple() -> dict:
    return {name: meta["model_id"] for name, meta in ALL_BACKENDS.items() if is_enabled(name)}


def inference_lock() -> threading.Lock:
    return _INFERENCE_LOCK
