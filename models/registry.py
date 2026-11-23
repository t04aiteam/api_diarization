"""Backend registry: which ASR/diarization backends exist, which are enabled,
and which is the default — all driven by .env, per the HDSD doc.

Only the `vibevoice` backend is enabled by default. `pyannote` is a client of
the pyannote service (ai-audio-jobs pyannote_server.py): add it to
ENABLED_MODELS and set PYANNOTE_URL.
"""

import asyncio
import os
import threading
from contextlib import asynccontextmanager

from models.idle import IdleModel

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


def _get_backend(name: str):
    """Get the current backend instance, loading it if necessary."""
    if not is_enabled(name):
        raise KeyError(f"backend '{name}' is not enabled")
    lock = _LOAD_LOCKS.setdefault(name, threading.Lock())
    with lock:
        if name not in _INSTANCES:
            backend = _build(name)
            backend.load()
            _INSTANCES[name] = backend
    return _INSTANCES[name]


def _unload_vibevoice() -> None:
    _INSTANCES.pop("vibevoice").unload()


_vibevoice = IdleModel(
    lambda: _get_backend("vibevoice"),
    _unload_vibevoice,
    int(os.getenv("MODEL_IDLE_SECONDS", "0")),
)


@asynccontextmanager
async def lifespan(_app):
    if DEFAULT_MODEL != "vibevoice":
        await asyncio.to_thread(_get_backend, DEFAULT_MODEL)
    async with _vibevoice.lifespan(preload=DEFAULT_MODEL == "vibevoice"):
        yield


def _run_backend(name, audio_path, hotwords, max_new_tokens):
    backend = _get_backend(name)
    if getattr(backend, "remote", False):
        result = backend.run(audio_path, hotwords, max_new_tokens)
    else:
        # Local inference is serialized. Remote pyannote queues independently.
        with inference_lock():
            result = backend.run(audio_path, hotwords, max_new_tokens)
    return backend.model_id, result


async def run(name, audio_path, hotwords, max_new_tokens):
    args = (name, audio_path, hotwords, max_new_tokens)
    if name == "vibevoice":
        return await _vibevoice.run(_run_backend, *args)
    return await asyncio.to_thread(_run_backend, *args)


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
