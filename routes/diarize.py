"""Transcription + speaker-diarization endpoints (VibeVoice ASR / pyannote)."""

import asyncio
import os
import tempfile

from fastapi import APIRouter, Form, HTTPException, UploadFile

from models import registry
from models.audio_utils import MIN_AUDIO_SECONDS, decode_audio
from schemas.response import DiarizeResponse, ModelsResponse

router = APIRouter()

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", 500))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_AUDIO_SECONDS = int(os.getenv("MAX_AUDIO_SECONDS", 3600))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS_DEFAULT", 32768))
_READ_CHUNK = 1024 * 1024


@router.get("/models", response_model=ModelsResponse)
async def list_models():
    status = registry.models_status()
    return {
        "default": registry.DEFAULT_MODEL,
        "enabled": sorted(name for name, m in status.items() if m["enabled"]),
        "models": status,
    }


async def _diarize_file(tmp_path, model, hotwords, max_new_tokens):
    try:
        try:
            # Only used to validate duration up-front; the backend itself
            # decodes audio_path directly (VibeVoiceASRProcessor / pyannote's
            # Pipeline both accept a file path and handle their own resample).
            _, duration = await asyncio.to_thread(decode_audio, tmp_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if duration < MIN_AUDIO_SECONDS:
            raise HTTPException(status_code=400, detail=f"Audio is shorter than {MIN_AUDIO_SECONDS}s")
        if duration > MAX_AUDIO_SECONDS:
            raise HTTPException(
                status_code=400,
                detail=f"Audio exceeds the {MAX_AUDIO_SECONDS}s (max 60 min) limit",
            )

        try:
            model_id, result = await registry.run(model, tmp_path, hotwords, max_new_tokens)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    finally:
        os.unlink(tmp_path)

    return model_id, result, duration


def _observe_completion(task):
    if not task.cancelled():
        task.exception()


@router.post("/diarize", response_model=DiarizeResponse)
async def diarize(
    file: UploadFile,
    model: str = Form(registry.DEFAULT_MODEL),
    hotwords: str = Form(""),
    max_new_tokens: int = Form(DEFAULT_MAX_NEW_TOKENS),
):
    if not registry.is_enabled(model):
        raise HTTPException(status_code=400, detail=f"Backend '{model}' is not enabled")
    if not 1 <= max_new_tokens <= 65536:
        raise HTTPException(status_code=400, detail="max_new_tokens must be between 1 and 65536")

    filename = file.filename or "audio"
    suffix = os.path.splitext(filename)[1] or ".wav"

    chunks, total = [], 0
    while True:
        chunk = await file.read(_READ_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File exceeds the {MAX_UPLOAD_MB} MB limit")
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    # Keep the temporary audio available until the worker finishes, even if
    # the client cancels while a model is loading or inference is still running.
    work = asyncio.create_task(_diarize_file(tmp_path, model, hotwords, max_new_tokens))
    work.add_done_callback(_observe_completion)
    model_id, result, duration = await asyncio.shield(work)

    speakers = sorted({s.speaker_id for s in result.segments})
    return {
        "filename": filename,
        "model": model,
        "model_id": model_id,
        "duration_seconds": round(duration, 2),
        "num_speakers": len(speakers),
        "speakers": speakers,
        "segments": [s.__dict__ for s in result.segments],
        "raw_text": result.raw_text,
        "warning": result.warning,
    }
