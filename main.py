"""
FastAPI server wrapping VibeVoice ASR for joint transcription + speaker
diarization (API Phien Am & Phan Tach Nguoi Noi).

Edge-inference service for the internal network: takes a meeting/call
recording (up to 60 min / 500 MB) and returns timed, speaker-tagged
utterances with transcribed text.

This instance is a parallel test environment: unlike the original,
network-locked deployment (which assumed the checkpoint was already cached),
`.env` here defaults to letting `from_pretrained()` reach the Hub so the
~18 GB checkpoint downloads and caches automatically on first startup. Flip
LOCAL_FILES_ONLY/HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE to 1 in `.env` once that
cache is warm to reproduce the original's fully-offline behavior.

.env IS auto-loaded here (no --env-file flag needed) — but load_dotenv()
must run before any transformers/huggingface_hub import, since those
libraries read HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE at import time. Hence the
load_dotenv() call sits above every other import.

Reference model: https://huggingface.co/microsoft/VibeVoice-ASR
"""

from dotenv import load_dotenv

load_dotenv()

from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from models import registry  # noqa: E402
from routes.diarize import router as diarize_router  # noqa: E402
from schemas.response import HealthResponse  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the default backend once at startup (can take minutes for ~18 GB
    # of weights) so /health only answers once the service is actually ready.
    registry.preload_default()
    yield


app = FastAPI(
    title="API Phien Am & Phan Tach Nguoi Noi (VibeVoice ASR)",
    description="Transcription + speaker-diarization REST API. POST /diarize "
    "returns timed, speaker-tagged segments with transcribed text.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(diarize_router)


@app.get("/health", response_model=HealthResponse)
async def health():
    return {
        "status": "ok",
        "default_model": registry.DEFAULT_MODEL,
        "models": registry.enabled_models_simple(),
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})
