# API Phien Am & Phan Tach Nguoi Noi (VibeVoice ASR)

A FastAPI REST service for joint **transcription + speaker diarization**: upload a
meeting/call recording, get back timed, speaker-tagged utterances with transcribed text.

This is a **parallel test environment**: the original deployment assumed the model was
already cached and ran fully offline. Here, `.env` defaults to letting the service reach
the Hub so the checkpoint downloads and caches automatically the first time it starts —
see [First Run / Model Download](#first-run--model-download) below.

> Implemented per `HDSD_API_Phien_Am_Phan_Tach_Nguoi_Noi_VibeVoice.docx`, using the real
> `vibevoice` package API from https://github.com/microsoft/VibeVoice (verified against
> the repo's own `demo/vibevoice_asr_inference_from_file.py`,
> `demo/vibevoice_asr_gradio_demo.py`, and `vibevoice/processor/vibevoice_asr_processor.py`).

## Backends

| Backend     | Model                                | Transcribes | Enabled by default |
|-------------|----------------------------------------|-------------|----------------------|
| `vibevoice` | `microsoft/VibeVoice-ASR` (9B, Qwen2.5-7B base) | Yes | Yes |
| `pyannote`  | `pyannote/speaker-diarization-3.1`     | No (diarization only) | No — gated model, needs internet + `HF_TOKEN` |

`vibevoice` isn't a standard `transformers.pipeline()` model — it's a custom
`VibeVoiceASRForConditionalGeneration` + `VibeVoiceASRProcessor` pair distributed via the
`vibevoice` package (not on PyPI, see [Setup](#setup)). Hotwords are passed through as the
processor's `context_info` string; structured segments come from the processor's own
`post_process_transcription()`, which returns `[]` (never raises) on unparseable output —
that's what triggers this service's `raw_text`/`warning` fallback.

## Prerequisites

- Python 3.10/3.11, CUDA GPU (required for `vibevoice`'s ~18 GB of weights; falls back to
  CPU/float32 automatically but will be very slow).
- Internet access for the **first** run only, to download the checkpoint — see below.

## Setup

```bash
pip install -r requirements.txt   # installs vibevoice from GitHub (git+https), plus deps
cp .env.example .env              # then edit ENABLED_MODELS / limits if needed
```

`vibevoice` isn't on PyPI; `requirements.txt` installs it straight from the source repo
(`vibevoice @ git+https://github.com/microsoft/VibeVoice.git`). If you'd rather work from
a local clone instead:

```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice && pip install -e .
```

## First Run / Model Download

With `LOCAL_FILES_ONLY=0` / `HF_HUB_OFFLINE=0` (the defaults in `.env.example`),
`VibeVoiceASRProcessor.from_pretrained()` / `VibeVoiceASRForConditionalGeneration.from_pretrained()`
download and cache `microsoft/VibeVoice-ASR` (~18 GB, bf16) plus the `Qwen/Qwen2.5-7B`
tokenizer into `~/.cache/huggingface` automatically the first time the service starts —
this needs internet access and enough disk space, and can take several minutes to tens of
minutes depending on bandwidth. Every run after that reuses the local cache and needs no
network access.

Once the cache is warm, you can reproduce the original deployment's fully-offline,
network-locked behavior by uncommenting the "Locked-down / production mode" block in
`.env` (`LOCAL_FILES_ONLY=1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) and restarting.

## Run Locally

```bash
uvicorn main:app --host 0.0.0.0 --port 6421
```

`.env` is loaded automatically at startup — no `--env-file` flag needed.

**Do not pass `--reload`**: every autoreload would reload ~18 GB of weights onto the GPU.

`/health` only responds once the FastAPI `lifespan` startup hook (which preloads the
default backend — including the first-run download, if needed) has finished — wait for it
before calling `/diarize`.

```bash
curl http://localhost:6421/health
# {"status": "ok", "default_model": "vibevoice", "models": {"vibevoice": "microsoft/VibeVoice-ASR"}}
```

Interactive docs (Swagger UI): `http://localhost:6421/docs`

## Endpoints

| Method | Path      | Body                              | Response                                                     |
|--------|-----------|-------------------------------------|-----------------------------------------------------------------|
| GET    | `/health` | —                                    | `{status, default_model, models}`                                |
| GET    | `/models` | —                                    | `{default, enabled, models: {name: {model_id, enabled, transcribes}}}` |
| POST   | `/diarize`| `multipart/form-data`                | JSON: filename, model, duration, speakers, segments, ...          |

No authentication — intended for use inside a trusted internal network only.

### `POST /diarize`

Form fields:

| Key              | Type | Required | Notes                                                              |
|-------------------|------|----------|------------------------------------------------------------------------|
| `file`            | File | Yes      | Recording: wav/flac/mp3/ogg/m4a... max 500 MB, max 60 min               |
| `model`           | Text | No       | Backend to use; default `vibevoice` (`pyannote` disabled by default)    |
| `hotwords`        | Text | No       | Comma-separated proper nouns/terms to improve transcription accuracy    |
| `max_new_tokens`  | Text | No       | Generation token limit, 1–65536; default 32768                         |

The audio processor handles decoding/resampling internally (input can be any common
format/sample rate) — no client-side conversion needed. `hotwords` is passed straight
through as the model's `context_info` field, e.g. proper nouns/terms it should transcribe
accurately.

```bash
curl -X POST http://localhost:6421/diarize \
  -F "file=@cuoc_hop.wav" \
  -F "hotwords=Trí Nam, EBITDA, Nguyễn Văn A"
```

Response:

```json
{
  "filename": "cuoc_hop.wav",
  "model": "vibevoice",
  "model_id": "microsoft/VibeVoice-ASR",
  "duration_seconds": 372.5,
  "num_speakers": 2,
  "speakers": [0, 1],
  "segments": [
    {"start_time": 0.0, "end_time": 4.2, "speaker_id": 0, "text": "Chào các anh chị, ta bắt đầu nhé."},
    {"start_time": 4.5, "end_time": 9.1, "speaker_id": 1, "text": "Vâng, nội dung thứ nhất..."}
  ],
  "raw_text": null,
  "warning": null
}
```

If the model's output can't be parsed into structured segments, `segments` comes back
empty, `warning` is set, and the full raw transcript is in `raw_text` instead.

## Error Codes

| HTTP | Meaning                                                                 | Fix                                          |
|------|----------------------------------------------------------------------------|--------------------------------------------------|
| 400  | Empty file / can't decode / shorter than 0.1s / longer than 60 min / backend not enabled | Check the file and `model` parameter          |
| 413  | File exceeds 500 MB                                                        | Trim the recording                                |
| 500  | Inference error / GPU out of memory                                        | Check server logs; try lowering `max_new_tokens`  |

## Project Layout

```
main.py                # FastAPI app, lifespan preload, /health
routes/diarize.py      # /models, /diarize
models/base.py         # Backend interface + Segment/DiarizeResult
models/registry.py     # enabled/default backend resolution (.env-driven), GPU lock
models/vibevoice.py    # VibeVoice ASR backend (transcribes=True)
models/pyannote.py     # pyannote diarization backend (transcribes=False, disabled by default)
models/audio_utils.py  # lightweight duration-check decode (0.1s / 60min validation)
schemas/response.py
requirements.txt
.env.example
```

## Operational Notes

- **Serialized GPU access**: only one `/diarize` inference runs at a time
  (`models/registry.py`'s `inference_lock`); concurrent requests queue rather than run in
  parallel.
- **Enabling `pyannote`**: set `ENABLED_MODELS=vibevoice,pyannote` and `HF_TOKEN` in
  `.env`, `pip install pyannote.audio`, and ensure the host has internet the first time it
  downloads (gated model, terms must be accepted on huggingface.co).
- **First-run download vs. locked-down mode**: with the default `.env.example` settings
  (`LOCAL_FILES_ONLY=0`, `HF_HUB_OFFLINE=0`, `TRANSFORMERS_OFFLINE=0`), the service can
  always fall back to the Hub — convenient for this test environment, but it means a
  missing/corrupt cache entry triggers a network fetch instead of a clear local error.
  Once `~/.cache/huggingface` is warm, flip those three to `1` to pin the service to the
  cache and reproduce the original deployment's network-locked behavior; after that, if
  the cache is ever deleted the service can't restart (no network fallback left).
- These offline flags are read by `transformers`/`huggingface_hub` at import time, so
  `.env` must load before those imports — see the `load_dotenv()` placement at the top of
  `main.py`.
