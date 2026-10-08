"""
VibeVoice ASR backend: joint transcription + speaker diarization.

microsoft/VibeVoice-ASR is a 9B-parameter audio-language model (Qwen2.5-7B
backbone) distributed as a custom `vibevoice` package, not a plain
`transformers.pipeline()` task. This mirrors the official reference
inference code (microsoft/VibeVoice repo: `demo/vibevoice_asr_inference_from_file.py`,
`demo/vibevoice_asr_gradio_demo.py`, `vibevoice/processor/vibevoice_asr_processor.py`):

- Model class: `VibeVoiceASRForConditionalGeneration`
- Processor class: `VibeVoiceASRProcessor` (also loads the `Qwen/Qwen2.5-7B`
  tokenizer via `language_model_pretrained_name`)
- Hotwords are passed as a free-text `context_info` string, not a list.
- Structured segments come from the processor's own
  `post_process_transcription()`, which returns `[]` (never raises) if the
  model's output can't be parsed as JSON — that's the trigger for this
  service's raw_text/warning fallback.

`from_pretrained(MODEL_ID, ...)` downloads and caches the ~18 GB of weights
(plus the Qwen2.5-7B tokenizer) from the Hub on first use if they aren't
already in `~/.cache/huggingface`; set `LOCAL_FILES_ONLY=1` once that cache is
warm to lock the service into fully-offline mode, mirroring production.

Reference: https://github.com/microsoft/VibeVoice
"""

import os
from typing import Optional

import torch

from models.base import Backend, DiarizeResult, Segment

MODEL_ID = "microsoft/VibeVoice-ASR"
TOKENIZER_ID = "Qwen/Qwen2.5-7B"

LOCAL_FILES_ONLY = os.getenv("LOCAL_FILES_ONLY", "0") == "1"
HF_TOKEN = os.getenv("HF_TOKEN") or None


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(device: str):
    return torch.bfloat16 if device == "cuda" else torch.float32


def _resolve_attn_implementation(device: str) -> str:
    if device == "cuda":
        try:
            import flash_attn  # noqa: F401

            return "flash_attention_2"
        except ImportError:
            pass
    return "sdpa"


def _parse_speaker_id(value) -> int:
    if isinstance(value, int):
        return value
    digits = "".join(filter(str.isdigit, str(value)))
    return int(digits) if digits else 0


def _to_segment(raw: dict) -> Optional[Segment]:
    try:
        return Segment(
            start_time=float(raw.get("start_time", raw.get("start", 0))),
            end_time=float(raw.get("end_time", raw.get("end", 0))),
            speaker_id=_parse_speaker_id(raw.get("speaker_id", raw.get("speaker", 0))),
            text=str(raw.get("text", "")).strip(),
        )
    except (TypeError, ValueError):
        return None


class VibeVoiceBackend(Backend):
    name = "vibevoice"
    model_id = MODEL_ID
    transcribes = True

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = None

    def load(self) -> None:
        from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
        from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

        self._device = _resolve_device()
        dtype = _resolve_dtype(self._device)
        attn_implementation = _resolve_attn_implementation(self._device)

        self._processor = VibeVoiceASRProcessor.from_pretrained(
            self.model_id,
            language_model_pretrained_name=TOKENIZER_ID,
            local_files_only=LOCAL_FILES_ONLY,
            token=HF_TOKEN,
        )

        self._model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=dtype,
            device_map=None,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
            local_files_only=LOCAL_FILES_ONLY,
            token=HF_TOKEN,
        )
        self._model = self._model.to(self._device).eval()

    def run(self, audio_path: str, hotwords: Optional[str], max_new_tokens: int) -> DiarizeResult:
        context_info = hotwords.strip() if hotwords and hotwords.strip() else None

        inputs = self._processor(
            audio=audio_path,
            sampling_rate=None,
            return_tensors="pt",
            add_generation_prompt=True,
            context_info=context_info,
        )
        inputs = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "pad_token_id": self._processor.pad_id,
            "eos_token_id": self._processor.tokenizer.eos_token_id,
        }

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **generation_config)

        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_length:]

        eos_id = self._processor.tokenizer.eos_token_id
        eos_positions = (generated_ids == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            generated_ids = generated_ids[: eos_positions[0] + 1]

        generated_text = self._processor.decode(generated_ids, skip_special_tokens=True).strip()

        try:
            raw_segments = self._processor.post_process_transcription(generated_text)
        except Exception:
            raw_segments = []

        segments = [s for s in (_to_segment(r) for r in raw_segments) if s is not None]

        if not segments:
            return DiarizeResult(
                segments=[],
                raw_text=generated_text,
                warning="Model output could not be parsed into structured segments; see raw_text.",
            )
        return DiarizeResult(segments=segments)
