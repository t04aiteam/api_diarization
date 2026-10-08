from typing import Dict, List, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    default_model: str
    models: Dict[str, str]


class ModelInfo(BaseModel):
    model_id: str
    enabled: bool
    transcribes: bool


class ModelsResponse(BaseModel):
    default: str
    enabled: List[str]
    models: Dict[str, ModelInfo]


class SegmentOut(BaseModel):
    start_time: float
    end_time: float
    speaker_id: int
    text: Optional[str] = None


class DiarizeResponse(BaseModel):
    filename: str
    model: str
    model_id: str
    duration_seconds: float
    num_speakers: int
    speakers: List[int]
    segments: List[SegmentOut]
    raw_text: Optional[str] = None
    warning: Optional[str] = None
