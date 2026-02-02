# app/schemas.py
from pydantic import BaseModel,Field
from typing import Literal


class VoiceDetectionRequest(BaseModel):
    language: Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    audioFormat: Literal["mp3"]
    audioBase64: str




class VoiceDetectionResponse(BaseModel):
    status: Literal["success"]
    language: Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    classification: Literal["AI_GENERATED", "HUMAN"]
    confidenceScore: float = Field(..., ge=0.0, le=1.0)
    explaination: str


class ErrorResponse(BaseModel):
    status: Literal["error"]
    message: str
