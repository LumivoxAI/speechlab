from pydantic import Field
from typing_extensions import Annotated

from ...transport.message import BaseMessage


class FishSpeechRequest(BaseMessage):
    session_id: int
    text: str
    reference_id: str | None = None
    seed: int | None = None
    max_new_tokens: int = 1024
    top_p: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.9
    temperature: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.6
    repetition_penalty: Annotated[float, Field(ge=0.9, le=2.0, strict=True)] = 1.2


class FishSpeechResponse(BaseMessage):
    session_id: int
    samplerate: int
    is_final: bool
    error: str | None = None
