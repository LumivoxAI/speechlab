from pydantic import Field
from typing_extensions import Annotated

from ...transport.message import BaseMessage


class RuNormRequest(BaseMessage):
    session_id: int
    text: str


class RuNormResponse(BaseMessage):
    session_id: int
    text: str | None = None
    error: str | None = None
