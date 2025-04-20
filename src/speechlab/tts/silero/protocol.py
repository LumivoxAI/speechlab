from pydantic import Field
from typing_extensions import Literal, Annotated

from ...transport.message import BaseMessage


class SileroRequest(BaseMessage):
    session_id: Annotated[
        int,
        Field(description="Служебное поле для идентификации и различения разных сессий/запросов."),
    ]

    text: Annotated[str, Field(description="Текст, который требуется озвучить.")]

    speaker_id: Annotated[
        str,
        Field(description="Идентификатор голоса, которым будет озвучен текст."),
    ] = "xenia"

    samplerate: Annotated[
        Literal[8000, 24000, 48000],
        Field(
            description="Частота дискретизации аудио в Гц",
        ),
    ] = 24000


class SileroResponse(BaseMessage):
    session_id: int
    samplerate: int
    is_final: bool
    error: str | None = None
