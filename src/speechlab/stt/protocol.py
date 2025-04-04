from ..transport.message import BaseMessage


class STTRequest(BaseMessage):
    session_id: int
    call_stt: bool
    remove_session: bool
    data_exist: bool


class STTResponse(BaseMessage):
    session_id: int
    error: str | None = None
    text: str | None = None
