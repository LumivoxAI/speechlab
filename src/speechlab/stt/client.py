from .protocol import STTRequest, STTResponse
from ..transport.client import BaseZMQAsyncClient


class STTAsyncClient(BaseZMQAsyncClient):
    def __init__(
        self,
        address: str,
        client_id: str = None,
        name: str = "STTClient",
    ) -> None:
        super().__init__(address, client_id=client_id, name=name)

    async def add_audio(self, session_id: int, audio_data: bytes) -> STTResponse:
        return await self._send_request(session_id, False, False, audio_data)

    async def stt(
        self,
        session_id: int,
        audio_data: bytes = None,
        remove_session: bool = True,
    ) -> STTResponse:
        return await self._send_request(session_id, True, remove_session, audio_data)

    async def remove_session(self, session_id: int) -> STTResponse:
        return await self._send_request(session_id, False, True)

    async def _send_request(
        self,
        session_id: int,
        call_stt: bool,
        remove_session: bool,
        audio_data: bytes = None,
    ) -> STTResponse:
        request = STTRequest(
            session_id=session_id,
            call_stt=call_stt,
            remove_session=remove_session,
            data_exist=audio_data is not None,
        )

        ok = await self.send(request, audio_data)
        if not ok:
            return STTResponse(session_id=session_id, error="Client send error")

        parts = await self.recv(STTResponse)
        if parts is None:
            return STTResponse(session_id=session_id, error="Client recv error")

        response = parts[0]
        if response.error:
            self.log.warning(f"Server returned error: {response.error}")
        return response
