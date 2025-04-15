from .handler import BaseSTTHandler
from .protocol import STTRequest, STTResponse
from ..transport.server import BaseZMQServer


class STTServer(BaseZMQServer[BaseSTTHandler]):
    def __init__(
        self,
        handler: BaseSTTHandler,
        address: str,
        name: str = "STTServer",
    ) -> None:
        super().__init__(handler, address, name)

    def _add_audio(
        self,
        client_id: str,
        session_id: int,
        audio_data: bytes | None,
    ) -> STTResponse | None:
        if audio_data is None:
            return STTResponse(
                session_id=session_id,
                error="Audio data flag set but no audio data provided",
            )

        try:
            self.handler.add_audio((client_id, session_id), audio_data)
            self.log.debug(f"Added {len(audio_data)} bytes of audio for {client_id}:{session_id}")
            return None
        except Exception as e:
            self.log.exception(f"Failed to add audio for {client_id}:{session_id}")
            return STTResponse(session_id=session_id, error=f"Failed to add audio: {str(e)}")

    def _call_stt(
        self,
        client_id: str,
        session_id: int,
        remove_session: bool,
    ) -> STTResponse:
        try:
            text = self.handler.stt((client_id, session_id), remove_session)
            self.log.debug(f"STT processed for {client_id}:{session_id}, result text: {text}")
            return STTResponse(session_id=session_id, text=text)
        except Exception as e:
            self.log.exception(f"Failed to process STT for {client_id}:{session_id}")
            return STTResponse(session_id=session_id, error=f"STT processing error: {str(e)}")

    def _remove_session(
        self,
        client_id: str,
        session_id: int,
    ) -> STTResponse:
        try:
            self.handler.remove_session((client_id, session_id))
            self.log.debug(f"Session {client_id}:{session_id} removed")
            return STTResponse(session_id=session_id)
        except Exception as e:
            self.log.exception(f"Failed to remove session {client_id}:{session_id}")
            return STTResponse(session_id=session_id, error=f"Failed to remove session: {str(e)}")

    def _process_request(
        self,
        client_id: str,
        request: STTRequest,
        audio_data: bytes | None,
    ) -> STTResponse:
        session_id = request.session_id

        if request.data_exist:
            response = self._add_audio(client_id, session_id, audio_data)
            if response is not None:
                return response

        if request.call_stt:
            return self._call_stt(client_id, session_id, request.remove_session)

        if request.remove_session:
            return self._remove_session(client_id, session_id)

        return STTResponse(session_id=session_id)

    def process(self) -> None:
        parts = self.recv3(STTRequest)
        if parts == None:
            return

        client_id_bin, request, audio_data = parts
        client_id = client_id_bin.decode("utf-8", errors="replace")
        response = self._process_request(client_id, request, audio_data)

        _ = self.send(client_id_bin, response)
        self.handler.check_inactive()
