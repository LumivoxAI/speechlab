from typing import AsyncIterator

from .protocol import XTTSRequest, XTTSResponse
from ...transport.client import BaseZMQAsyncClient


class XTTSAsyncClient(BaseZMQAsyncClient):
    def __init__(
        self,
        address: str,
        client_id: str = None,
        name: str = "XTTSClient",
    ) -> None:
        super().__init__(address, client_id=client_id, name=name)
        self._samplerate = 0

    @property
    def samplerate(self) -> int:
        return self._samplerate

    async def tts(self, request: XTTSRequest) -> AsyncIterator[bytes]:
        ok = await self.send(request)
        if not ok:
            raise RuntimeError("Failed to send request to server")

        session_id = request.session_id
        try:
            while True:
                parts = await self.recv(XTTSResponse)
                if parts is None:
                    raise RuntimeError("Failed to receive response from server")
                meta: XTTSResponse = parts[0]
                if meta.session_id != session_id:
                    raise RuntimeError(
                        f"Received response for unknown session {meta.session_id}, expected {session_id}"
                    )
                if meta.error:
                    raise RuntimeError(f"Server returned error: {meta.error}")

                self._samplerate = meta.samplerate
                if meta.is_final:
                    break

                data: bytes = parts[1]
                if data is None:
                    raise RuntimeError("Received empty data from server, expected audio data")

                yield data
        except GeneratorExit:
            pass
