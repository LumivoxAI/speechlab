from .protocol import RuNormRequest, RuNormResponse
from ...transport.client import BaseZMQAsyncClient


class RuNormAsyncClient(BaseZMQAsyncClient):
    def __init__(
        self,
        address: str,
        client_id: str = None,
        name: str = "RuNormClient",
    ) -> None:
        super().__init__(address, client_id=client_id, name=name)

    async def preprocess(self, request: RuNormRequest) -> str:
        ok = await self.send(request)
        if not ok:
            raise RuntimeError("Failed to send request to server")

        session_id = request.session_id
        parts = await self.recv(RuNormResponse)
        if parts is None:
            raise RuntimeError("Failed to receive response from server")
        meta: RuNormResponse = parts[0]
        if meta.session_id != session_id:
            raise RuntimeError(
                f"Received response for unknown session {meta.session_id}, expected {session_id}"
            )
        if meta.error:
            raise RuntimeError(f"Server returned error: {meta.error}")

        return meta.text
