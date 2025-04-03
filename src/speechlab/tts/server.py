from .handler import BaseTTSHandler
from ..transport.server import BaseZMQServer


class TTSServer(BaseZMQServer):
    def __init__(self, handler: BaseTTSHandler, address: str, name: str = "TTSServer") -> None:
        super().__init__(address, name=name)
        self._handler = handler
        self._request_type = handler.request_type()

    def process(self) -> None:
        parts = self.recv2(self._request_type)
        if parts == None:
            return
        client_id, request = parts
        for responce, data in self._handler.tts(request):
            if not self.send(client_id, responce, data):
                break

    def close(self) -> None:
        super().close()
        try:
            if self._handler is not None:
                self._handler.close()
                self._handler = None
        except Exception:
            self.log.exception(f"Error during server shutdown")
