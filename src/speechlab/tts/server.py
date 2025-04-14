from .handler import BaseTTSHandler
from ..transport.server import BaseZMQServer


class TTSServer(BaseZMQServer[BaseTTSHandler]):
    def __init__(self, handler: BaseTTSHandler, address: str, name: str = "TTSServer") -> None:
        super().__init__(address, handler, name)
        self._request_type = handler.request_type()

    def process(self) -> None:
        parts = self.recv2(self._request_type)
        if parts == None:
            return
        client_id, request = parts
        for response, data in self.handler.tts(request):
            if not self.send(client_id, response, data):
                break
