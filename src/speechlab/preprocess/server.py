from .handler import BasePreprocessHandler
from ..transport.server import BaseZMQServer


class PreprocessServer(BaseZMQServer):
    def __init__(
        self,
        handler: BasePreprocessHandler,
        address: str,
        name: str = "PreprocessServer",
    ) -> None:
        super().__init__(address, name=name)
        self._handler = handler
        self._request_type = handler.request_type()

    def process(self) -> None:
        parts = self.recv2(self._request_type)
        if parts == None:
            return
        client_id, request = parts
        response = self._handler.preprocess(request)
        self.send(client_id, response)

    def close(self) -> None:
        super().close()
        try:
            if self._handler is not None:
                self._handler.close()
                self._handler = None
        except Exception:
            self.log.exception(f"Error during server shutdown")
