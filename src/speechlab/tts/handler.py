from abc import abstractmethod
from typing import Iterator

from ..transport.handler import BaseHandler
from ..transport.message import BaseMessage


class BaseTTSHandler(BaseHandler):
    @abstractmethod
    def request_type(self) -> type[BaseMessage]: ...

    @abstractmethod
    def tts(self, msg: BaseMessage) -> Iterator[tuple[BaseMessage, bytes | None]]: ...
