from abc import ABC, abstractmethod
from typing import Iterator

from ..transport.message import BaseMessage


class BaseTTSHandler(ABC):
    @abstractmethod
    def request_type(self) -> type[BaseMessage]: ...

    @abstractmethod
    def tts(self, msg: BaseMessage) -> Iterator[tuple[BaseMessage, bytes | None]]: ...

    @abstractmethod
    def close(self) -> None: ...
