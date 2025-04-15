from abc import ABC, abstractmethod
from typing import Iterator

from ..transport.message import BaseMessage


class BasePreprocessHandler(ABC):
    @abstractmethod
    def request_type(self) -> type[BaseMessage]: ...

    @abstractmethod
    def preprocess(self, msg: BaseMessage) -> BaseMessage: ...
