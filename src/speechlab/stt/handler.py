from abc import abstractmethod

from ..transport.handler import BaseHandler


class BaseSTTHandler(BaseHandler):
    @abstractmethod
    def add_audio(self, key: tuple[str, int], data: bytes) -> None: ...

    @abstractmethod
    def stt(self, key: tuple[str, int], remove_session: bool) -> str: ...

    @abstractmethod
    def remove_session(self, key: tuple[str, int]) -> None: ...

    @abstractmethod
    def check_inactive(self) -> None: ...
