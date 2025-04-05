from abc import ABC, abstractmethod


class BaseSTTHandler(ABC):
    @abstractmethod
    def add_audio(self, key: tuple[str, int], data: bytes) -> None: ...

    @abstractmethod
    def stt(self, key: tuple[str, int], remove_session: bool) -> str: ...

    @abstractmethod
    def reset_by_client(self, client_id: str) -> None: ...

    @abstractmethod
    def close(self) -> None: ...
