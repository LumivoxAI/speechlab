from abc import ABC, abstractmethod


class BaseHandler(ABC):
    @abstractmethod
    def close(self) -> None: ...
