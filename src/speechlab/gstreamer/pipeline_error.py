from typing import Self
from threading import Lock, Event


class PipelineError(Exception):
    """
    Base exception for the Pipeline system.

    Two ways to create:

        # 1. Simple — just a message
        raise PipelineError("something went wrong")

        # 2. Wrap an existing exception — saves original traceback
        try:
            do_work()
        except Exception as exc:
            results["error"] = PipelineError.wrap(exc, "worker stage failed")

        # Later in main thread — raise and it works natively
        raise results["error"]
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    @classmethod
    def wrap(cls, exc: Exception, message: str) -> Self:
        """
        Wrap an existing exception with additional context message.

        Sets __cause__ so Python automatically shows both tracebacks:
        original exception and the place where wrapped error was raised.

        Args:
            exc:     Original exception to wrap.
            message: Your context — where and why it failed.

        Returns:
            PipelineError instance, ready to save or raise.
        """
        instance = cls(f"{message}: {exc}")
        instance.__cause__ = exc
        return instance

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.message!r})"


class PipelineErrorStorage:
    def __init__(self, stop_event: Event) -> None:
        self._errors: list[PipelineError] = []
        self._lock = Lock()
        self._stop_event = stop_event

    def add(self, error: PipelineError) -> None:
        with self._lock:
            self._errors.append(error)
        self._stop_event.set()

    def get(self) -> list[PipelineError]:
        with self._lock:
            return self._errors
