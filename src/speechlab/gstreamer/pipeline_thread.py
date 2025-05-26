from typing import Callable
from threading import Event, Thread

from .logger import GstLogger
from .pipeline_error import PipelineError, PipelineErrorStorage


class PipelineThread:
    def __init__(
        self,
        name: str,
        target: Callable[[Event], None],
        stop_event: Event,
        errors: PipelineErrorStorage,
        logger: GstLogger,
    ) -> None:
        self._logger = logger
        self._target = target
        self._stop_event = stop_event
        self._errors = errors
        self._thread = Thread(
            target=self._loop,
            name=name,
            daemon=True,
        )
        self._thread.start()

    def _loop(self) -> None:
        try:
            self._target(self._stop_event)
        except Exception as ext:
            msg = f"unhandled exception in thread {self._thread.name}"
            self.errors.add(PipelineError.wrap(ext, msg))

    def stop(self, timeout: float = 2.0) -> None:
        if self._thread is None:
            return

        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            self._logger.warning(f"{self._thread.name} thread did not exit in time")
        self._thread = None


class PipelineThreadStorage:
    def __init__(
        self,
        name: str,
        stop_event: Event,
        errors: PipelineErrorStorage,
        logger: GstLogger,
    ) -> None:
        self._name = name
        self._stop_event = stop_event
        self._errors = errors
        self._logger = logger
        self._threads: list[PipelineThread] = []

    def add(
        self,
        name: str,
        target: Callable[[Event], None],
    ) -> None:
        thread = PipelineThread(
            name,
            target,
            self._stop_event,
            self._errors,
            self._logger,
        )
        self._threads.append(thread)

    def stop(self, timeout: float = 2.0) -> None:
        for thread in self._threads:
            thread.stop(timeout=timeout)
        self._threads = []
