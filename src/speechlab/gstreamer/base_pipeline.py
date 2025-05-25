import threading
from abc import ABC, abstractmethod
from time import time_ns

import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from .logger import logger
from .elements import Tee, AppSrc, AppSink, BaseElement, _ms_to_ns
from .pipeline_error import PipelineError, PipelineErrorStorage

Gst.init(None)


class BasePipeline(ABC):
    def __init__(self, name: str) -> None:
        self._name = name
        self._pipeline: Gst.Pipeline | None = None
        self._bus: Gst.Bus | None = None
        self._bus_thread: threading.Thread | None = None

        self._stop_event = threading.Event()
        self._errors = PipelineErrorStorage(self._stop_event)
        self._started = False
        self._stopped = False

    # --- Public API --------------------------------------------------------

    def start(self) -> None:
        if self._started:
            raise PipelineError("pipeline is already started")
        if self._stopped:
            raise PipelineError("pipeline has been stopped")

        try:
            self._pipeline = Gst.Pipeline.new(None)
            if self._pipeline is None:
                raise PipelineError("failed to create GStreamer pipeline")

            self._bus = self._pipeline.get_bus()
            self._build()

            ret = self._pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise PipelineError("failed to set pipeline to PLAYING state")

            self._bus_thread = threading.Thread(
                target=self._bus_loop,
                name=f"{self.name}-bus",
                daemon=True,
            )
            self._bus_thread.start()

            self._on_started()
            self._started = True

        except Exception:
            self._stopped = True
            self._shutdown()
            raise

    @property
    def name(self) -> str:
        return self._name

    @property
    def errors(self) -> PipelineErrorStorage:
        return self._errors

    # --- Hooks -------------------------------------------------------------

    @abstractmethod
    def _build(self) -> None:
        """Assemble and link all elements, add them to self._pipeline."""

    @abstractmethod
    def _on_started(self) -> None:
        """Called at the end of start(), before _running = True."""

    def _on_eos(self) -> None:
        """Called when EOS is received on the bus."""

    def _before_shutdown(self) -> None:
        """Called at the start of _shutdown(), before threads are joined."""

    # --- Protected API -----------------------------------------------------

    def _add_and_link(self, elements: list[BaseElement]) -> None:
        for element in elements:
            self._pipeline.add(element.impl)

        for i in range(len(elements) - 1):
            elements[i].link(elements[i + 1])

    def _link_tee(self, tee: Tee, branches: list[list[BaseElement]]) -> None:
        """
        Add all branch elements to pipeline and link tee -> each branch.
        The tee itself must already be added to the pipeline via _add_and_link.
        Each branch is a list of elements starting right after the tee (typically a Queue).
        """
        for branch in branches:
            for element in branch:
                self._pipeline.add(element.impl)

            tee.link(branch[0])

            for i in range(len(branch) - 1):
                branch[i].link(branch[i + 1])

    def _stop(self) -> list[PipelineError]:
        if not self._started or self._stopped:
            return []
        self._shutdown()
        return self.errors.get()

    # --- Private ----------------------------------------------------------

    def _bus_loop(self) -> None:
        while not self._stop_event.is_set():
            msg = self._bus.timed_pop_filtered(
                100 * Gst.MSECOND,
                Gst.MessageType.ERROR | Gst.MessageType.WARNING | Gst.MessageType.EOS,
            )
            if msg is None:
                continue

            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                error = PipelineError(
                    f"gStreamer error from '{msg.src.get_name()}': {err.message}. Debug: {debug}"
                )
                self.errors.add(error)
                self._on_eos()  # unblock anything waiting on EOS
                break

            elif msg.type == Gst.MessageType.WARNING:
                warn, debug = msg.parse_warning()
                logger.warning(
                    f"GStreamer warning from '{msg.src.get_name()}': {warn.message}. Debug: {debug}"
                )

            elif msg.type == Gst.MessageType.EOS:
                logger.info(f"{self.name} received EOS")
                self._stop_event.set()
                self._on_eos()
                break

        logger.debug(f"{self.name} bus thread exiting")

    def _shutdown(self) -> None:
        self._stop_event.set()

        self._before_shutdown()

        if self._bus_thread is not None:
            self._bus_thread.join(timeout=2.0)
            if self._bus_thread.is_alive():
                logger.warning(f"{self.name} bus thread did not exit in time")
            self._bus_thread = None

        if self._pipeline is not None:
            ret = self._pipeline.set_state(Gst.State.NULL)
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.warning(f"{self.name} failed to set pipeline to NULL state")
            self._pipeline = None

        self._bus = None
        self._started = False
        self._stopped = True


