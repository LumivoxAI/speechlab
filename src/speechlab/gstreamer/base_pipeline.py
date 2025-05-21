import threading
from abc import ABC, abstractmethod

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from .logger import logger
from .elements import Tee, AppSrc, AppSink, BaseElement

Gst.init(None)


class PipelineError(Exception):
    pass


class BasePipeline(ABC):
    def __init__(self) -> None:
        self._pipeline: Gst.Pipeline | None = None
        self._bus: Gst.Bus | None = None
        self._bus_thread: threading.Thread | None = None

        self._stop_event = threading.Event()
        self._error: PipelineError | None = None
        self._error_lock = threading.Lock()
        self._running = False

    # --- Public API --------------------------------------------------------

    def start(self) -> None:
        if self._running:
            raise RuntimeError("Pipeline is already running")

        self._stop_event.clear()
        self._error = None

        try:
            self._pipeline = Gst.Pipeline.new(None)
            if self._pipeline is None:
                raise PipelineError("Failed to create GStreamer pipeline")

            self._bus = self._pipeline.get_bus()
            self._build()

            ret = self._pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise PipelineError("Failed to set pipeline to PLAYING state")

            self._bus_thread = threading.Thread(
                target=self._bus_loop,
                name=f"{self.__class__.__name__}-bus",
                daemon=True,
            )
            self._bus_thread.start()

            self._on_started()

        except Exception:
            self._shutdown()
            raise

        self._running = True
        logger.info(f"{self.__class__.__name__} started")

    @property
    def running(self) -> bool:
        return self._running

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

    def _stop(self) -> None:
        if not self._running:
            return
        self._shutdown()
        self._check_error()

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
                    f"GStreamer error from '{msg.src.get_name()}': {err.message}. Debug: {debug}"
                )
                logger.error(str(error))
                with self._error_lock:
                    self._error = error
                self._stop_event.set()
                self._on_eos()  # unblock anything waiting on EOS
                break

            elif msg.type == Gst.MessageType.WARNING:
                warn, debug = msg.parse_warning()
                logger.warning(
                    f"GStreamer warning from '{msg.src.get_name()}': {warn.message}. Debug: {debug}"
                )

            elif msg.type == Gst.MessageType.EOS:
                logger.info(f"{self.__class__.__name__} received EOS")
                self._stop_event.set()
                self._on_eos()
                break

        logger.debug(f"{self.__class__.__name__} bus thread exiting")

    def _shutdown(self) -> None:
        self._stop_event.set()

        self._before_shutdown()

        if self._bus_thread is not None:
            self._bus_thread.join(timeout=2.0)
            if self._bus_thread.is_alive():
                logger.warning(f"{self.__class__.__name__} bus thread did not exit in time")
            self._bus_thread = None

        if self._pipeline is not None:
            ret = self._pipeline.set_state(Gst.State.NULL)
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.warning(f"{self.__class__.__name__} failed to set pipeline to NULL state")
            self._pipeline = None

        self._bus = None
        self._running = False
        logger.info(f"{self.__class__.__name__} stopped")

    def _check_error(self) -> None:
        with self._error_lock:
            if self._error is not None:
                raise self._error
