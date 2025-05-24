from enum import IntEnum
from typing import Self

import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from .logger import logger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ms_to_ns(ms: int) -> int:
    return ms * 1_000_000


def _mcs_to_ns(mcs: int) -> int:
    return mcs * 1_000


def _ns_to_ms(ns: int) -> int:
    return ns // 1_000_000


def _ns_to_mcs(ns: int) -> int:
    return ns // 1_000


S16LE_DTYPE = np.dtype(np.int16)
S16LE_BYTES_PER_SAMPLE = 2


GST_FORMAT_MAP: dict[str, np.dtype] = {
    "S16LE": np.dtype(np.int16),
    "F32LE": np.dtype(np.float32),
}

SUPPORTED_FORMATS = tuple(GST_FORMAT_MAP.keys())


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class BaseElement:
    def __init__(self, factory: str, name: str | None = None) -> None:
        self._factory = factory
        self._name = name
        self._impl: Gst.Element | None = None

    def _make(self) -> Self:
        self._impl = Gst.ElementFactory.make(self._factory, self._name)
        if self._impl is None:
            raise RuntimeError(
                f"Failed to create GStreamer element '{self}'. "
                f"Is the corresponding plugin installed?"
            )
        return self

    def link(self, next_element: Self) -> Self:
        if not self._impl.link(next_element._impl):
            raise RuntimeError(f"Failed to link {self} -> {next_element}")
        return next_element

    @property
    def impl(self) -> Gst.Element:
        if self._impl is None:
            raise RuntimeError(f"Element '{self}' has not been created yet")
        return self._impl

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def factory(self) -> str:
        return self._factory

    def __str__(self) -> str:
        if self._name is not None:
            return f"{self._factory}:{self._name}"
        return self._factory

    def __repr__(self) -> str:
        return str(self)


# ---------------------------------------------------------------------------
# PipeWire
# ---------------------------------------------------------------------------


class PipeWireSrc(BaseElement):
    def __init__(
        self,
        target_object: str | None = None,
        client_name: str | None = None,
        min_buffers: int = 2,
        max_buffers: int = 4,
        name: str | None = None,
    ) -> None:
        super().__init__("pipewiresrc", name)

        if min_buffers <= 0:
            raise ValueError("min_buffers must be positive")
        if max_buffers <= 0:
            raise ValueError("max_buffers must be positive")
        if min_buffers > max_buffers:
            raise ValueError("min_buffers must be <= max_buffers")

        self._make()

        # pw-dump | jq '.[] | select(.type == "PipeWire:Interface:Node") | {id, serial: .info.props["object.serial"], name: .info.props["node.name"]}'
        # node.name - optimal and stable
        # object.serial - stable
        # node.id - unstable
        # None - default
        if target_object is not None:
            self._impl.set_property("target-object", str(target_object))
        if client_name is not None:
            self._impl.set_property("client-name", client_name)
        # Set the timestamp when the buffer exits pipewiresrc
        self._impl.set_property("do-timestamp", True)
        # Instead of PipeWire, GStreamer handles the buffering (which is optimal for audio)
        self._impl.set_property("use-bufferpool", False)

        # Number of buffers “in flight” between PipeWire and GStreamer.
        # Less means lower latency; more means better load tolerance.
        self._impl.set_property("min-buffers", min_buffers)
        self._impl.set_property("max-buffers", max_buffers)

        # Metadata for WirePlumber — Proper routing and priority policies
        props = [
            "props",
            "media.type=Audio",
            "media.category=Capture",
            "media.role=Communication",
        ]

        self._impl.set_property("stream-properties", Gst.Structure.new_from_string(",".join(props)))


# ---------------------------------------------------------------------------
# Audio processing
# ---------------------------------------------------------------------------


class AudioConvert(BaseElement):
    def __init__(self, name: str | None = None) -> None:
        super().__init__("audioconvert", name)
        self._make()
        self._impl.set_property("dithering", 0)
        self._impl.set_property("noise-shaping", 0)


class AudioResample(BaseElement):
    def __init__(self, quality: int = 4, name: str | None = None) -> None:
        super().__init__("audioresample", name)

        if not 0 <= quality <= 10:
            raise ValueError("quality must be between 0 and 10")

        self._make()
        self._impl.set_property("quality", quality)


class CapsFilter(BaseElement):
    def __init__(
        self,
        samplerate: int,
        channels: int,
        name: str | None = None,
    ) -> None:
        super().__init__("capsfilter", name)

        if samplerate <= 0:
            raise ValueError("samplerate must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")

        self._make()

        # interleaved = [L, R, L, R, ...]
        layout = "interleaved"
        caps = [
            "audio/x-raw",
            "format=S16LE",
            f"rate={samplerate}",
            f"channels={channels}",
            f"layout={layout}",
        ]

        self._impl.set_property("caps", Gst.Caps.from_string(",".join(caps)))


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------


class AudioQueueDropPolicy(IntEnum):
    DISABLED = 0  # block upstream when full, no drops
    DROP_NEW = 1  # drop new incoming buffers when full
    DROP_OLD = 2  # drop old buffered data when full (best for real-time audio)


class AudioQueue(BaseElement):
    def __init__(
        self,
        max_size_time_ms: int = 200,
        drop_policy: AudioQueueDropPolicy = AudioQueueDropPolicy.DROP_OLD,
        name: str | None = None,
    ) -> None:
        super().__init__("queue", name)
        if max_size_time_ms <= 0:
            raise ValueError("max_size_time_ms must be positive")
        self._make()

        self._impl.set_property("max-size-time", _ms_to_ns(max_size_time_ms))
        self._impl.set_property("max-size-buffers", 0)  # without buffer restrictions
        self._impl.set_property("max-size-bytes", 0)  # without bytes restrictions
        self._impl.set_property("leaky", int(drop_policy))


# ---------------------------------------------------------------------------
# AppSink
# ---------------------------------------------------------------------------


class AppSink(BaseElement):
    _MAX_MAP_FAILURES = 3
    _MAX_INVALID_BUFFERS = 10

    def __init__(self, name: str | None = None) -> None:
        super().__init__("appsink", name)

        self._map_fail_count = 0
        self._invalid_buffer_count = 0

        self._channels: int | None = None
        self._frame_size: int | None = None

        self._make()

        # Pull mode (try-pull-sample)
        self._impl.set_property("emit-signals", False)
        # Buffer should be passed to Python as soon as it arrives.
        self._impl.set_property("sync", False)
        self._impl.set_property("max-buffers", 1)
        self._impl.set_property("max-bytes", 0)
        self._impl.set_property("max-time", 0)
        self._impl.set_property("drop", True)
        # Do not keep a copy of the last debug buffer
        self._impl.set_property("enable-last-sample", False)
        # Stop immediately
        self._impl.set_property("wait-on-eos", False)

    def try_pull_raw(self, timeout_ms: int = 100) -> Gst.Sample | None:
        return self._impl.emit("try-pull-sample", _ms_to_ns(timeout_ms))

    def try_pull(self, timeout_ms: int = 100) -> tuple[np.ndarray | None, int | None]:
        sample = self._impl.emit("try-pull-sample", _ms_to_ns(timeout_ms))
        if sample is None:
            return None, None

        buf = sample.get_buffer()
        if buf is None:
            return None, None

        if self._channels is None:
            self._channels = self._get_channels_from_sample(sample)
            self._frame_size = self._channels * S16LE_BYTES_PER_SAMPLE

        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok or mapinfo is None:
            self._map_fail_count += 1
            if self._map_fail_count >= self._MAX_MAP_FAILURES:
                raise RuntimeError("repeated failed to map GStreamer buffer")
            return None, None
        self._map_fail_count = 0

        try:
            size = mapinfo.size
            if size == 0:
                return None, None

            if size % self._frame_size != 0:
                self._invalid_buffer_count += 1
                if self._invalid_buffer_count >= self._MAX_INVALID_BUFFERS:
                    raise RuntimeError(f"repeated invalid buffer size {size}")
                return None, None
            self._invalid_buffer_count = 0

            arr = np.frombuffer(memoryview(mapinfo.data), dtype=S16LE_DTYPE).copy()

            if self._channels != 1:
                arr = arr.reshape(-1, self._channels)

            pts = buf.pts
            if pts is not None and pts != Gst.CLOCK_TIME_NONE:
                pts = sample.get_segment().to_running_time(Gst.Format.TIME, pts)
            else:
                pts = None

            return arr, pts
        finally:
            buf.unmap(mapinfo)

    @staticmethod
    def _get_channels_from_sample(sample: Gst.Sample) -> int:
        caps = sample.get_caps()
        if caps is None or caps.is_empty() or caps.is_any():
            raise RuntimeError("sample has no valid caps")
        s = caps.get_structure(0)
        if s is None:
            raise RuntimeError("caps has no structure")

        success, value = s.get_int("channels")
        if not success or value <= 0:
            raise RuntimeError(f"invalid channels value in caps")

        return value


# ---------------------------------------------------------------------------
# Tee
# ---------------------------------------------------------------------------


class Tee(BaseElement):
    def __init__(self, name: str | None = None) -> None:
        super().__init__("tee", name)
        self._make()

    def link(self, next_element: BaseElement) -> BaseElement:
        src_pad_template = self._impl.get_pad_template("src_%u")
        src_pad = self._impl.request_pad(src_pad_template, None, None)
        if src_pad is None:
            raise RuntimeError(f"Failed to request src pad from {self}")

        sink_pad = next_element.impl.get_static_pad("sink")
        if sink_pad is None:
            raise RuntimeError(f"Failed to get sink pad from {next_element}")

        ret = src_pad.link(sink_pad)
        if ret != Gst.PadLinkReturn.OK:
            raise RuntimeError(f"Failed to link {self} -> {next_element}: {ret}")

        return next_element


