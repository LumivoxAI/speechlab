from __future__ import annotations

import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

from .logger import logger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ms_to_ns(ms: int) -> int:
    return ms * 1_000_000


def _ms_to_us(ms: int) -> int:
    return ms * 1_000


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

    def _make(self) -> BaseElement:
        self._impl = Gst.ElementFactory.make(self._factory, self._name)
        if self._impl is None:
            raise RuntimeError(
                f"Failed to create GStreamer element '{self}'. "
                f"Is the corresponding plugin installed?"
            )
        return self

    def link(self, next_element: BaseElement) -> BaseElement:
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
        format: str,
        samplerate: int,
        channels: int,
        name: str | None = None,
    ) -> None:
        super().__init__("capsfilter", name)

        if format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format '{format}'. Use one of: {SUPPORTED_FORMATS}")
        if samplerate <= 0:
            raise ValueError("samplerate must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")

        self._make()

        # interleaved = [L, R, L, R, ...]
        layout = "interleaved"
        caps = [
            f"audio/x-raw",
            f"format={format}",
            f"rate={samplerate}",
            f"channels={channels}",
            f"layout={layout}",
        ]

        self._impl.set_property("caps", Gst.Caps.from_string(",".join(caps)))
