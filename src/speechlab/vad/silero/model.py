from abc import abstractmethod

import numpy as np

from ...transport.model import BaseModel

SAMPLE_RATE = 16000
WINDOW_SIZE = 512


def validate_chunked_pcm16(data: np.ndarray) -> np.ndarray:
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if data.dtype != np.int16:
        raise TypeError("Input data must use np.int16 dtype")
    if data.ndim != 1:
        raise ValueError("Input data must be a one-dimensional PCM16 array")
    if data.shape[0] % WINDOW_SIZE != 0:
        raise ValueError(f"Input data length must be a multiple of {WINDOW_SIZE}")

    return data


class SileroModel(BaseModel):
    """
    One-dimensional PCM16 mono audio at 16 kHz (numpy array of dtype np.int16).
    """

    @abstractmethod
    def __call__(self, data: np.ndarray) -> list[float]:
        """
        Returns speech probabilities for each 512-sample chunk.
        """
        ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...
