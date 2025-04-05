import gc
from time import perf_counter_ns

import numpy as np
import torch

from .model import GigaAMModel
from ..handler import BaseSTTHandler

INT16_TO_FLOAT32_SCALE = np.float32(1.0 / 32768.0)


class STTGigaAMHandler(BaseSTTHandler):
    def __init__(
        self,
        model: GigaAMModel,
        inactivity_timeout_ms: int = 10 * 1000,
    ) -> None:
        self._model = model
        self._dtype = model._dtype
        self._inactivity_timeout_ns = inactivity_timeout_ms * 1_000_000
        # (client_id, session_id) -> audio data
        self._data_by_key: dict[tuple[str, int], torch.Tensor] = {}
        self._deadlines_by_key: dict[tuple[str, int], int] = {}

    def add_audio(self, key: tuple[str, int], data: bytes) -> None:
        data = np.frombuffer(data, dtype=np.int16).astype(np.float32) * INT16_TO_FLOAT32_SCALE
        data = torch.tensor(data, dtype=self._dtype)
        self._deadlines_by_key[key] = perf_counter_ns() + self._inactivity_timeout_ns

        try:
            self._data_by_key[key] = torch.cat((self._data_by_key[key], data))
        except KeyError:
            self._data_by_key[key] = data

    def stt(self, key: tuple[str, int], remove_session: bool) -> str:
        if key not in self._data_by_key:
            raise ValueError(f"Key {key} not found in data")

        text = self._model.stt(self._data_by_key[key])
        if remove_session:
            self._data_by_key.pop(key, None)
            self._deadlines_by_key.pop(key, None)
        else:
            self._deadlines_by_key[key] = perf_counter_ns() + self._inactivity_timeout_ns

        return text

    def remove_session(self, key: tuple[str, int]) -> None:
        if key in self._data_by_key:
            self._data_by_key.pop(key, None)
            self._deadlines_by_key.pop(key, None)

    def check_inactive(self) -> None:
        now = perf_counter_ns()
        inactive_keys = [key for key, deadline in self._deadlines_by_key.items() if deadline < now]

        for key in inactive_keys:
            self._data_by_key.pop(key, None)
            self._deadlines_by_key.pop(key, None)

    def close(self) -> None:
        if self._data_by_key is not None:
            self._data_by_key.clear()
            self._data_by_key = None

        if self._deadlines_by_key is not None:
            self._deadlines_by_key.clear()
            self._deadlines_by_key = None

        if self._dtype is not None:
            del self._model
            self._model = None
            self._dtype = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
