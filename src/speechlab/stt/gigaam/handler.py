import numpy as np
import torch

from .model import GigaAMModel
from ..base_stt import STTBaseHandler

INT16_TO_FLOAT32_SCALE = np.float32(1.0 / 32768.0)


class STTGigaAMHandler(STTBaseHandler):
    def __init__(
        self,
        model: GigaAMModel,
    ) -> None:
        self._model = model
        self._dtype = model._dtype
        self._data_by_key: dict[tuple[str, int], torch.Tensor] = {}

    def stt(self, key: tuple[str, int], remove_session: bool) -> str:
        if key not in self._data_by_key:
            raise ValueError(f"Key {key} not found in data")

        text = self._model.stt(self._data_by_key[key])
        if remove_session:
            self._data_by_key.pop(key, None)
        return text

    def add_audio(self, key: tuple[str, int], data: bytes) -> None:
        data = np.frombuffer(data, dtype=np.int16).astype(np.float32) * INT16_TO_FLOAT32_SCALE
        data = torch.tensor(data, dtype=self._dtype)

        try:
            self._data_by_key[key] = torch.cat((self._data_by_key[key], data))
        except KeyError:
            self._data_by_key[key] = data

    def reset_by_client(self, client_id: str) -> None:
        for key in [key for key in self._data_by_key if key[0] == client_id]:
            self._data_by_key.pop(key, None)

    def close(self) -> None:
        self._data_by_key.clear()
        self._model.to("cpu")
        del self._model
        self._model = None

        from gc import collect

        from torch.cuda import empty_cache

        collect()
        empty_cache()
