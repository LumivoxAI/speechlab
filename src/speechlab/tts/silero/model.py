import gc
from typing import Any, Iterator

import torch
from numpy import ndarray

from ...transport.model import BaseModel


class SileroModel(BaseModel):
    def __init__(self, impl: Any) -> None:
        super().__init__()

        self._impl = impl
        self._speakers = impl.speakers

    def tts(
        self,
        text: str,
        speaker_id: str,
        samplerate: int,
    ) -> Iterator[ndarray]:
        """
        Audio is returned in numpy array format.
        The data type is normalized float
        """

        if speaker_id not in self._speakers:
            raise ValueError(f"Wrong speaker_id: '{speaker_id}'")

        if samplerate not in [8000, 24000, 48000]:
            raise ValueError(f"Wrong samplerate: '{samplerate}'")

        data = self._impl.apply_tts(
            text=text,
            speaker=speaker_id,
            sample_rate=samplerate,
        ).numpy()
        data.shape = (-1,)

        yield data

    def close(self) -> None:
        if self._impl is not None:
            del self._impl
            self._impl = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
