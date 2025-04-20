from typing import Iterator

import numpy as np

from .model import SileroModel
from ..handler import BaseTTSHandler
from .protocol import SileroRequest, SileroResponse

FLOAT32_TO_INT16_SCALE = np.float32(32767.0)


class SileroHandler(BaseTTSHandler):
    def __init__(self, model: SileroModel) -> None:
        self._model = model

    def request_type(self) -> type[SileroRequest]:
        return SileroRequest

    def tts(self, msg: SileroRequest) -> Iterator[tuple[SileroResponse, bytes | None]]:
        meta = SileroResponse(
            session_id=msg.session_id,
            samplerate=self._model.samplerate,
            is_final=False,
            error=None,
        )

        try:
            for data in self._model.tts(
                text=msg.text,
                speaker_id=msg.speaker_id,
                samplerate=msg.samplerate,
            ):
                bin = (data * FLOAT32_TO_INT16_SCALE).astype(np.int16).tobytes()
                yield (meta, bin)
        except GeneratorExit:
            # disable send final message if iterator is closed
            return
        except Exception as e:
            meta.error = str(e)
            meta.is_final = True
            yield (meta, None)
            return

        meta.is_final = True
        yield (meta, None)

    def close(self) -> None:
        self._model.close()
        del self._model
        self._model = None
