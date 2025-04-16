from typing import Iterator

import numpy as np

from .model import XTTSModel
from ..handler import BaseTTSHandler
from .protocol import XTTSRequest, XTTSResponse

FLOAT32_TO_INT16_SCALE = np.float32(32767.0)


class XTTSHandler(BaseTTSHandler):
    def __init__(self, model: XTTSModel) -> None:
        self._model = model

    def request_type(self) -> type[XTTSRequest]:
        return XTTSRequest

    def tts(self, msg: XTTSRequest) -> Iterator[tuple[XTTSResponse, bytes | None]]:
        meta = XTTSResponse(
            session_id=msg.session_id,
            samplerate=self._model.samplerate,
            is_final=False,
            error=None,
        )

        try:
            for data in self._model.tts(
                text=msg.text,
                speaker_id=msg.speaker_id,
                language=msg.language,
                temperature=msg.temperature,
                length_penalty=msg.length_penalty,
                repetition_penalty=msg.repetition_penalty,
                top_k=msg.top_k,
                top_p=msg.top_p,
                speed=msg.speed,
                enable_text_splitting=msg.enable_text_splitting,
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
