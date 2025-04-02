from typing import Iterator

import numpy as np

from .model import FishSpeechModel
from ..handler import BaseTTSHandler
from .protocol import FishSpeechRequest, FishSpeechResponse

FLOAT32_TO_INT16_SCALE = np.float32(32767.0)


class FishSpeechHandler(BaseTTSHandler):
    def __init__(
        self,
        model: FishSpeechModel,
    ) -> None:
        self._model = model

    def request_type(self) -> type[FishSpeechRequest]:
        return FishSpeechRequest

    def tts(self, msg: FishSpeechRequest) -> Iterator[tuple[FishSpeechResponse, bytes | None]]:
        session_id = msg.session_id
        samplerate = 44100
        meta = FishSpeechResponse(
            session_id=session_id,
            samplerate=samplerate,
            is_final=False,
            error=None,
        )

        try:
            for data in self._model.tts(
                text=msg.text,
                reference_id=msg.reference_id,
                seed=msg.seed,
                max_new_tokens=msg.max_new_tokens,
                top_p=msg.top_p,
                temperature=msg.temperature,
                repetition_penalty=msg.repetition_penalty,
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
