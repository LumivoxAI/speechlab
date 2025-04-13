import gc

import torch
from loguru import logger

from .model import RuNormModel
from ..handler import BasePreprocessHandler
from .protocol import RuNormRequest, RuNormResponse


class RuNormHandler(BasePreprocessHandler):
    def __init__(self, model: RuNormModel, name: str = "RuNorm") -> None:
        self._model = model
        self._log = logger.bind(name=name)

    def request_type(self) -> type[RuNormRequest]:
        return RuNormRequest

    def preprocess(self, msg: RuNormRequest) -> RuNormResponse:
        try:
            text = self._model.preprocess(msg.text)
            self._log.debug(f"Processed for {msg.session_id}, result: {text}, input: {msg.text}")
            return RuNormResponse(
                session_id=msg.session_id,
                text=text,
                error=None,
            )
        except Exception as e:
            self._log.exception(f"Failed to process text for {msg.session_id}")
            return RuNormResponse(
                session_id=msg.session_id,
                text=None,
                error=str(e),
            )

    def close(self) -> None:
        del self._model
        self._model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
