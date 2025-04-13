import os.path
from typing import Callable
from pathlib import Path

from runorm import RUNorm

from .model import RuNormModel
from .config import RuNormConfig


class RuNormModelLoader:
    def __init__(self, model_dir: Path | str) -> None:
        self._model_dir = Path(model_dir) / "runorm"
        self._model_dir.mkdir(parents=True, exist_ok=True)

    def _warm_up_model(self, model: RuNormModel) -> None:
        _ = model.preprocess("Замок на двери замка 5 мая 2024")

    def get_model(self, config: RuNormConfig) -> RuNormModel:
        RuNormConfig.model_validate(config)

        model = RuNormModel()
        model.load(
            model_size=config.model_size.value,
            device=config.device.value,
            workdir=str(self._model_dir),
        )

        self._warm_up_model(model)

        return model
