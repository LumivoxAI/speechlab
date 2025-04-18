from pathlib import Path

from .model import RuNormModel
from .config import RuNormConfig
from ...transport.loader import BaseLoader


class RuNormModelLoader(BaseLoader):
    def __init__(self, data_dir: Path | str) -> None:
        super().__init__(data_dir, "runorm")

    def _warm_up_model(self, model: RuNormModel) -> None:
        _ = model.preprocess("Замок на двери замка 5 мая 2024")

    def get_model(self, config: RuNormConfig) -> RuNormModel:
        RuNormConfig.model_validate(config)

        model = RuNormModel()
        model.load(
            model_size=config.model_size.value,
            device=config.device.value,
            workdir=str(self.model_dir),
        )

        self._warm_up_model(model)

        return model
