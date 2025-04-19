from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from pathlib import Path

from .model import BaseModel
from .config import BaseConfig

TConfig = TypeVar("TConfig", bound=BaseConfig)


class BaseLoader(ABC, Generic[TConfig]):
    def __init__(self, data_dir: Path | str, name: str) -> None:
        data_dir = Path(data_dir)

        self._model_dir = data_dir / "model" / name
        self._model_dir.mkdir(parents=True, exist_ok=True)

        self._config_dir = data_dir / "config"
        self._config_dir.mkdir(parents=True, exist_ok=True)

        self._reference_dir = data_dir / "reference"
        self._reference_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    @property
    def reference_dir(self) -> Path:
        return self._reference_dir

    def from_default(self) -> BaseModel:
        cfg = TConfig.create_default()
        return self.from_config(cfg)

    def from_file(self, cfg_file_name: str) -> BaseModel:
        cfg = TConfig.load_from_file(self._config_dir / cfg_file_name)
        return self.from_config(cfg)

    @abstractmethod
    def from_config(self, config: TConfig) -> BaseModel: ...
