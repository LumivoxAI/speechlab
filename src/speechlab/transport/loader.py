from abc import ABC
from pathlib import Path


class BaseLoader(ABC):
    def __init__(self, data_dir: Path | str, name: str) -> None:
        data_dir = Path(data_dir)

        self._model_dir = data_dir / "model" / name
        self._model_dir.mkdir(parents=True, exist_ok=True)

        self._reference_dir = data_dir / "reference"
        self._reference_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    @property
    def reference_dir(self) -> Path:
        return self._reference_dir
