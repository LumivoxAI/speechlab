from typing import Self
from pathlib import Path

from tomlkit import dumps, loads
from pydantic import BaseModel


class BaseConfig(BaseModel):
    @classmethod
    def create_default(cls) -> Self:
        return cls()

    def save_to_file(self, filepath: str | Path) -> None:
        data = self.model_dump()
        toml_str = dumps(data)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(toml_str)

    @classmethod
    def load_from_file(cls, filepath: str | Path) -> Self:
        with open(filepath, "r", encoding="utf-8") as f:
            data = loads(f.read())
        return cls.model_validate(data)
