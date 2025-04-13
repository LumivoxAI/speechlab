from enum import StrEnum
from typing import Self

from pydantic import BaseModel, model_validator


class DeviceOption(StrEnum):
    CUDA = "cuda"
    CPU = "cpu"


class ModelSize(StrEnum):
    SMALL = "small"
    MEDIUM = "medium"
    BIG = "big"


class RuNormConfig(BaseModel):
    device: DeviceOption = DeviceOption.CUDA
    model_size: ModelSize = ModelSize.MEDIUM

    @model_validator(mode="after")
    def validate_options(self) -> Self:
        return self
