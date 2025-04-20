from enum import StrEnum
from typing import Self

from pydantic import model_validator

from ...transport.config import BaseConfig


class DeviceOption(StrEnum):
    CUDA = "cuda"
    CPU = "cpu"


class ModelVersion(StrEnum):
    RU_V4 = "v4_ru"
    RU_V3_1 = "v3_1_ru"
    RU_V3 = "ru_v3"


class SileroConfig(BaseConfig):
    device: DeviceOption = DeviceOption.CUDA
    version: ModelVersion = ModelVersion.RU_V4

    @model_validator(mode="after")
    def validate_options(self) -> Self:
        return self
