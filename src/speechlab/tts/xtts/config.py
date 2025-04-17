from enum import StrEnum
from typing import Self

from pydantic import BaseModel, model_validator


class DeviceOption(StrEnum):
    CUDA = "cuda"
    CPU = "cpu"


class ModelVersion(StrEnum):
    Multilingual = "coqui/XTTS-v2"
    RuIPA = "omogr/xtts-ru-ipa"
    Donu = "NeuroDonu/RU-XTTS-DonuModel"


class XTTSConfig(BaseModel):
    device: DeviceOption = DeviceOption.CUDA
    version: ModelVersion = ModelVersion.Multilingual
    use_deepspeed: bool = True

    @model_validator(mode="after")
    def validate_options(self) -> Self:
        return self
