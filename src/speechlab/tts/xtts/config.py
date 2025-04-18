from enum import StrEnum
from typing import Self

from pydantic import model_validator

from ...transport.config import BaseConfig


class DeviceOption(StrEnum):
    CUDA = "cuda"
    CPU = "cpu"


class ModelVersion(StrEnum):
    MULTILINGUAL = "coqui/XTTS-v2"
    RU_IPA = "omogr/xtts-ru-ipa"
    DONU = "NeuroDonu/RU-XTTS-DonuModel"


class XTTSConfig(BaseConfig):
    device: DeviceOption = DeviceOption.CUDA
    version: ModelVersion = ModelVersion.MULTILINGUAL
    use_deepspeed: bool = False

    @model_validator(mode="after")
    def validate_options(self) -> Self:
        if self.use_deepspeed:
            try:
                import deepspeed  # type: ignore
            except ImportError:
                raise ImportError(
                    "deepspeed is not installed. Please install it with `pip install deepspeed`."
                )

        return self
