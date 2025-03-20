from enum import StrEnum
from typing import Self

from pydantic import BaseModel, model_validator


class DeviceOption(StrEnum):
    CUDA = "cuda"
    CPU = "cpu"


class ModelOption(StrEnum):
    RNNT_V2 = "v2_rnnt"


class GigaAMConfig(BaseModel):
    device: DeviceOption = DeviceOption.CUDA
    model: ModelOption = ModelOption.RNNT_V2
    half_encoder: bool = True
    flash_attn: bool = False

    @model_validator(mode="after")
    def validate_options(self) -> Self:
        if self.device != DeviceOption.CUDA:
            if self.half_encoder:
                raise ValueError("half_encoder can only be True when device is CUDA")
            if self.flash_attn:
                raise ValueError("flash_attn can only be True when device is CUDA")

        return self
