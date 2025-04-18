from enum import StrEnum
from typing import Self

from pydantic import model_validator

from ...transport.config import BaseConfig


class DeviceOption(StrEnum):
    CUDA = "cuda"
    CPU = "cpu"


class ModelOption(StrEnum):
    RNNT_V2 = "v2_rnnt"


class GigaAMConfig(BaseConfig):
    device: DeviceOption = DeviceOption.CUDA
    model: ModelOption = ModelOption.RNNT_V2
    half_encoder: bool = True
    compile: bool = False  # `pip install triton`

    @model_validator(mode="after")
    def validate_options(self) -> Self:
        if self.device != DeviceOption.CUDA and self.half_encoder:
            raise ValueError("half_encoder can only be True when device is CUDA")

        if self.compile:
            try:
                import triton  # type: ignore

                is_installed = True
            except ImportError:
                is_installed = False

            if not is_installed:
                raise ImportError(
                    "triton is not installed. Please install it with `pip install triton`."
                )

        return self
