from enum import StrEnum
from typing import Self

from pydantic import BaseModel, model_validator


class DeviceOption(StrEnum):
    CUDA = "cuda"
    CPU = "cpu"


class ModelVersion(StrEnum):
    V1_5 = "1.5"


class FishSpeechConfig(BaseModel):
    device: DeviceOption = DeviceOption.CUDA
    version: ModelVersion = ModelVersion.V1_5
    half_precision: bool = True
    compile: bool = True  # `pip install triton`

    @model_validator(mode="after")
    def validate_options(self) -> Self:
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
