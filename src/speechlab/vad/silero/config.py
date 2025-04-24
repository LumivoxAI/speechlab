from enum import StrEnum
from typing import Self

from pydantic import model_validator

from ...transport.config import BaseConfig


class DeviceOption(StrEnum):
    TORCH_CUDA = "torch_cuda"
    TORCH_CPU = "torch_cpu"
    ONNX_GPU = "onnx_gpu"
    ONNX_CPU = "onnx_cpu"

    @property
    def is_torch(self) -> bool:
        return self == DeviceOption.TORCH_CUDA or self == DeviceOption.TORCH_CPU

    @property
    def torch_device_name(self) -> str:
        return "cuda" if self == DeviceOption.TORCH_CUDA else "cpu"

    @property
    def is_onnx(self) -> bool:
        return self == DeviceOption.ONNX_GPU or self == DeviceOption.ONNX_CPU


class ModelVersion(StrEnum):
    # file_names: silero_vad.jit and silero_vad.onnx
    V6_0 = "https://github.com/snakers4/silero-vad/raw/refs/tags/v6.0/src/silero_vad/data/"
    V5_1_2 = "https://github.com/snakers4/silero-vad/raw/refs/tags/v5.1.2/src/silero_vad/data/"
    V4_0 = "https://github.com/snakers4/silero-vad/raw/refs/tags/v4.0/files/"


class SileroConfig(BaseConfig):
    device: DeviceOption = DeviceOption.TORCH_CUDA
    version: ModelVersion = ModelVersion.V6_0

    @model_validator(mode="after")
    def validate_options(self) -> Self:
        return self
