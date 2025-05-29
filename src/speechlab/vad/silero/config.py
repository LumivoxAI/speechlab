from enum import StrEnum
from typing import Self

from pydantic import model_validator

from ...transport.config import BaseConfig


class DeviceOption(StrEnum):
    TORCH_CUDA = "torch_cuda"
    TORCH_CPU = "torch_cpu"
    TORCH_AUTO = "torch_auto"
    ONNX_GPU = "onnx_gpu"
    ONNX_CPU = "onnx_cpu"
    ONNX_AUTO = "onnx_auto"

    @property
    def is_torch(self) -> bool:
        return self in {DeviceOption.TORCH_CUDA, DeviceOption.TORCH_CPU, DeviceOption.TORCH_AUTO}

    @property
    def torch_device_name(self) -> str:
        if self == DeviceOption.TORCH_AUTO:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"

        return "cuda" if self == DeviceOption.TORCH_CUDA else "cpu"

    @property
    def is_onnx(self) -> bool:
        return self in {DeviceOption.ONNX_GPU, DeviceOption.ONNX_CPU, DeviceOption.ONNX_AUTO}

    @property
    def is_auto(self) -> bool:
        return self in {DeviceOption.TORCH_AUTO, DeviceOption.ONNX_AUTO}

    @property
    def onnx_provider_name(self) -> str:
        return "CUDAExecutionProvider" if self == DeviceOption.ONNX_GPU else "CPUExecutionProvider"

    @property
    def onnx_provider_candidates(self) -> list[str]:
        if self == DeviceOption.ONNX_AUTO:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if self == DeviceOption.ONNX_GPU:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]


class ModelVersion(StrEnum):
    # file_names: silero_vad.jit and silero_vad.onnx
    V6_2_1 = "https://github.com/snakers4/silero-vad/raw/refs/tags/v6.2.1/src/silero_vad/data/"
    V5_1_2 = "https://github.com/snakers4/silero-vad/raw/refs/tags/v5.1.2/src/silero_vad/data/"
    V4_0 = "https://github.com/snakers4/silero-vad/raw/refs/tags/v4.0/files/"


class SileroConfig(BaseConfig):
    device: DeviceOption = DeviceOption.TORCH_CUDA
    version: ModelVersion = ModelVersion.V6_2_1

    @model_validator(mode="after")
    def validate_options(self) -> Self:
        return self
