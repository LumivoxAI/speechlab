from typing import Union
from pathlib import Path

import torch
from torch import Tensor, nn, device, float16, autocast, inference_mode

from .tokenizer import Tokenizer
from ...transport.model import BaseModel
from .mel_spectrogram.torch import MelSpectrogram


class GigaAMModel(nn.Module, BaseModel):
    def __init__(
        self,
        preprocessor: MelSpectrogram,
        encoder: nn.Module,
        head: nn.Module,
        device: device,
    ) -> None:
        super().__init__()

        self._preprocessor = preprocessor
        self.encoder = encoder
        self.head = head
        self._tokenizer = Tokenizer()
        self._device = device
        self._autocast = self._device.type != "cpu"

    def init(self) -> None:
        self._dtype = next(self.parameters()).dtype
        self.encoder.joint_enc = self.head.joint.enc
        self.head.joint.enc = None
        self.head.init()

    @inference_mode()
    def stt(self, data: Tensor) -> str:
        features = self._preprocessor(data.to(self._device))
        with autocast(
            device_type=self._device.type,
            dtype=float16,
            enabled=self._autocast,
        ):
            enc_proj = self.encoder(features)

        tokens = self.head(enc_proj).tolist()
        return self._tokenizer(tokens)

    def onnx_converter(
        self,
        model_name: str,
        module: nn.Module,
        root_dir: Path,
        dynamic_axes: Union[dict[str, list[int]], dict[str, dict[int, str]]] | None = None,
        opset_version: int | None = None,
    ) -> None:
        inputs = module.input_example()
        input_names = module.input_names()
        output_names = module.output_names()
        saved_dtype = next(module.parameters()).dtype

        import warnings

        root_dir.mkdir(exist_ok=True, parents=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=(UserWarning, torch.jit.TracerWarning))
            torch.onnx.export(
                module.to(torch.float32),
                inputs,
                str(root_dir / f"{model_name}.onnx"),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
                export_params=True,
            )

        module.to(saved_dtype)

    def to_onnx(self, model_name: str, root_dir: Path) -> None:
        self.onnx_converter(
            model_name=f"{model_name}_encoder",
            module=self.encoder,
            root_dir=root_dir,
            dynamic_axes=self.encoder.dynamic_axes(),
        )
        self.onnx_converter(
            model_name=f"{model_name}_decoder",
            module=self.head.decoder,
            root_dir=root_dir,
        )
