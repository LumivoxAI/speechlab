from pathlib import Path

from .model import SileroModel
from .config import ModelVersion, SileroConfig
from ...utils.download import download_file
from ...transport.loader import BaseLoader

TORCH_FILE_NAME = "silero_vad.jit"
ONNX_FILE_NAME = "silero_vad.onnx"


class SileroModelLoader(BaseLoader[SileroConfig]):
    def __init__(self, data_dir: Path | str) -> None:
        super().__init__(SileroConfig, data_dir, "silero_vad")

        self._hashes = {
            ModelVersion.V6_2_1: {
                TORCH_FILE_NAME: "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720",
                ONNX_FILE_NAME: "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
            },
            ModelVersion.V5_1_2: {
                TORCH_FILE_NAME: "85c48e1f0ecb604e5d2a268f3ccfb912d4f7e935acdc86af5a3fc5b0aea7b29a",
                ONNX_FILE_NAME: "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f",
            },
            ModelVersion.V4_0: {
                TORCH_FILE_NAME: "082e21870cf7722b0c7fa5228eaed579efb6870df81192b79bed3f7bac2f738a",
                ONNX_FILE_NAME: "a35ebf52fd3ce5f1469b2a36158dba761bc47b973ea3382b3186ca15b1f5af28",
            },
        }

    def _download(self, config: SileroConfig) -> Path:
        model_file_name = TORCH_FILE_NAME if config.device.is_torch else ONNX_FILE_NAME
        local_dir = self.model_dir / config.version.name
        local_dir.mkdir(parents=True, exist_ok=True)
        model_path = local_dir / model_file_name
        model_url = config.version.value + model_file_name

        download_file(model_url, model_path, self._hashes[config.version][model_file_name])

        return model_path

    def from_config(self, config: SileroConfig) -> SileroModel:
        config = SileroConfig.model_validate(config)
        model_path = self._download(config)
        device = config.device

        if device.is_torch:
            from .torch_model import SileroTorchModel

            return SileroTorchModel.load(model_path, device.torch_device_name)

        if device.is_onnx:
            from .onnx_model import SileroOnnxModel

            return SileroOnnxModel.load(model_path, device)

        raise ValueError(f"Unsupported device: {device}")
