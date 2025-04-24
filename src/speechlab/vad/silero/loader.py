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
            ModelVersion.V6_0: {
                TORCH_FILE_NAME: "b178b4bc39467ecd5e9eb0bc6f2fcc291c1cc1cc55a68fea8c0c2c0d5fbbe66c",
                ONNX_FILE_NAME: None,
            },
            ModelVersion.V5_1_2: {
                TORCH_FILE_NAME: "85c48e1f0ecb604e5d2a268f3ccfb912d4f7e935acdc86af5a3fc5b0aea7b29a",
                ONNX_FILE_NAME: None,
            },
            ModelVersion.V4_0: {
                TORCH_FILE_NAME: "082e21870cf7722b0c7fa5228eaed579efb6870df81192b79bed3f7bac2f738a",
                ONNX_FILE_NAME: None,
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
        SileroConfig.model_validate(config)
        model_path = self._download(config)

        if config.device.is_torch:
            from .torch_model import SileroTorchModel

            return SileroTorchModel.load(model_path, config.device.torch_device_name)

        raise NotImplementedError
