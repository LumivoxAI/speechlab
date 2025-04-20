from pathlib import Path

import yaml
import torch

from .model import SileroModel
from .config import ModelVersion, SileroConfig
from ...utils.download import download_file
from ...transport.loader import BaseLoader

_MODEL_LIST_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-models/refs/heads/master/models.yml"
)


class SileroModelLoader(BaseLoader[SileroConfig]):
    def __init__(self, data_dir: Path | str) -> None:
        super().__init__(SileroConfig, data_dir, "silero")

        self._hashes = {
            ModelVersion.RU_V3.name: "bf2bcab8e814edb17569503b23bd74e8cc8f584b0d2f7c7e08e2720cc48dc08c",
            ModelVersion.RU_V3_1.name: "cf60b47ec8a9c31046021d2d14b962ea56b8a5bf7061c98accaaaca428522f85",
            ModelVersion.RU_V4.name: "896ab96347d5bd781ab97959d4fd6885620e5aab52405d3445626eb7c1414b00",
        }

    def _download_models_list(self, file_path: Path) -> None:
        from requests import get

        response = get(_MODEL_LIST_URL, timeout=10)
        response.raise_for_status()
        with open(file_path, "w") as f:
            f.write(response.text)

    def _get_package_url(self, version: ModelVersion) -> str:
        file_path = self.model_dir / "model-list.json"
        if not file_path.is_file():
            self._download_models_list(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            actual_name = version.value

            found = False
            for lang_models in yaml.safe_load(f).get("tts_models").values():
                for name, models in lang_models.items():
                    if name != actual_name:
                        continue
                    if found:
                        raise ValueError(f"Found multiple models for {name}")

                    found = True
                    model = models.get("latest")

                    rates = model.get("sample_rate")
                    for expected_rate in [8000, 24000, 48000]:
                        if expected_rate not in rates:
                            raise ValueError(f"Fount model {name} without rate {rates}")

                    return model.get("package")

    def _download(self, version: ModelVersion) -> Path:
        model_url = self._get_package_url(version)

        local_dir = self.model_dir / version.name
        local_dir.mkdir(parents=True, exist_ok=True)
        model_path = local_dir / "model.pt"

        download_file(model_url, model_path, self._hashes[version.name])

        return model_path

    def _warm_up_model(self, model: SileroModel) -> None:
        _ = list(
            model.tts(
                "Привет, мир!",
                "xenia",
                24000,
            )
        )

    def from_config(self, config: SileroConfig) -> SileroModel:
        SileroConfig.model_validate(config)
        device = config.device.value

        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        if device == "cuda":
            torch.backends.cudnn.enabled = True

        model_path = self._download(config.version)
        impl = torch.package.PackageImporter(model_path).load_pickle("tts_models", "model")
        impl.to(device)

        model = SileroModel(impl)
        self._warm_up_model(model)

        return model
