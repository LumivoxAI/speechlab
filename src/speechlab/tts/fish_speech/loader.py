from typing import Callable
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from fish_speech.models.vqgan.inference import load_model as load_vqgan_model
from fish_speech.models.vqgan.modules.firefly import FireflyArchitecture
from fish_speech.models.text2semantic.inference import load_model

from .model import FishSpeechModel
from .config import FishSpeechConfig


class FishSpeechModelLoader:
    def __init__(self, models_dir: Path | str) -> None:
        self._root_dir = Path(models_dir) / "fish_speech"

    def _download(self, model_version: str) -> Path:
        local_dir = self._root_dir / model_version
        local_dir.mkdir(parents=True, exist_ok=True)
        model_path = snapshot_download(
            repo_id=f"fishaudio/fish-speech-{model_version}",
            local_dir=str(local_dir),
        )

        return Path(model_path)

    def _load_llama(
        self,
        model_path: Path,
        device: str,
        precision: torch.dtype,
        compile: bool,
    ) -> tuple[torch.nn.Module, Callable]:
        try:
            model, decode_one_token = load_model(
                str(model_path), device, precision, compile=compile
            )
            with torch.device(device):
                model.setup_caches(
                    max_batch_size=1,
                    max_seq_len=model.config.max_seq_len,
                    dtype=next(model.parameters()).dtype,
                )
            return model, decode_one_token
        except Exception as e:
            raise ValueError(f"Failed to load LLAMA model: {e}")

    def _load_vqgan(
        self,
        model_path: Path,
        vqgan_config: str,
        device: str,
    ) -> FireflyArchitecture:
        try:
            return load_vqgan_model(
                config_name=vqgan_config,
                checkpoint_path=str(model_path),
                device=device,
            )
        except Exception as e:
            raise ValueError(f"Failed to load VQ-GAN model: {e}")

    # def _warm_up_model(self, model: GigaAMModel) -> None:
    #     data = torch.tensor(generate_warmup_audio_f32n(16000, 20), dtype=model._dtype)
    #     model.stt(data)

    def get_model(self, config: FishSpeechConfig) -> FishSpeechModel:
        FishSpeechConfig.model_validate(config)
        device = config.device.value

        import warnings

        warnings.simplefilter(action="ignore", category=FutureWarning)

        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        if device == "cuda":
            torch.backends.cudnn.enabled = True

        precision = torch.half if config.half_precision else torch.bfloat16
        model_path = self._download(config.version.value)
        llama, decode_one_token = self._load_llama(
            model_path,
            device,
            precision,
            config.compile,
        )
        vqgan = self._load_vqgan(
            model_path / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
            "firefly_gan_vq",
            device,
        )

        model = FishSpeechModel(
            llama=llama,
            decode_one_token=decode_one_token,
            vqgan=vqgan,
            precision=precision,
            compile=config.compile,
        )

        # self._warm_up_model(model)

        return model
