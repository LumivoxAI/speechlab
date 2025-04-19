import os.path
from typing import Callable
from pathlib import Path

import torch
import fish_speech
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from huggingface_hub import snapshot_download
from fish_speech.models.text2semantic.inference import load_model

from .model import FishSpeechModel
from .config import FishSpeechConfig
from .audio_manager import AudioManager
from ...transport.loader import BaseLoader


class FishSpeechModelLoader(BaseLoader[FishSpeechConfig]):
    def __init__(self, data_dir: Path | str) -> None:
        super().__init__(FishSpeechConfig, data_dir, "fish_speech")

    def _download(self, model_version: str) -> Path:
        local_dir = self.model_dir / model_version
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
        # from fish_speech.models.text2semantic.inference.launch_thread_safe_queue
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
    ) -> torch.nn.Module:
        # from fish_speech.models.vqgan.inference.load_model
        # result == fish_speech.models.vqgan.modules.firefly.FireflyArchitecture
        try:
            config_dir = os.path.join(fish_speech.__path__[0], "configs")
            with initialize_config_dir(version_base="1.3", config_dir=config_dir):
                cfg = compose(config_name=vqgan_config)

            model = instantiate(cfg)
            state_dict = torch.load(
                str(model_path), map_location=device, mmap=True, weights_only=True
            )
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            if any("generator" in k for k in state_dict):
                state_dict = {
                    k.replace("generator.", ""): v
                    for k, v in state_dict.items()
                    if "generator." in k
                }

            model.load_state_dict(state_dict, strict=False, assign=True)
            model.eval()
            model.to(device)

            return model
        except Exception as e:
            raise ValueError(f"Failed to load VQ-GAN model: {e}")

    def _warm_up_model(self, model: FishSpeechModel) -> None:
        _ = list(model.tts("Привет, мир!"))

    def from_config(self, config: FishSpeechConfig) -> FishSpeechModel:
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
        audio_mng = AudioManager(self.reference_dir, vqgan, precision)

        model = FishSpeechModel(
            llama=llama,
            audio_mng=audio_mng,
            decode_one_token=decode_one_token,
            device=vqgan.device,
        )

        self._warm_up_model(model)

        return model
