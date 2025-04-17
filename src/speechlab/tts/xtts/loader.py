from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig

from .model import XTTSModel
from .config import XTTSConfig, ModelVersion


class XTTSModelLoader:
    def __init__(self, model_dir: Path | str, reference_dir: Path | str) -> None:
        self._model_dir = Path(model_dir) / "xtts"
        self._reference_dir = Path(reference_dir) / "fish_speech"

    def _download(self, version: ModelVersion) -> Path:
        local_dir = self._model_dir / version.name
        local_dir.mkdir(parents=True, exist_ok=True)
        model_path = snapshot_download(
            repo_id=version.value,
            local_dir=str(local_dir),
            allow_patterns=[
                "config.json",
                "model.pth",
                "speakers_xtts.pth",
                "vocab.json",
                "hash.md5",
            ],
        )

        return Path(model_path)

    def _warm_up_model(self, model: XTTSModel) -> None:
        _ = list(
            model.tts(
                "Привет, мир!",
                speaker_id=model._impl.speaker_manager.speaker_names[0],
            )
        )

    def get_model(self, config: XTTSConfig) -> XTTSModel:
        XTTSConfig.model_validate(config)
        device = config.device.value

        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        if device == "cuda":
            torch.backends.cudnn.enabled = True

        model_path = self._download(config.version)
        speaker_file_path = model_path / "speakers_xtts.pth"
        is_speaker_file = speaker_file_path.exists()

        xtts_config = XttsConfig()
        xtts_config.load_json(model_path / "config.json")
        impl = Xtts.init_from_config(xtts_config)
        impl.load_checkpoint(
            xtts_config,
            checkpoint_dir=str(model_path),
            checkpoint_path=str(model_path / "model.pth"),
            vocab_path=str(model_path / "vocab.json"),
            speaker_file_path=str(speaker_file_path) if is_speaker_file else "-",
            eval=True,
            strict=True,
            use_deepspeed=config.use_deepspeed,
        )

        text_preprocessor = lambda text: text
        if config.version == ModelVersion.RuIPA:
            from omogre import Transcriptor

            transcriptor = Transcriptor(data_path=self._model_dir / "omogre")
            text_preprocessor = lambda text: " ".join(transcriptor([text]))

        if device == "cuda":
            impl.cuda()
        else:
            impl.cpu()

        model = XTTSModel(impl, text_preprocessor, self._reference_dir)
        if is_speaker_file:
            self._warm_up_model(model)

        return model
