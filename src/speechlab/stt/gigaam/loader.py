from pathlib import Path

import hydra
import torch
from omegaconf.dictconfig import DictConfig

from .model import GigaAMModel
from .config import GigaAMConfig
from ...utils.audio import generate_warmup_audio_f32n
from ...utils.download import download_file
from .mel_spectrogram.torch import MelSpectrogram

_URL_BASE = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM"


class GigaAMModelLoader:
    def __init__(self, models_dir: Path | str) -> None:
        self._root_dir = Path(models_dir) / "gigaam"
        self._hashes = {
            "v2_rnnt.ckpt": "3d5674d8b59813d455e34c8ce1c5a7ca4da63fa0f32bcd32b1c37a1224d17b8b",
        }

    def _download(self, model_name: str) -> Path:
        model_file_name = model_name + ".ckpt"
        model_url = f"{_URL_BASE}/{model_file_name}"
        model_path = self._root_dir / model_file_name
        download_file(model_url, model_path, self._hashes[model_file_name])

        return model_path

    def _load_preprocessor(self, cfg: DictConfig) -> MelSpectrogram:
        """
        cfg = {
            '_target_': 'speechlab.stt.gigaam.preprocess.FeatureExtractor',
            'sample_rate': 16000,
            'features': 64,
        }
        """
        assert cfg.sample_rate == 16000
        assert cfg.features == 64
        return MelSpectrogram()

    def _load_encoder(
        self,
        cfg: DictConfig,
        half_encoder: bool,
    ) -> torch.nn.Module:
        """
        cfg = {
            '_target_': 'speechlab.stt.gigaam.encoder.ConformerEncoder',
            'feat_in': 64,
            'n_layers': 16,
            'd_model': 768,
            'subsampling_factor': 4,
            'ff_expansion_factor': 4,
            'self_attention_model': 'rotary',
            'pos_emb_max_len': 5000,
            'n_heads': 16,
            'conv_kernel_size': 31,
            'flash_attn': False
        }
        """
        cfg["_target_"] = "speechlab.stt.gigaam.encoder.ConformerEncoder"
        del cfg["self_attention_model"]
        del cfg["flash_attn"]
        encoder = hydra.utils.instantiate(cfg)
        if half_encoder:
            encoder = encoder.half()

        return encoder

    def _load_head(self, cfg: DictConfig) -> torch.nn.Module:
        """
        cfg = {
            '_target_': 'speechlab.stt.gigaam.decoder.RNNTHead',
            'decoder': {
                'pred_hidden': 320,
                'pred_rnn_layers': 1,
                'num_classes': 34
            },
            'joint': {
                'enc_hidden': 768,
                'pred_hidden': 320,
                'joint_hidden': 320,
                'num_classes': 34
            }
        }
        """
        cfg["_target_"] = "speechlab.stt.gigaam.decoder.RNNTHead"
        return hydra.utils.instantiate(cfg)

    def _check_decoding(
        self,
        cfg: DictConfig,
    ) -> None:
        """
        cfg = {
            '_target_': 'speechlab.stt.gigaam.decoding.RNNTGreedyDecoding',
            'vocabulary': [' ', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я']
        }
        """
        assert len(cfg.vocabulary) == 33  # without `ё` + `blank`

    def _warm_up_model(self, model: GigaAMModel) -> None:
        data = torch.tensor(generate_warmup_audio_f32n(16000, 20), dtype=model._dtype)
        model.stt(data)

    def get_model(self, config: GigaAMConfig) -> GigaAMModel:
        GigaAMConfig.model_validate(config)

        if not hydra.__version__.startswith("1.3"):
            raise RuntimeError(
                f"Hydra version {hydra.__version__} is not supported. Please use version 1.3.x."
            )

        model_name = str(config.model.value)
        model_path = self._download(model_name)

        checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")
        checkpoint["cfg"].model_name = model_name
        ccfg = checkpoint["cfg"]

        device = torch.device(config.device.value)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        if device.type == "cuda":
            torch.backends.cudnn.enabled = True

        preprocessor = self._load_preprocessor(ccfg.preprocessor)
        encoder = self._load_encoder(ccfg.encoder, config.half_encoder)
        head = self._load_head(ccfg.head)
        self._check_decoding(ccfg.decoding)

        model = GigaAMModel(preprocessor, encoder, head, device)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model.init()
        model = model.eval().to(device)

        if config.compile:
            model.encoder = torch.compile(
                model.encoder,
                backend="inductor",
                mode="default",
                fullgraph=True,
            )
            model.head = torch.compile(
                model.head,
                backend="aot_eager",
                dynamic=True,
            )

        self._warm_up_model(model)

        return model
