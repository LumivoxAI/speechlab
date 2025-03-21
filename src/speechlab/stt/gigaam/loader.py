from pathlib import Path

import hydra
import torch
from torch import nn
from omegaconf.dictconfig import DictConfig

from .model import GigaAMModel
from .config import GigaAMConfig
from ...utils.audio import generate_warmup_audio_f32n
from ...utils.download import download_file

_URL_BASE = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM"


class GigaAMModelLoader:
    def __init__(self, models_dir: Path | str) -> None:
        self._root_dir = Path(models_dir) / "gigaam"
        self._hashes = {
            "v2_rnnt.ckpt": "3d5674d8b59813d455e34c8ce1c5a7ca4da63fa0f32bcd32b1c37a1224d17b8b",
        }

    def _download(self, model_name: str) -> tuple[Path, Path | None]:
        model_file_name = model_name + ".ckpt"
        model_url = f"{_URL_BASE}/{model_file_name}"
        model_path = self._root_dir / model_file_name
        download_file(model_url, model_path, self._hashes[model_file_name])

        tokenizer_path = None

        return model_path, tokenizer_path

    def _load_preprocessor(self, cfg: DictConfig) -> nn.Module:
        """
        cfg = {
            '_target_': 'speechlab.stt.gigaam.preprocess.FeatureExtractor',
            'sample_rate': 16000,
            'features': 64,
        }
        """
        cfg["_target_"] = "speechlab.stt.gigaam.preprocess.FeatureExtractor"
        return hydra.utils.instantiate(cfg)

    def _load_encoder(
        self,
        cfg: DictConfig,
        half_encoder: bool,
    ) -> nn.Module:
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

    def _load_head(self, cfg: DictConfig) -> nn.Module:
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

    def _load_decoding(
        self,
        cfg: DictConfig,
        tokenizer_path: str | Path | None,
    ) -> nn.Module:
        """
        cfg = {
            '_target_': 'speechlab.stt.gigaam.decoding.RNNTGreedyDecoding',
            'vocabulary': [' ', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я']
        }
        """
        cfg["_target_"] = "speechlab.stt.gigaam.decoding.RNNTGreedyDecoding"
        if tokenizer_path is not None:
            cfg.model_path = str(tokenizer_path)

        return hydra.utils.instantiate(cfg)

    def _warm_up_model(self, model: GigaAMModel) -> None:
        data = torch.tensor(generate_warmup_audio_f32n(16000, 20), dtype=model._dtype)
        model.stt(data)

    def get_model(self, config: GigaAMConfig) -> GigaAMModel:
        GigaAMConfig.model_validate(config)

        model_name = str(config.model.value)
        model_path, tokenizer_path = self._download(model_name)

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
        decoding = self._load_decoding(ccfg.decoding, tokenizer_path)

        model = GigaAMModel(preprocessor, encoder, head, decoding, device)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model = model.eval().to(device)
        model._dtype = next(model.parameters()).dtype

        self._warm_up_model(model)

        return model
