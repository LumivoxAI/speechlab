import io
from typing import Any
from pathlib import Path

import torchaudio
from numpy import ndarray
from torch import nn, mean, dtype, tensor
from fish_speech.utils import autocast_exclude_mps
from fish_speech.utils.file import (
    AUDIO_EXTENSIONS,
    list_files,
    read_ref_text,
    audio_to_bytes,
)
from fish_speech.inference_engine import VQManager
from fish_speech.models.vqgan.modules.firefly import FireflyArchitecture
from fish_speech.models.text2semantic.inference import GenerateResponse


class AudioManager(VQManager):
    def __init__(
        self,
        root_dir: Path,
        vqgan: nn.Module,
        precision: dtype,
    ) -> None:
        if not isinstance(vqgan, FireflyArchitecture):
            raise ValueError(f"Unknown model type: {type(vqgan)}")
        if vqgan.spec_transform.sample_rate != 44100:
            raise ValueError(f"FishSpeech samplerate mismatch: {vqgan.spec_transform.sample_rate}")

        self._ref_by_id: dict = {}
        self._root_dir = root_dir
        self.decoder_model = vqgan  # for VQManager
        self._device = vqgan.device
        self._precision = precision

        backends = torchaudio.list_audio_backends()
        if "ffmpeg" in backends:
            self._backend = "ffmpeg"
        else:
            self._backend = "soundfile"

    def load_reference(self, id: str) -> tuple:
        if id in self._ref_by_id:
            return self._ref_by_id[id]

        ref_folder = self._root_dir / id
        ref_folder.mkdir(parents=True, exist_ok=True)
        ref_audios = list_files(ref_folder, AUDIO_EXTENSIONS, recursive=True, sort=False)

        prompt_tokens = [
            self.encode_reference(
                reference_audio=audio_to_bytes(str(ref_audio)),
                enable_reference_audio=True,
            )
            for ref_audio in ref_audios
        ]
        prompt_texts = [
            read_ref_text(str(ref_audio.with_suffix(".lab"))) for ref_audio in ref_audios
        ]
        self._ref_by_id[id] = (prompt_tokens, prompt_texts)

        return prompt_tokens, prompt_texts

    def get_audio_segment(self, result: GenerateResponse) -> ndarray:
        codes = result.codes
        # Don't use autocast on MPS devices
        with autocast_exclude_mps(device_type=self._device.type, dtype=self._precision):
            feature_lengths = tensor([codes.shape[1]], device=self._device)

            segment = self.decoder_model.decode(
                indices=codes[None],
                feature_lengths=feature_lengths,
            )[0].squeeze()

        return segment.float().cpu().numpy()

    # for VQManager
    def load_audio(self, reference_audio, sr) -> ndarray | Any:
        if len(reference_audio) > 255 or not Path(reference_audio).exists():
            audio_data = reference_audio
            reference_audio = io.BytesIO(audio_data)

        waveform, original_sr = torchaudio.load(reference_audio, backend=self._backend)

        if waveform.shape[0] > 1:
            waveform = mean(waveform, dim=0, keepdim=True)

        if original_sr != sr:
            resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=sr)
            waveform = resampler(waveform)

        audio = waveform.squeeze().numpy()
        return audio

    def close(self) -> None:
        if self._ref_by_id is not None:
            self._ref_by_id.clear()
            self._ref_by_id = None

        if self._root_dir is not None:
            self._device = None
            self._precision = None
            self.decoder_model.to("cpu")
            del self.decoder_model
            self.decoder_model = None
