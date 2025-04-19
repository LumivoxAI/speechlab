import gc
from typing import Callable, Iterator
from pathlib import Path

import torch
from numpy import ndarray
from TTS.tts.models.xtts import Xtts

from ...transport.model import BaseModel

AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".m4a",
    ".wma",
    ".aac",
    ".aiff",
    ".aif",
    ".aifc",
}


class XTTSModel(BaseModel):
    def __init__(self, impl: Xtts, text_preprocessor: Callable, reference_dir: Path) -> None:
        super().__init__()

        self._impl = impl
        self._reference_dir = reference_dir
        self._text_preprocessor = text_preprocessor
        self._speakers_cache: dict[str, tuple] = {}

    def _get_speaker_from_reference(self, reference_id: str) -> tuple:
        ref_folder = self._reference_dir / reference_id
        if not ref_folder.exists():
            raise FileNotFoundError(f"Reference folder {ref_folder} does not exist.")

        files = [str(file) for ext in AUDIO_EXTENSIONS for file in ref_folder.rglob(f"*{ext}")]
        if len(files) == 0:
            raise FileNotFoundError(f"No audio files found in {ref_folder}.")

        gpt_cond_latents, speaker_embedding = self._impl.get_conditioning_latents(
            audio_path=files,
            max_ref_length=self._impl.config.max_ref_len,
            gpt_cond_len=self._impl.config.gpt_cond_len,
            gpt_cond_chunk_len=self._impl.config.gpt_cond_chunk_len,
            librosa_trim_db=None,
            sound_norm_refs=self._impl.config.sound_norm_refs,
            load_sr=self._impl.config.model_args.input_sample_rate,
        )
        self._speakers_cache[reference_id] = (gpt_cond_latents, speaker_embedding)

        return gpt_cond_latents, speaker_embedding

    def _get_speaker(self, speaker_id: str) -> tuple:
        if speaker_id in self._speakers_cache:
            return self._speakers_cache[speaker_id]

        smanager = self._impl.speaker_manager
        if smanager is not None and speaker_id in smanager.speaker_names:
            gpt_cond_latents, speaker_embedding = smanager.speakers[speaker_id].values()
            self._speakers_cache[speaker_id] = (gpt_cond_latents, speaker_embedding)
            return gpt_cond_latents, speaker_embedding

        return self._get_speaker_from_reference(speaker_id)

    @property
    def samplerate(self) -> int:
        return self._impl.config.model_args.output_sample_rate

    def tts(
        self,
        text: str,
        speaker_id: str,
        language: str = "ru",
        temperature: float | None = None,
        length_penalty: float | None = None,
        repetition_penalty: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        speed: float = 1.0,
        enable_text_splitting: bool = False,
    ) -> Iterator[ndarray]:
        """
        Audio is returned in numpy array format.
        The data type is normalized float
        """
        gpt_cond_latent, speaker_embedding = self._get_speaker(speaker_id)
        chunks = self._impl.inference_stream(
            text=self._text_preprocessor(text),
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            # Streaming
            stream_chunk_size=20,
            overlap_wav_len=1024,
            # GPT inference
            temperature=temperature or self._impl.config.temperature,
            length_penalty=length_penalty or self._impl.config.length_penalty,
            repetition_penalty=repetition_penalty or self._impl.config.repetition_penalty,
            top_k=top_k or self._impl.config.top_k,
            top_p=top_p or self._impl.config.top_p,
            do_sample=True,
            speed=speed,
            enable_text_splitting=enable_text_splitting,
        )

        segments_cnt = 0

        try:
            for chunk in chunks:
                if chunk is None:
                    continue
                segments_cnt += 1
                yield chunk.cpu().numpy()
        except GeneratorExit:
            segments_cnt = 1  # disable RuntimeError if iterator is closed

        if segments_cnt == 0:
            raise RuntimeError("No audio generated, please check the input text.")

    def close(self) -> None:
        if self._impl is not None:
            del self._impl
            self._impl = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
