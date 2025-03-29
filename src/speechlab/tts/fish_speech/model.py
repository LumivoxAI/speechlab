from queue import Queue
from typing import Union, Callable, Generator
from threading import Event, Thread

import numpy as np
from torch import Tensor, nn, dtype
from pydantic import BaseModel
from fish_speech.utils.file import audio_to_bytes
from fish_speech.utils.schema import ServeTTSRequest
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.text2semantic.inference import (
    GenerateRequest,
    WrappedGenerateResponse,
    generate_long,
)


class Reference(BaseModel):
    tokens: Tensor
    text: str

    # Allow arbitrary types for pytorch related types
    class Config:
        arbitrary_types_allowed = True


class FishSpeechModel:
    def __init__(
        self,
        llama: nn.Module,
        decode_one_token: Callable,
        vqgan: nn.Module,
        precision: dtype,
        compile: bool,
    ) -> None:
        super().__init__()
        llama_queue = self._launch_llama_queue(llama, decode_one_token)

        self.inference_engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=vqgan,
            precision=precision,
            compile=compile,
        )
        self.warmup(self.inference_engine)

    def _launch_llama_queue(
        self,
        llama: nn.Module,
        decode_one_token: Callable,
    ) -> Queue:
        input_queue = Queue()
        init_event = Event()

        def worker() -> None:
            init_event.set()

            while True:
                item: GenerateRequest | None = input_queue.get()
                if item is None:
                    break

                kwargs = item.request
                response_queue = item.response_queue

                try:
                    for chunk in generate_long(
                        model=llama, decode_one_token=decode_one_token, **kwargs
                    ):
                        response_queue.put(
                            WrappedGenerateResponse(status="success", response=chunk)
                        )
                except Exception as e:
                    response_queue.put(WrappedGenerateResponse(status="error", response=e))

        Thread(target=worker, daemon=True).start()
        init_event.wait()

        return input_queue

    def warmup(self, inference_engine: TTSInferenceEngine) -> None:
        """Warm up the inference engine."""
        try:
            list(
                inference_engine.inference(
                    ServeTTSRequest(
                        text="Hello world.",
                        references=[],
                        reference_id=None,
                        max_new_tokens=1024,
                        chunk_length=200,
                        top_p=0.7,
                        repetition_penalty=1.5,
                        temperature=0.7,
                        format="wav",
                    )
                )
            )
        except Exception as e:
            raise ValueError(f"Failed to warm up the inference engine: {e}")

    @property
    def sample_rate(self) -> int:
        """Get the sample rate of the audio."""
        return self.inference_engine.decoder_model.spec_transform.sample_rate

    def make_reference(self, audio_path: str, text: str) -> Reference:
        """Create a reference object from audio and text."""
        audio_bytes = audio_to_bytes(audio_path)
        if audio_bytes is None:
            raise ValueError("Failed to load audio file.")

        tokens = self.inference_engine.encode_reference(audio_bytes, True)
        return Reference(tokens=tokens, text=text)

    def generate_streaming(
        self,
        text: str,
        references: Union[list[Reference], Reference] = [],
        seed: int | None = None,
        streaming: bool = False,
        max_new_tokens: int = 0,
        chunk_length: int = 200,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        temperature: float | None = None,
    ) -> Generator:
        """
        Generate audio from text.

        Args:
            text (str): Text to generate audio from.
            references (Union[List[Reference], Reference], optional): List of reference audios. Defaults to [].
            seed (Optional[int], optional): Random seed. Defaults to None.
            streaming (bool, optional): Stream the audio. Defaults to False.
            max_new_tokens (int, optional): Maximum number of tokens. Defaults to 0 (no limit).
            chunk_length (int, optional): Chunk length for streaming. Defaults to 200.
            top_p (Optional[float], optional): Top-p sampling. Defaults to None.
            repetition_penalty (Optional[float], optional): Repetition penalty. Defaults to None.
            temperature (Optional[float], optional): Sampling temperature. Defaults to None.
        """
        references = [references] if isinstance(references, Reference) else references

        request = ServeTTSRequest(
            text=text,
            preprocessed_references=references,
            seed=seed,
            streaming=streaming,
            max_new_tokens=max_new_tokens,
            chunk_length=chunk_length,
            top_p=top_p or 0.7,
            repetition_penalty=repetition_penalty or 1.2,
            temperature=temperature or 0.7,
        )

        count = 0
        for result in self.inference_engine.inference(request):
            match result.code:
                case "header":
                    pass  # In this case, we only want to yield the audio (amplitude)
                    # User can save with a library like soundfile if needed

                case "error":
                    if isinstance(result.error, Exception):
                        raise result.error
                    else:
                        raise RuntimeError("Unknown error")

                case "segment":
                    count += 1
                    if isinstance(result.audio, tuple) and streaming:
                        yield result.audio[1]

                case "final":
                    count += 1
                    if isinstance(result.audio, tuple) and not streaming:
                        yield result.audio[1]

        if count == 0:
            raise RuntimeError("No audio generated, please check the input text.")

    def generate(
        self,
        text: str,
        references: Union[list[Reference], Reference] = [],
        seed: int | None = None,
        streaming: bool = False,
        max_new_tokens: int = 0,
        chunk_length: int = 200,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        temperature: float | None = None,
    ) -> Union[Generator, np.ndarray]:
        """
        Wrapper for the generate_streaming method.
        Returns either a generator or directly the final audio.

        Args:
            text (str): Text to generate audio from.
            references (Union[List[Reference], Reference], optional): List of reference audios. Defaults to [].
            seed (Optional[int], optional): Random seed. Defaults to None.
            streaming (bool, optional): Stream the audio. Defaults to False.
            max_new_tokens (int, optional): Maximum number of tokens. Defaults to 0 (no limit).
            chunk_length (int, optional): Chunk length for streaming. Defaults to 200.
            top_p (Optional[float], optional): Top-p sampling. Defaults to None.
            repetition_penalty (Optional[float], optional): Repetition penalty. Defaults to None.
            temperature (Optional[float], optional): Sampling temperature. Defaults to None.
        """

        generator = self.generate_streaming(
            text=text,
            references=references,
            seed=seed,
            streaming=streaming,
            max_new_tokens=max_new_tokens,
            chunk_length=chunk_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
        )

        if streaming:
            return generator
        else:
            audio = np.concatenate(list(generator))
            return audio
