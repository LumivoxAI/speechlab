from queue import Queue
from typing import Callable, Generator
from threading import Event, Thread

from torch import nn, dtype
from fish_speech.utils.schema import ServeTTSRequest
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.text2semantic.inference import (
    GenerateRequest,
    WrappedGenerateResponse,
    generate_long,
)


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

    @property
    def samplerate(self) -> int:
        return self.inference_engine.decoder_model.spec_transform.sample_rate

    def tts(
        self,
        text: str,
        reference_id: str | None = None,
        seed: int | None = None,
        max_new_tokens: int = 1024,
        top_p: float = 0.7,
        repetition_penalty: float = 1.2,
        temperature: float = 0.7,
    ) -> Generator:
        request = ServeTTSRequest(
            text=text,
            chunk_length=200,  # Chunk length for streaming. Defaults to 200.
            format="wav",
            references=[],
            reference_id=reference_id,
            seed=seed,
            use_memory_cache="on",
            normalize=False,
            streaming=True,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
        )

        for result in self.inference_engine.inference(request):
            match result.code:
                case "header":
                    pass  # skip block with samplerate

                case "error":
                    if isinstance(result.error, Exception):
                        raise result.error
                    else:
                        raise RuntimeError("Unknown error")

                case "segment":
                    yield result.audio[1]  # audio[0] == samplerate

                case "final":
                    pass  # skip block with all segments
