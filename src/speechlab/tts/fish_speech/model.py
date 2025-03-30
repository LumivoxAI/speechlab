import gc
from queue import Queue
from typing import Callable, Generator
from threading import Event, Thread

import torch
from torch import nn, device, inference_mode
from fish_speech.utils import set_seed
from fish_speech.models.text2semantic.inference import (
    GenerateRequest,
    GenerateResponse,
    WrappedGenerateResponse,
    generate_long,
)

from .audio_manager import AudioManager


class FishSpeechModel:
    def __init__(
        self,
        llama: nn.Module,
        audio_mng: AudioManager,
        decode_one_token: Callable,
        device: str | device,
    ) -> None:
        super().__init__()

        self._audio_mng = audio_mng
        self._llama_queue = self._launch_llama_queue(llama, decode_one_token)
        self._device = device

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

    @inference_mode()
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
        prompt_tokens, prompt_texts = [], []
        if reference_id is not None:
            prompt_tokens, prompt_texts = self._audio_mng.load_reference(reference_id)

        if seed is not None:
            set_seed(seed)

        chunk_length = 200
        request = dict(
            device=self._device,
            max_new_tokens=max_new_tokens,
            text=text,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            compile=False,  # Only affects logging
            iterative_prompt=chunk_length > 0,
            chunk_length=chunk_length,
            max_length=4096,
            prompt_tokens=prompt_tokens,
            prompt_text=prompt_texts,
        )
        response_queue = Queue()
        self._llama_queue.put(
            GenerateRequest(
                request=request,
                response_queue=response_queue,
            )
        )

        segments_cnt = 0

        while True:
            wrapped_result: WrappedGenerateResponse = response_queue.get()
            if wrapped_result.status == "error":
                if isinstance(wrapped_result.response, Exception):
                    raise wrapped_result.response
                else:
                    raise Exception("Unknown error")

            if not isinstance(wrapped_result.response, GenerateResponse):
                raise TypeError(
                    "Expected GenerateResponse, got {type(wrapped_result.response).__name__}"
                )

            result: GenerateResponse = wrapped_result.response
            if result.action != "next":
                yield self._audio_mng.get_audio_segment(result)
                segments_cnt += 1
            else:
                break

        # Clean up the memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        if segments_cnt == 0:
            raise RuntimeError("No audio generated, please check the input text.")
