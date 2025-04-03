import gc
from queue import Empty, Queue
from typing import Callable, Iterator
from threading import Event, Thread

import torch
from numpy import ndarray
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
        self._llama = llama
        self._stop_event = Event()
        self._llama_queue = Queue()
        self._device = device

        start_event = Event()
        self._thread = Thread(
            daemon=True,
            target=self._llama_worker,
            args=(start_event, self._stop_event, decode_one_token),
        )
        self._thread.start()
        start_event.wait()

    def _llama_worker(
        self,
        start_event: Event,
        stop_event: Event,
        decode_one_token: Callable,
    ) -> None:
        llama = self._llama
        queue_get = self._llama_queue.get

        start_event.set()
        while not stop_event.is_set():
            try:
                item: GenerateRequest | None = queue_get(timeout=1.0)
            except Empty:
                continue

            if item is None:
                break

            kwargs = item.request
            response_queue = item.response_queue

            try:
                for chunk in generate_long(
                    model=llama, decode_one_token=decode_one_token, **kwargs
                ):
                    response_queue.put(WrappedGenerateResponse(status="success", response=chunk))
            except Exception as e:
                response_queue.put(WrappedGenerateResponse(status="error", response=e))

    @inference_mode()
    def tts(
        self,
        text: str,
        reference_id: str | None = None,
        seed: int | None = None,
        max_new_tokens: int = 1024,
        top_p: float = 0.9,
        temperature: float = 0.6,
        repetition_penalty: float = 1.2,
    ) -> Iterator[ndarray]:
        """
        Audio is returned in numpy array format.
        The data type is normalized float
        Sampling frequency is 44100 Hz.
        """
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

        try:
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
        except GeneratorExit:
            segments_cnt = 1  # disable RuntimeError if iterator is closed

        # Clean up the memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        if segments_cnt == 0:
            raise RuntimeError("No audio generated, please check the input text.")

    def close(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
            self._thread.join()
            self._stop_event = None
            self._thread = None

        if self._llama is not None:
            self._llama.to("cpu")
            del self._llama
            self._llama = None
            self._llama_queue = None

        if self._audio_mng is not None:
            self._audio_mng.close()
            self._audio_mng = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
