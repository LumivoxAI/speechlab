from typing import Any, Self

import numpy as np
import torch

from .model import SAMPLE_RATE, WINDOW_SIZE, SileroModel, validate_chunked_pcm16


class SileroTorchModel(SileroModel):
    def __init__(self, impl: Any, device: torch.device) -> None:
        super().__init__()

        self._impl = impl
        self._device = device

    @staticmethod
    def load(model_path: str, device_name: str) -> Self:
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        device = torch.device(device_name)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")

            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        impl = torch.jit.load(model_path, map_location=device)
        impl.eval()

        model = SileroTorchModel(impl, device)
        model._warm_up()

        return model

    @torch.inference_mode()
    def __call__(self, data: np.ndarray) -> list[float]:
        validate_chunked_pcm16(data)

        impl = self._impl
        speech_probs = []
        torch_frame = torch.from_numpy(data.astype(np.float32) / 32768.0).to(self._device)
        for i in range(0, len(torch_frame), WINDOW_SIZE):
            torch_chunk = torch_frame[i : i + WINDOW_SIZE]
            speech_probs.append(float(impl(torch_chunk, SAMPLE_RATE).item()))

        return speech_probs

    def _warm_up(self) -> None:
        _ = self(np.zeros((WINDOW_SIZE * 10,), dtype=np.int16))
        self.reset()

    def reset(self) -> None:
        self._impl.reset_states()

    def clear(self) -> None:
        if self._impl is None:
            return

        import gc

        if self._device.type == "cuda":
            self._impl.cpu()

        del self._impl
        self._impl = None
        gc.collect()

        if self._device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
