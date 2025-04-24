from typing import Any, Self

import numpy as np
import torch

from .model import SileroModel

_SAMPLERATE = 16000
_WINDOW_SIZE = 512


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
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        impl = torch.jit.load(model_path, map_location=device)
        impl.eval()

        model = SileroTorchModel(impl, device)
        model._warm_up()

        return model

    @torch.inference_mode()
    def __call__(self, data: np.ndarray) -> list[float]:
        """
        The data type np.int16 (S16LE)
        """
        if data.shape[0] % _WINDOW_SIZE != 0:
            raise ValueError(f"Input data length must be a multiple of {_WINDOW_SIZE}")

        impl = self._impl
        speech_probs = []
        torch_frame = torch.from_numpy(data.astype(np.float32) / 32768.0).to(self._device)
        for i in range(0, len(torch_frame), _WINDOW_SIZE):
            torch_chunk = torch_frame[i : i + _WINDOW_SIZE]
            speech_probs.append(impl(torch_chunk, _SAMPLERATE).item())

        return speech_probs

    def _warm_up(self) -> None:
        _ = self(np.zeros((_WINDOW_SIZE * 10,), dtype=np.int16))
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
