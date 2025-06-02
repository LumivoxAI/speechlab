from __future__ import annotations

import gc
import os
import sys
import tempfile
from typing import Any, Self, Generator
from pathlib import Path
from contextlib import contextmanager

import numpy as np

from .model import SAMPLE_RATE, WINDOW_SIZE, SileroModel, validate_chunked_pcm16
from .config import DeviceOption

_STATE_SHAPE = (2, 1, 128)
_CONTEXT_SIZE = 64
_SILERO_GPU_MEMCPY_WARNING = "Memcpy nodes are added to the graph"


@contextmanager
def _filter_silero_gpu_warning(enabled: bool) -> Generator[None, Any, None]:
    if not enabled:
        yield
        return

    # ORT writes this warning from native code to stderr during session creation.
    # Capture only that window and replay every other line unchanged.
    sys.stderr.flush()
    original_stderr_fd = os.dup(2)

    with tempfile.TemporaryFile(mode="w+b") as captured_stderr:
        try:
            os.dup2(captured_stderr.fileno(), 2)
            yield
        finally:
            sys.stderr.flush()
            os.dup2(original_stderr_fd, 2)
            os.close(original_stderr_fd)

            captured_stderr.seek(0)
            captured_output = captured_stderr.read().decode("utf-8", errors="replace")
            for line in captured_output.splitlines(keepends=True):
                if "transformer_memcpy.cc" in line and _SILERO_GPU_MEMCPY_WARNING in line:
                    continue
                sys.stderr.write(line)

            sys.stderr.flush()


class SileroOnnxModel(SileroModel):
    def __init__(self, session: object, providers: list[str]) -> None:
        super().__init__()

        self._session = session
        self._providers = providers
        self.reset()

    @property
    def active_providers(self) -> tuple[str, ...]:
        return tuple(self._providers)

    @classmethod
    def load(cls, model_path: str | Path, device: DeviceOption) -> Self:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for Silero ONNX VAD. Install the project's ONNX extra first."
            ) from exc

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        available_providers = ort.get_available_providers()
        if device == DeviceOption.ONNX_GPU:
            if "CUDAExecutionProvider" not in available_providers:
                raise RuntimeError(
                    "CUDAExecutionProvider is not available for Silero ONNX VAD. "
                    f"Available providers: {available_providers}"
                )
            providers_to_try = [["CUDAExecutionProvider", "CPUExecutionProvider"]]
        elif device == DeviceOption.ONNX_AUTO and "CUDAExecutionProvider" in available_providers:
            providers_to_try = [
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
                ["CPUExecutionProvider"],
            ]
        else:
            providers_to_try = [["CPUExecutionProvider"]]

        if not any(provider in available_providers for provider in providers_to_try[0]):
            raise RuntimeError(
                "No supported ONNX Runtime providers are available. "
                f"Requested {device.value}, available providers: {available_providers}"
            )

        session = None
        last_error = None

        for providers in providers_to_try:
            try:
                with _filter_silero_gpu_warning(enabled=device == DeviceOption.ONNX_GPU):
                    session = ort.InferenceSession(
                        str(model_path), providers=providers, sess_options=opts
                    )
                break
            except Exception as exc:
                last_error = exc

        if session is None:
            requested_provider = (
                device.onnx_provider_name if device != DeviceOption.ONNX_AUTO else "auto"
            )
            raise RuntimeError(
                "Failed to initialize Silero ONNX VAD session. "
                f"Requested provider: {requested_provider}. Available providers: {available_providers}. "
                f"Tried provider chains: {providers_to_try}"
            ) from last_error

        model = cls(session, list(session.get_providers()))
        model._warm_up()
        return model

    def _run_chunk(self, chunk: np.ndarray) -> float:
        if self._session is None:
            raise RuntimeError("Silero ONNX VAD model has been cleared")

        x = chunk.astype(np.float32, copy=False).reshape(1, WINDOW_SIZE) / 32768.0
        x = np.concatenate([self._context, x], axis=1)

        ort_inputs = {
            "input": x,
            "state": self._state,
            "sr": self._sr,
        }
        out, state = self._session.run(None, ort_inputs)
        self._state = state
        self._context = x[:, -_CONTEXT_SIZE:]

        return float(out.reshape(-1)[0])

    def __call__(self, data: np.ndarray) -> list[float]:
        validate_chunked_pcm16(data)
        return [
            self._run_chunk(data[i : i + WINDOW_SIZE]) for i in range(0, len(data), WINDOW_SIZE)
        ]

    def _warm_up(self) -> None:
        _ = self(np.zeros((WINDOW_SIZE,), dtype=np.int16))
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT_SIZE), dtype=np.float32)
        self._sr = np.array([SAMPLE_RATE], dtype=np.int64)

    def clear(self) -> None:
        self._state = np.zeros((0,), dtype=np.float32)
        self._context = np.zeros((0,), dtype=np.float32)
        self._sr = np.zeros((0,), dtype=np.int64)
        self._providers = []

        if self._session is not None:
            self._session = None

        gc.collect()
