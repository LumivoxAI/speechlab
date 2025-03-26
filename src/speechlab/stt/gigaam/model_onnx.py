from typing import Callable
from pathlib import Path

import numpy as np
import onnxruntime as rt

_PRED_HIDDEN = 320
_DTYPE = np.float32
_BLANK_IDX = 33
_MAX_LETTERS_PER_FRAME = 3
_VOCAB = [
    " ",
    "а",
    "б",
    "в",
    "г",
    "д",
    "е",
    "ж",
    "з",
    "и",
    "й",
    "к",
    "л",
    "м",
    "н",
    "о",
    "п",
    "р",
    "с",
    "т",
    "у",
    "ф",
    "х",
    "ц",
    "ч",
    "ш",
    "щ",
    "ъ",
    "ы",
    "ь",
    "э",
    "ю",
    "я",
]


class GigaAMModel:
    def __init__(
        self,
        base_dir: Path,
        model_name: str,
        preprocessor: Callable,
    ) -> None:
        self._base_dir = base_dir
        self._model_name = model_name
        self._preprocessor = preprocessor

        self._opts = rt.SessionOptions()
        self._opts.intra_op_num_threads = 16
        self._opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL

        self._encoder = self._load("encoder")
        assert [node.name for node in self._encoder.get_inputs()] == ["audio_signal"]
        assert [node.name for node in self._encoder.get_outputs()] == ["encoded_proj"]

        self._decoder = self._load("decoder")
        assert [node.name for node in self._decoder.get_inputs()] == [
            "x",
            "token_in",
            "h_in",
            "c_in",
        ]
        assert [node.name for node in self._decoder.get_outputs()] == [
            "token_out",
            "h_out",
            "c_out",
        ]

        self._init_hc = np.zeros(shape=(1, _PRED_HIDDEN), dtype=_DTYPE)

    def _load(self, component: str) -> rt.InferenceSession:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        file_path = self._base_dir / f"{self._model_name}_{component}.onnx"
        return rt.InferenceSession(
            file_path,
            providers=providers,
            sess_options=self._opts,
        )

    def stt(self, input_signal: np.array) -> str:
        audio_signal = self._preprocessor(input_signal).astype(_DTYPE)

        encoder_inputs = {"audio_signal": audio_signal}
        encoded_proj = self._encoder.run(["encoded_proj"], encoder_inputs)[0]
        token_ids = []

        decoder = self._decoder
        d_outputs = ["token_out", "h_out", "c_out"]
        d_inputs = {
            "token_in": np.array(_BLANK_IDX, dtype=np.int64),
            "h_in": self._init_hc,
            "c_in": self._init_hc,
        }

        for d_inputs["x"] in encoded_proj:
            for _ in range(_MAX_LETTERS_PER_FRAME):
                token, h, c = decoder.run(d_outputs, d_inputs)

                if token != _BLANK_IDX:
                    d_inputs["token_in"] = token
                    d_inputs["h_in"] = h
                    d_inputs["c_in"] = c
                    token_ids.append(token)
                else:
                    break

        return "".join(_VOCAB[tok] for tok in token_ids)
