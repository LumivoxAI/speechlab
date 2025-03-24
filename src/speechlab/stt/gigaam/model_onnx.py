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
    def __init__(self, base_dir: Path, model_name: str, preprocessor) -> None:
        self._base_dir = base_dir
        self._model_name = model_name
        self._preprocessor = preprocessor

        self._opts = rt.SessionOptions()
        self._opts.intra_op_num_threads = 16
        self._opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL

        self._encoder = self._load("encoder")
        assert [node.name for node in self._encoder.get_inputs()] == ["audio_signal", "length"]
        assert [node.name for node in self._encoder.get_outputs()] == ["encoded", "encoded_len"]

        self._decoder = self._load("decoder")
        assert [node.name for node in self._decoder.get_inputs()] == ["x", "line", "h_in", "c_in"]
        assert [node.name for node in self._decoder.get_outputs()] == ["dec", "h_out", "c_out"]

    def _load(self, component: str) -> rt.InferenceSession:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        file_path = self._base_dir / f"{self._model_name}_{component}.onnx"
        return rt.InferenceSession(
            file_path,
            providers=providers,
            sess_options=self._opts,
        )

    def stt(self, input_signal: np.array) -> str:
        pinput_signal = self._preprocessor(input_signal)

        encoder_inputs = {
            "audio_signal": pinput_signal.astype(_DTYPE),
            "length": [pinput_signal.shape[-1]],
        }
        encoder_outputs = ["encoded"]
        enc_proj = self._encoder.run(encoder_outputs, encoder_inputs)[0]

        token_ids = []
        prev_token = np.int64(_BLANK_IDX)

        pred_states = [
            np.zeros(shape=(1, _PRED_HIDDEN), dtype=_DTYPE),
            np.zeros(shape=(1, _PRED_HIDDEN), dtype=_DTYPE),
        ]
        decoder = self._decoder
        decoder_outputs = ["dec", "h_out", "c_out"]

        for j in range(enc_proj.shape[0]):
            emitted_letters = 0
            while emitted_letters < _MAX_LETTERS_PER_FRAME:
                decoder_inputs = {
                    "x": np.array(prev_token, dtype=np.int64),
                    "line": enc_proj[j],
                    "h_in": pred_states[0],
                    "c_in": pred_states[1],
                }

                token, h_out, c_out = decoder.run(decoder_outputs, decoder_inputs)

                if token != _BLANK_IDX:
                    prev_token = token
                    pred_states[0] = h_out
                    pred_states[1] = c_out
                    token_ids.append(token)
                    emitted_letters += 1
                else:
                    break

        return "".join(_VOCAB[tok] for tok in token_ids)
