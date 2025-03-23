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
        self._decoder = self._load("decoder")
        self._joint = self._load("joint")

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
        enc_inputs = {
            node.name: data
            for (node, data) in zip(
                self._encoder.get_inputs(),
                [pinput_signal.astype(_DTYPE), [pinput_signal.shape[-1]]],
            )
        }
        output_names = [node.name for node in self._encoder.get_outputs()]
        enc_features = self._encoder.run(output_names, enc_inputs)[0]

        token_ids = []
        prev_token = _BLANK_IDX

        pred_states = [
            np.zeros(shape=(1, 1, _PRED_HIDDEN), dtype=_DTYPE),
            np.zeros(shape=(1, 1, _PRED_HIDDEN), dtype=_DTYPE),
        ]
        pred_sess = self._decoder
        joint_sess = self._joint
        for j in range(enc_features.shape[-1]):
            emitted_letters = 0
            while emitted_letters < _MAX_LETTERS_PER_FRAME:
                pred_inputs = {
                    node.name: data
                    for (node, data) in zip(pred_sess.get_inputs(), [[[prev_token]]] + pred_states)
                }
                pred_outputs = pred_sess.run(
                    [node.name for node in pred_sess.get_outputs()], pred_inputs
                )

                joint_inputs = {
                    node.name: data
                    for node, data in zip(
                        joint_sess.get_inputs(),
                        [enc_features[:, :, [j]], pred_outputs[0].swapaxes(1, 2)],
                    )
                }
                log_probs = joint_sess.run(
                    [node.name for node in joint_sess.get_outputs()], joint_inputs
                )
                token = int(log_probs[0].argmax(-1)[0][0][0])

                if token != _BLANK_IDX:
                    prev_token = token
                    pred_states = pred_outputs[1:]
                    token_ids.append(token)
                    emitted_letters += 1
                else:
                    break

        return "".join(_VOCAB[tok] for tok in token_ids)
