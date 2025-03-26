import numpy as np
from librosa.feature import melspectrogram


class MelSpectrogram:
    def __init__(self) -> None:
        features: int = 64
        samplerate: int = 16000

        self._samplerate = samplerate
        self._n_fft = samplerate // 40
        self._win_length = samplerate // 40
        self._hop_length = samplerate // 100
        self._n_mels = features

    def __call__(self, data: np.array) -> np.array:
        return melspectrogram(
            y=data,
            sr=self._samplerate,
            n_fft=self._n_fft,
            win_length=self._win_length,
            hop_length=self._hop_length,
            n_mels=self._n_mels,
            # from torch defaults
            fmin=0.0,
            fmax=None,
            window="hann",
            power=2.0,
            center=True,
            pad_mode="reflect",
            norm=None,
            htk=True,
        )
