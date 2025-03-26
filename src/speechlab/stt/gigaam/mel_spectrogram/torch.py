import numpy as np
from torch import Tensor, nn, from_numpy
from torchaudio.transforms import MelSpectrogram as melspectrogram


class MelSpectrogram(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        features: int = 64
        samplerate: int = 16000

        self._impl = melspectrogram(
            sample_rate=samplerate,
            n_fft=samplerate // 40,
            win_length=samplerate // 40,
            hop_length=samplerate // 100,
            n_mels=features,
            # defaults
            # f_min = 0.0,
            # f_max = None,
            # pad: int = 0,
            # window_fn = torch.hann_window,
            # power = 2.0,
            # normalized = False,
            # wkwargs = None,
            # center = True,
            # pad_mode = "reflect",
            # onesided = None,
            # norm = None,
            # mel_scale = "htk",
        )

    def forward(self, data: Tensor) -> Tensor:
        return self._impl(data)


class MelSpectrogramNumpy:
    def __init__(self) -> None:
        self._impl = MelSpectrogram()

    def __call__(self, data: np.array) -> np.array:
        return self._impl(from_numpy(data)).numpy()
