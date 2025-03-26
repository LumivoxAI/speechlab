from torch import Tensor, nn
from torchaudio.transforms import MelSpectrogram


class TorchMelSpectrogram(nn.Module):
    def __init__(self, samplerate: int = 16000, features: int = 64) -> None:
        super().__init__()
        self.featurizer = MelSpectrogram(
            sample_rate=samplerate,
            n_fft=samplerate // 40,
            win_length=samplerate // 40,
            hop_length=samplerate // 100,
            n_mels=features,
        )

    def forward(self, input_signal: Tensor) -> Tensor:
        return self.featurizer(input_signal)
