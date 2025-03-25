from torch import Tensor, nn, log
from torchaudio.transforms import MelSpectrogram


class SpecScaler(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return log(x.clamp_(1e-9, 1e9))


class FeatureExtractor(nn.Module):
    def __init__(self, sample_rate: int, features: int) -> None:
        super().__init__()
        self.hop_length = sample_rate // 100
        self.featurizer = nn.Sequential(
            MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=sample_rate // 40,
                win_length=sample_rate // 40,
                hop_length=self.hop_length,
                n_mels=features,
            ),
            SpecScaler(),
        )

    def forward(self, input_signal: Tensor) -> Tensor:
        return self.featurizer(input_signal)
