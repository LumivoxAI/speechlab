import torch
import torchaudio
from torch import Tensor, nn


class SpecScaler(nn.Module):
    """
    Module that applies logarithmic scaling to spectrogram values.
    This module clamps the input values within a certain range and then applies a natural logarithm.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.log(x.clamp_(1e-9, 1e9))


class FeatureExtractor(nn.Module):
    """
    Module for extracting Log-mel spectrogram features from raw audio signals.
    This module uses Torchaudio's MelSpectrogram transform to extract features
    and applies logarithmic scaling.
    """

    def __init__(self, sample_rate: int, features: int) -> None:
        super().__init__()
        self.hop_length = sample_rate // 100
        self.featurizer = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=sample_rate // 40,
                win_length=sample_rate // 40,
                hop_length=self.hop_length,
                n_mels=features,
            ),
            SpecScaler(),
        )

    def out_len(self, input_lengths: Tensor) -> Tensor:
        """
        Calculates the output length after the feature extraction process.
        """
        return input_lengths.div(self.hop_length, rounding_mode="floor").add(1).long()

    def forward(self, input_signal: Tensor, length: Tensor) -> tuple[Tensor, Tensor]:
        """
        Extract Log-mel spectrogram features from the input audio signal.
        """
        return self.featurizer(input_signal), self.out_len(length)
