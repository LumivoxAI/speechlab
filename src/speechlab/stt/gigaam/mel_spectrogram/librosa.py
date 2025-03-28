import warnings

import numpy as np
import librosa
from scipy.sparse import csr_matrix
from scipy.signal.windows import hann


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
        return librosa.feature.melspectrogram(
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


class MelSpectrogramFast:
    def __init__(self) -> None:
        features: int = 64
        samplerate: int = 16000

        self._n_fft = samplerate // 40
        self._hop_length = samplerate // 100

        data_dim = 1
        win_length = samplerate // 40
        fft_window = hann(win_length, False)
        fft_window = librosa.util.pad_center(fft_window, size=self._n_fft)
        fft_window = librosa.util.expand_to(fft_window, ndim=1 + data_dim, axes=-2)
        self._fft_window = fft_window

        mel_basis = librosa.filters.mel(
            sr=float(samplerate),
            n_fft=self._n_fft,
            n_mels=features,
            fmin=0.0,
            fmax=None,
            htk=True,
            norm=None,
            dtype=np.float32,
        )
        self._mel_basis_sparse = csr_matrix(mel_basis)

    def stft(
        self,
        y: np.ndarray,
        *,
        n_fft: int,
        hop_length: int,
        pad_mode: str = "reflect",
    ) -> np.ndarray:
        librosa.util.valid_audio(y)

        fft_window = self._fft_window

        # Pad the time series so that frames are centered
        if n_fft > y.shape[-1]:
            warnings.warn(f"n_fft={n_fft} is too large for input signal of length={y.shape[-1]}")

        # Set up the padding array to be empty, and we'll fix the target dimension later
        padding = [(0, 0)]

        # How many frames depend on left padding?
        start_k = int(np.ceil(n_fft // 2 / hop_length))

        # What's the first frame that depends on extra right-padding?
        tail_k = (y.shape[-1] + n_fft // 2 - n_fft) // hop_length + 1

        if tail_k <= start_k:
            # If tail and head overlap, then just copy-pad the signal and carry on
            start = 0
            extra = 0
            padding[-1] = (n_fft // 2, n_fft // 2)
            y = np.pad(y, padding, mode=pad_mode)
        else:
            # If tail and head do not overlap, then we can implement padding on each part separately
            # and avoid a full copy-pad

            # "Middle" of the signal starts here, and does not depend on head padding
            start = start_k * hop_length - n_fft // 2
            padding[-1] = (n_fft // 2, 0)

            # +1 here is to ensure enough samples to fill the window
            # fixes bug #1567
            y_pre = np.pad(
                y[..., : (start_k - 1) * hop_length - n_fft // 2 + n_fft + 1],
                padding,
                mode=pad_mode,
            )
            y_frames_pre = librosa.util.frame(y_pre, frame_length=n_fft, hop_length=hop_length)
            # Trim this down to the exact number of frames we should have
            y_frames_pre = y_frames_pre[..., :start_k]

            # How many extra frames do we have from the head?
            extra = y_frames_pre.shape[-1]

            # Determine if we have any frames that will fit inside the tail pad
            if tail_k * hop_length - n_fft // 2 + n_fft <= y.shape[-1] + n_fft // 2:
                padding[-1] = (0, n_fft // 2)
                y_post = np.pad(
                    y[..., (tail_k) * hop_length - n_fft // 2 :], padding, mode=pad_mode
                )
                y_frames_post = librosa.util.frame(
                    y_post, frame_length=n_fft, hop_length=hop_length
                )
                # How many extra frames do we have from the tail?
                extra += y_frames_post.shape[-1]
            else:
                # In this event, the first frame that touches tail padding would run off
                # the end of the padded array
                # We'll circumvent this by allocating an empty frame buffer for the tail
                # this keeps the subsequent logic simple
                post_shape = list(y_frames_pre.shape)
                post_shape[-1] = 0
                y_frames_post = np.empty_like(y_frames_pre, shape=post_shape)

        fft = librosa.core.get_fftlib()

        dtype = librosa.util.dtype_r2c(y.dtype)

        # Window the time series.
        y_frames = librosa.util.frame(y[..., start:], frame_length=n_fft, hop_length=hop_length)

        # Pre-allocate the STFT matrix
        shape = list(y_frames.shape)

        # This is our frequency dimension
        shape[-2] = 1 + n_fft // 2

        # If there's padding, there will be extra head and tail frames
        shape[-1] += extra

        stft_matrix = np.zeros(shape, dtype=dtype, order="F")

        # Fill in the warm-up
        if extra > 0:
            off_start = y_frames_pre.shape[-1]
            stft_matrix[..., :off_start] = fft.rfft(fft_window * y_frames_pre, axis=-2)

            off_end = y_frames_post.shape[-1]
            if off_end > 0:
                stft_matrix[..., -off_end:] = fft.rfft(fft_window * y_frames_post, axis=-2)
        else:
            off_start = 0

        n_columns = int(
            librosa.util.MAX_MEM_BLOCK // (np.prod(y_frames.shape[:-1]) * y_frames.itemsize)
        )
        n_columns = max(n_columns, 1)

        for bl_s in range(0, y_frames.shape[-1], n_columns):
            bl_t = min(bl_s + n_columns, y_frames.shape[-1])

            stft_matrix[..., bl_s + off_start : bl_t + off_start] = fft.rfft(
                fft_window * y_frames[..., bl_s:bl_t], axis=-2
            )

        return stft_matrix

    def __call__(self, data: np.array) -> np.array:
        S = np.abs(
            self.stft(
                data,
                n_fft=self._n_fft,
                hop_length=self._hop_length,
                pad_mode="reflect",
            )
        )

        return self._mel_basis_sparse.dot(S * S)
