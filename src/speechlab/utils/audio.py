from wave import Wave_read
from pathlib import Path

import numpy as np

INT16_TO_FLOAT32_SCALE = np.float32(1.0 / 32768.0)

# f32n - float32 normalized, i.e. values are in range [-1, 1]


def generate_warmup_audio_f32n(samplerate: int = 16000, duration_sec: int = 20) -> np.ndarray:
    # Generate a sinusoidal signal (speech imitation)
    t = np.linspace(0, duration_sec, int(samplerate * duration_sec), endpoint=False)
    # Create a signal with several frequencies typical for human voice
    freq1, freq2 = 150, 450
    signal = 0.5 * np.sin(2 * np.pi * freq1 * t) + 0.3 * np.sin(2 * np.pi * freq2 * t)

    return signal / np.max(np.abs(signal))


def read_wav_as_f32n(filepath: Path | str) -> np.ndarray:
    with Wave_read(str(filepath)) as wf:
        if wf.getsampwidth() != 2:
            raise ValueError("Unsupported sample width")
        if wf.getnchannels() != 1:
            raise ValueError("Unsupported number of channels")
        if wf.getframerate() != 16000:
            raise ValueError("Unsupported sample rate")

        return (
            np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)
            * INT16_TO_FLOAT32_SCALE
        )
