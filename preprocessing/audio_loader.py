"""Decode optional video-window audio into Whisper-ready mono waveforms.

The dataset calls this FFmpeg helper when audio fusion is enabled; output is a
finite 16 kHz floating-point tensor for the matching temporal window.
"""

import subprocess
from pathlib import Path

import numpy as np
import torch


def decode_audio_window(
    audio_path: str | Path,
    *,
    start_sec: float = 0.0,
    duration_sec: float,
    sampling_rate: int = 16_000,
) -> torch.Tensor:
    """Return the requested window as a mono 16 kHz waveform tensor."""
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio source video does not exist: {path}")
    if start_sec < 0 or duration_sec <= 0 or sampling_rate != 16_000:
        raise ValueError("Require nonnegative start, positive duration, and 16 kHz sampling")
    command = [
        "ffmpeg", "-v", "error", "-ss", str(start_sec), "-i", str(path),
        "-t", str(duration_sec), "-vn", "-ac", "1", "-ar", str(sampling_rate),
        "-f", "f32le", "pipe:1",
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    waveform = np.frombuffer(result.stdout, dtype="<f4").copy()
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise RuntimeError(f"No valid audio samples decoded from {path}")
    return torch.from_numpy(waveform)