"""Load fixed temporal windows from Kinetics metadata for model training.

Connects ``windows.csv`` records to video/audio decoders and marks corrupt
samples so ``safe_collate`` and the trainer can skip them cleanly.
"""

import logging
import subprocess
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, default_collate

from preprocessing.audio_loader import decode_audio_window
from preprocessing.video_loader import VideoDecodeError, decode_video_frames, preprocess_video_frames


logger = logging.getLogger(__name__)


def safe_collate(samples: list[dict]) -> dict:
    """Batch valid samples and report how many invalid items were skipped."""
    valid = [sample for sample in samples if not sample.get("_invalid", False)]
    skipped = len(samples) - len(valid)
    if not valid:
        return {"_skipped": skipped, "_valid_count": 0}
    waveforms = [sample.pop("waveform") for sample in valid] if "waveform" in valid[0] else None
    batch = default_collate(valid)
    if waveforms is not None:
        batch["waveform"] = waveforms
    batch["_skipped"] = skipped
    batch["_valid_count"] = len(valid)
    return batch


class KineticsWindowDataset(Dataset):
    """Reads ``windows.csv`` and yields sampled VideoMAE clips.

    ``video_root`` must be the directory that contains the class-named video
    directories referenced by the CSV's relative ``path`` column.
    """

    REQUIRED_COLUMNS = {"video_id", "split", "start_sec", "end_sec", "path"}

    def __init__(
        self,
        windows_csv: str | Path,
        video_root: str | Path,
        image_processor,
        *,
        split: str,
        num_frames: int = 16,
        include_audio: bool = False,
    ):
        """Load one CSV split and configure its frame and optional audio inputs."""
        self.csv_path = Path(windows_csv)
        self.video_root = Path(video_root)
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Windows CSV not found: {self.csv_path}")
        if not self.video_root.is_dir():
            raise FileNotFoundError(f"Video root not found: {self.video_root}")
        self.image_processor = image_processor
        self.num_frames = int(num_frames)
        self.include_audio = include_audio
        self.skipped_count = 0
        self._bad_paths: dict[Path, str] = {}
        if self.num_frames < 1:
            raise ValueError("num_frames must be positive")

        frame = pd.read_csv(self.csv_path)
        missing = self.REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{self.csv_path} is missing columns: {sorted(missing)}")
        self.rows = frame.loc[frame["split"].astype(str) == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"No windows for split={split!r} in {self.csv_path}")
        self.split = split

    def __len__(self) -> int:
        """Return the number of windows in the selected split."""
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        """Decode and preprocess one window, returning IDs with its tensors."""
        row = self.rows.iloc[index]
        relative_path = Path(str(row["path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe/non-relative video path in CSV: {relative_path}")
        video_path = self.video_root / relative_path
        if video_path in self._bad_paths:
            return self._invalid_sample(video_path, self._bad_paths[video_path])
        try:
            frames = decode_video_frames(
                video_path,
                start_sec=float(row["start_sec"]),
                end_sec=float(row["end_sec"]),
                num_frames=self.num_frames,
            )
            pixels = preprocess_video_frames(frames, self.image_processor)
            if tuple(pixels.shape) != (self.num_frames, 3, 224, 224):
                raise VideoDecodeError(f"invalid preprocessed shape {tuple(pixels.shape)}")
            if pixels.dtype != torch.float32 or not torch.isfinite(pixels).all():
                raise VideoDecodeError("preprocessing returned invalid tensor values")
            sample = {
                "pixel_values": pixels,
                "video_id": str(row["video_id"]),
                "window_id": int(row["window_id"]) if "window_id" in row else int(index),
                "split": self.split,
            }
            if self.include_audio:
                sample["waveform"] = decode_audio_window(
                    video_path,
                    start_sec=float(row["start_sec"]),
                    duration_sec=float(row["end_sec"]) - float(row["start_sec"]),
                )
            return sample
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            reason = str(exc)
            return self._invalid_sample(video_path, reason)

    def _invalid_sample(self, video_path: Path, reason: str) -> dict:
        """Cache a failed video path and return a marker for the collator."""
        is_new = video_path not in self._bad_paths
        self._bad_paths.setdefault(video_path, reason)
        if is_new:
            self.skipped_count += 1
        if is_new and (self.skipped_count <= 10 or self.skipped_count % 100 == 0):
            logger.warning(
                "Skipping invalid video sample (%d skipped): %s: %s",
                self.skipped_count,
                video_path,
                reason,
            )
        return {"_invalid": True, "_skip_reason": reason}

