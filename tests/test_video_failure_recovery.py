"""Test video decoding, invalid-sample filtering, and resumable batch progress.

Small fixtures isolate the production decoder, dataset/collator, and sampler
without running a full model or training job.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from datasets.temporal_dataset import KineticsWindowDataset, safe_collate
from preprocessing.video_loader import (
    VideoDecodeError,
    _decode_sequential,
    decode_video_frames,
    preprocess_video_frames,
)
from training.trainer import OrderedEpochSampler, TemporalTrainer


class DummyProcessor:
    """Mimic the image processor output shape without loading a checkpoint."""

    size = {"shortest_edge": 224}

    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        return {"pixel_values": torch.zeros(1, len(images), 3, 224, 224)}


class ToyDataset(Dataset):
    """Provide one valid and one optionally invalid training sample."""

    def __init__(self, invalid_index):
        self.invalid_index = invalid_index

    def __len__(self):
        return 2

    def __getitem__(self, index):
        if index == self.invalid_index:
            return {"_invalid": True, "_skip_reason": "test corruption"}
        return {"pixel_values": torch.tensor([[1.0, 2.0], [2.0, 1.0]])}


class ToyModel(nn.Module):
    """Minimal trainable model matching the trainer's prediction contract."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2)

    def forward(self, pixel_values, audio_features=None):
        return {
            "prediction": self.projection(pixel_values[:, 0]),
            "target": pixel_values[:, 1],
        }


class VideoDecodeTests(unittest.TestCase):
    """Cover decoder correctness and training recovery after bad media."""

    def make_video(self, path: Path) -> None:
        """Write a tiny synthetic clip for deterministic decoder tests."""
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24)
        )
        if not writer.isOpened():
            self.skipTest("OpenCV MJPG writer unavailable")
        for value in range(12):
            writer.write(np.full((24, 32, 3), value * 10, dtype=np.uint8))
        writer.release()

    def test_valid_video_has_exact_rgb_frames_and_preprocessed_tensor(self):
        """Verify frame sampling and processor output dimensions and values."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "valid.avi"
            self.make_video(path)
            frames = decode_video_frames(path, start_sec=0, end_sec=1.0, num_frames=4)
            self.assertEqual(len(frames), 4)
            self.assertTrue(all(frame.shape == (24, 32, 3) and frame.dtype == np.uint8 for frame in frames))
            pixels = preprocess_video_frames(frames, DummyProcessor())
            self.assertEqual(tuple(pixels.shape), (4, 3, 224, 224))
            self.assertEqual(pixels.dtype, torch.float32)
            self.assertTrue(torch.isfinite(pixels).all())

    def test_corrupt_and_missing_videos_raise_cleanly(self):
        """Ensure invalid files produce clear decoder exceptions."""
        with tempfile.TemporaryDirectory() as directory:
            corrupt = Path(directory) / "corrupt.mp4"
            corrupt.write_bytes(b"not a video")
            with self.assertRaises(VideoDecodeError):
                decode_video_frames(corrupt, num_frames=4)
            with self.assertRaises(FileNotFoundError):
                decode_video_frames(Path(directory) / "missing.mp4", num_frames=4)

    def test_random_seek_failure_retries_sequentially(self):
        """Verify that a seek failure falls back to sequential decoding."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "valid.avi"
            self.make_video(path)
            expected = _decode_sequential(path, np.array([0, 3, 7, 10]))
            with patch(
                "preprocessing.video_loader._decode_random",
                side_effect=VideoDecodeError("forced seek failure"),
            ), patch(
                "preprocessing.video_loader._decode_sequential",
                return_value=expected,
            ) as retry:
                frames = decode_video_frames(path, start_sec=0, end_sec=1.1, num_frames=4)
            retry.assert_called_once()
            self.assertEqual(len(frames), 4)
            self.assertTrue(all(np.array_equal(left, right) for left, right in zip(frames, expected)))

    def test_dataset_marks_corrupt_sample_and_collator_filters_it(self):
        """Check a corrupt dataset item is marked and excluded from its batch."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "videos").mkdir()
            csv_path = root / "windows.csv"
            csv_path.write_text(
                "video_id,split,start_sec,end_sec,path\n1,train,0,1,missing.mp4\n",
                encoding="utf-8",
            )
            dataset = KineticsWindowDataset(
                csv_path, root / "videos", DummyProcessor(), split="train", num_frames=4
            )
            item = dataset[0]
            self.assertTrue(item["_invalid"])
            self.assertEqual(dataset.skipped_count, 1)
            batch = safe_collate([item])
            self.assertEqual(batch["_valid_count"], 0)
            self.assertEqual(batch["_skipped"], 1)

    def test_resume_advances_past_skipped_source_batch(self):
        """Ensure resuming continues after consumed batches, including skipped ones."""
        with tempfile.TemporaryDirectory() as directory:
            first_index = OrderedEpochSampler(2, seed=7, epoch=0).__iter__().__next__()
            data = ToyDataset(invalid_index=first_index)
            config = {
                "epochs": 1,
                "batch_size": 1,
                "seed": 7,
                "checkpoint_interval_steps": 100,
                "log_interval_steps": 100,
                "amp": False,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
            }
            trainer = TemporalTrainer(
                model=ToyModel(), train_dataset=data, val_dataset=data,
                output_dir=directory, config=config, dataset_info={},
                device=torch.device("cpu"),
            )
            trainer.fit(1, max_train_batches=1, max_validation_batches=2)
            self.assertEqual(trainer.next_batch_index, 1)
            self.assertEqual(trainer.global_step, 0)
            self.assertGreaterEqual(trainer.skipped_samples, 1)

            resumed = TemporalTrainer(
                model=ToyModel(), train_dataset=data, val_dataset=data,
                output_dir=directory, config=config, dataset_info={},
                device=torch.device("cpu"),
            )
            resumed.resume()
            self.assertEqual(resumed.next_batch_index, 1)
            resumed.fit(1, max_train_batches=1, max_validation_batches=2)
            self.assertEqual(resumed.global_step, 1)
            self.assertGreaterEqual(resumed.skipped_samples, trainer.skipped_samples)


if __name__ == "__main__":
    unittest.main()
