import json
import subprocess
import tempfile
import unittest
from collections import defaultdict, Counter
from pathlib import Path
from unittest.mock import patch

import torch

from classifier import (
    _assign_group_splits,
    _append_error,
    _load_manifest,
    _load_or_extract_feature,
    _metrics,
    _sample_frame_pair,
)
from models.binary_classifier import BinaryClassificationHead


class BinaryClassifierTests(unittest.TestCase):
    def test_head_returns_two_finite_probabilities_and_trains_independently(self):
        head = BinaryClassificationHead()
        features = torch.randn(2, 768)
        logits = head(features)
        probabilities = head.probabilities(logits)
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(2), atol=1e-6))
        logits.sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in head.parameters()))
        self.assertFalse(features.requires_grad)

    def test_group_split_keeps_pairs_together_and_hits_target_fractions(self):
        records = []
        groups = defaultdict(list)
        for pair_id in range(100):
            for label in (0, 1):
                index = len(records)
                records.append({"label": label})
                groups[f"pair_{pair_id}"] .append(index)
        assignments = _assign_group_splits(
            records,
            dict(groups),
            {"train": 0.70, "validation": 0.15, "test": 0.15},
            seed=42,
        )
        for indices in groups.values():
            self.assertEqual(len({assignments[index] for index in indices}), 1)
        counts = {
            split: Counter(records[index]["label"] for index, value in assignments.items() if value == split)
            for split in ("train", "validation", "test")
        }
        self.assertEqual(counts["train"], Counter({0: 70, 1: 70}))
        self.assertEqual(counts["validation"], Counter({0: 15, 1: 15}))
        self.assertEqual(counts["test"], Counter({0: 15, 1: 15}))

    def test_manifest_rejects_label_path_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            root.mkdir()
            manifest = Path(directory) / "splits.json"
            manifest.write_text(json.dumps({
                "dataset_root": str(root.resolve()),
                "records": [{"path": "deepfake/42.mp4", "label": 0, "label_name": "real"}],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Label/path mismatch"):
                _load_manifest(manifest, root)

    def test_manifest_rejects_groups_crossing_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            root.mkdir()
            manifest = Path(directory) / "splits.json"
            manifest.write_text(json.dumps({
                "dataset_root": str(root.resolve()),
                "records": [
                    {"path": "real/1.mp4", "label": 0, "label_name": "real",
                     "split": "train", "group_id": "shared"},
                    {"path": "deepfake/1.mp4", "label": 1, "label_name": "deepfake",
                     "split": "test", "group_id": "shared"},
                ],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "crosses splits"):
                _load_manifest(manifest, root)

    def test_metrics_include_confusion_and_roc_auc(self):
        result = _metrics([0, 0, 1, 1], [0.1, 0.8, 0.6, 0.9])
        self.assertEqual(result["confusion_matrix_real_deepfake"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["precision_deepfake"], 2 / 3)
        self.assertAlmostEqual(result["recall_deepfake"], 1.0)
        self.assertAlmostEqual(result["f1_deepfake"], 0.8)
        self.assertAlmostEqual(result["roc_auc"], 0.75)

    def test_feature_cache_avoids_reextracting_valid_features(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "feature.pt"
            feature = torch.randn(768)
            calls = []

            def extract():
                calls.append(True)
                return feature

            first = _load_or_extract_feature(cache_path, extract)
            second = _load_or_extract_feature(cache_path, extract)
            self.assertEqual(len(calls), 1)
            self.assertTrue(torch.equal(first, second))

    @patch("classifier.subprocess.run")
    def test_sample_frame_pair_parses_one_or_two_frames(self, run):
        run.return_value = type("Result", (), {
            "returncode": 0,
            "stdout": bytes(2048),
        })()
        first, second = _sample_frame_pair(Path("video.mp4"))
        self.assertEqual(first.shape, (32, 32))
        self.assertEqual(second.shape, (32, 32))
        self.assertIn("eq(n,0)+eq(n,25)", run.call_args.args[0][7])

        run.return_value.stdout = bytes(1024)
        first, second = _sample_frame_pair(Path("short.mp4"))
        self.assertEqual(first.shape, (32, 32))
        self.assertIsNone(second)

    def test_sample_error_log_preserves_ffmpeg_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            error = subprocess.CalledProcessError(
                1, ["ffmpeg"], stderr=b"Output file does not contain any stream"
            )
            _append_error(Path(directory), {"path": "deepfake/no_audio.mp4", "label": 1}, error)
            entry = json.loads((Path(directory) / "errors.jsonl").read_text(encoding="utf-8"))
            self.assertIn("does not contain any stream", entry["stderr"])


if __name__ == "__main__":
    unittest.main()