"""Save and restore training checkpoints without exposing partial files.

The manager stores model/optimizer progress and random-number generator states
so a resumed run can continue from its saved batch boundary.
"""

import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def capture_rng_state() -> dict:
    """Capture Python, NumPy, and Torch random states, including CUDA when used."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    """Restore random generators from a previously captured state."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_torch_save(payload: dict, destination: Path) -> None:
    """Write a temporary checkpoint, then replace the destination atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary_name)
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.remove(temporary_name)


def _atomic_json_save(payload: dict, destination: Path) -> None:
    """Write human-readable history JSON through a temporary file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.remove(temporary_name)


class CheckpointManager:
    """Manage latest, best, and JSON history artifacts in one output directory."""

    def __init__(self, output_dir: str | Path):
        """Create the output directory and define checkpoint file paths."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.output_dir / "latest.pt"
        self.best_path = self.output_dir / "best.pt"
        self.history_path = self.output_dir / "history.json"

    def save(self, state: dict, *, best: bool = False) -> None:
        """Write latest first atomically; best is a separate retained artifact."""
        payload = dict(state)
        payload["rng_state"] = capture_rng_state()
        _atomic_torch_save(payload, self.latest_path)
        if best:
            _atomic_torch_save(payload, self.best_path)
        _atomic_json_save(
            {
                "global_step": payload.get("global_step"),
                "epoch": payload.get("epoch"),
                "next_batch_index": payload.get("next_batch_index"),
                "history": payload.get("history", []),
                "best_validation_loss": payload.get("best_validation_loss"),
                "config": payload.get("config", {}),
                "dataset": payload.get("dataset", {}),
            },
            self.history_path,
        )

    def load_latest(self, *, map_location="cpu") -> dict:
        """Load the latest complete checkpoint onto the requested device."""
        if not self.latest_path.is_file():
            raise FileNotFoundError(f"No latest checkpoint found at {self.latest_path}")
        try:
            # These are user-created training checkpoints and include RNG tuples.
            return torch.load(self.latest_path, map_location=map_location, weights_only=False)
        except Exception as exc:
            raise RuntimeError(
                f"Could not restore {self.latest_path}; the file may be corrupt. "
                "The previous checkpoint is retained only if separately copied."
            ) from exc

