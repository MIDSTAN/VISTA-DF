"""Run and resume temporal representation training and validation.

Consumes collated video/audio windows, updates trainable model heads, and
coordinates optimizer state, batch-boundary checkpoints, and run controls.
"""

import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from losses.intra_modal_loss import temporal_prediction_loss
from datasets.temporal_dataset import safe_collate
from training.checkpoint_manager import CheckpointManager, restore_rng_state
from training.training_controller import TrainingController


class OrderedEpochSampler(Sampler[int]):
    """Deterministic shuffled epoch order that can resume at a batch boundary."""

    def __init__(self, size: int, seed: int, epoch: int, start_index: int = 0):
        """Create a repeatable shuffled order, optionally skipping consumed items."""
        generator = torch.Generator().manual_seed(seed + epoch)
        self.indices = torch.randperm(size, generator=generator).tolist()
        self.indices = self.indices[start_index:]

    def __iter__(self):
        """Yield the remaining dataset indices for this epoch."""
        return iter(self.indices)

    def __len__(self):
        """Return the number of indices left in this sampler."""
        return len(self.indices)


class TemporalTrainer:
    def __init__(
        self,
        *,
        model,
        train_dataset,
        val_dataset,
        output_dir: str | Path,
        config: dict,
        dataset_info: dict,
        device: torch.device,
        controller: TrainingController | None = None,
        audio_model=None,
    ):
        """Set up data loading, trainable parameters, optimization, and resume state."""
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.manager = CheckpointManager(output_dir)
        self.config = config
        self.dataset_info = dataset_info
        self.device = device
        self.controller = controller
        self.audio_model = audio_model.to(device).eval() if audio_model is not None else None
        self.batch_size = int(config["batch_size"])
        self.seed = int(config["seed"])
        self.checkpoint_interval = int(config["checkpoint_interval_steps"])
        self.amp_enabled = bool(config["amp"] and device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("No trainable parameters found")
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        steps_per_epoch = math.ceil(len(self.train_dataset) / self.batch_size)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, int(config["epochs"]) * steps_per_epoch)
        )
        self.epoch = 0
        self.next_batch_index = 0
        self.global_step = 0
        self.best_validation_loss = math.inf
        self.history: list[dict] = []
        self.epoch_loss_sum = 0.0
        self.epoch_examples = 0
        self.skipped_samples = 0

    def _state(self) -> dict:
        """Collect all model and progress values needed for exact resumption."""
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
            "next_batch_index": self.next_batch_index,
            "global_step": self.global_step,
            "best_validation_loss": self.best_validation_loss,
            "history": self.history,
            "epoch_loss_sum": self.epoch_loss_sum,
            "epoch_examples": self.epoch_examples,
            "skipped_samples": self.skipped_samples,
            "config": self.config,
            "dataset": self.dataset_info,
            "training_objective": "next VideoMAE frame feature prediction (MSE + cosine distance)",
        }

    def save(self, *, best: bool = False) -> None:
        """Write current training state through the checkpoint manager."""
        self.manager.save(self._state(), best=best)

    def resume(self) -> None:
        """Restore model, optimizer, progress, and random states from latest.pt."""
        state = self.manager.load_latest(map_location="cpu")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler"):
            self.scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler"):
            self.scaler.load_state_dict(state["scaler"])
        self.epoch = int(state["epoch"])
        self.next_batch_index = int(state["next_batch_index"])
        self.global_step = int(state["global_step"])
        self.best_validation_loss = float(state.get("best_validation_loss", math.inf))
        self.history = list(state.get("history", []))
        self.epoch_loss_sum = float(state.get("epoch_loss_sum", 0.0))
        self.epoch_examples = int(state.get("epoch_examples", 0))
        self.skipped_samples = int(state.get("skipped_samples", 0))
        if state.get("rng_state"):
            restore_rng_state(state["rng_state"])
        saved_config = state.get("config", {})
        for key in ("epochs", "batch_size", "seed", "learning_rate", "weight_decay"):
            if key in saved_config and saved_config[key] != self.config[key]:
                raise ValueError(f"Cannot resume with changed {key}: {saved_config[key]} -> {self.config[key]}")
        print(
            f"Restored step={self.global_step}, epoch={self.epoch}, "
            f"next_batch={self.next_batch_index}",
            flush=True,
        )

    def _loader(self, dataset, *, sampler=None, shuffle=False):
        """Build a stable-order loader that filters invalid decoded samples."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
            num_workers=0,  # keeps the decoded sample order exactly recoverable
            pin_memory=self.device.type == "cuda",
            drop_last=False,
            collate_fn=safe_collate,
        )

    def _batch_loss(self, batch):
        """Run optional audio encoding and return loss components for one batch."""
        pixels = batch["pixel_values"].to(self.device, non_blocking=True)
        audio_features = None
        if self.audio_model is not None:
            waveforms = [waveform.numpy() for waveform in batch["waveform"]]
            input_features = self.audio_model.waveform_to_features(waveforms).to(
                self.device, non_blocking=True
            )
            with torch.no_grad():
                audio_features = self.audio_model(input_features)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.amp_enabled,
        ):
            output = self.model(pixels, audio_features=audio_features)
        return temporal_prediction_loss(output["prediction"], output["target"])

    @torch.no_grad()
    def validate(self, max_batches: int | None = None) -> float:
        """Measure example-weighted loss without updating model parameters."""
        self.model.eval()
        loader = self._loader(self.val_dataset)
        total_loss = 0.0
        total_examples = 0
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            if batch["_valid_count"] == 0:
                self.skipped_samples += batch["_skipped"]
                continue
            loss, _ = self._batch_loss(batch)
            count = batch["_valid_count"]
            total_loss += float(loss) * count
            total_examples += count
        if total_examples == 0:
            raise RuntimeError("Validation dataset is empty")
        return total_loss / total_examples

    def _finish_epoch(self, epoch: int) -> None:
        """Validate, record the completed epoch, and save latest/best states."""
        validation_loss = self.validate()
        train_loss = self.epoch_loss_sum / max(1, self.epoch_examples)
        record = {
            "epoch": epoch + 1,
            "global_step": self.global_step,
            "train_temporal_loss": train_loss,
            "validation_temporal_loss": validation_loss,
        }
        self.history.append(record)
        improved = validation_loss < self.best_validation_loss
        if improved:
            self.best_validation_loss = validation_loss
        self.epoch = epoch + 1
        self.next_batch_index = 0
        self.epoch_loss_sum = 0.0
        self.epoch_examples = 0
        self.save(best=improved)
        print(f"epoch summary: {record}", flush=True)

    def fit(
        self,
        epochs: int,
        *,
        max_train_batches: int | None = None,
        max_validation_batches: int | None = None,
    ) -> None:
        """Train through the requested epochs, saving resumable batch progress."""
        while self.epoch < epochs:
            epoch = self.epoch
            batches_per_epoch = math.ceil(len(self.train_dataset) / self.batch_size)
            if self.next_batch_index >= batches_per_epoch:
                # A pause/stop may have happened immediately after the final
                # optimizer update, before validation and epoch bookkeeping.
                self._finish_epoch(epoch)
                continue
            start_sample = self.next_batch_index * self.batch_size
            sampler = OrderedEpochSampler(
                len(self.train_dataset), self.seed, epoch, start_index=start_sample
            )
            loader = self._loader(self.train_dataset, sampler=sampler)
            self.model.train()
            processed_this_call = 0
            for local_batch_index, batch in enumerate(loader):
                batch_index = self.next_batch_index
                self.next_batch_index = batch_index + 1
                processed_this_call += 1
                skipped = batch["_skipped"]
                if skipped:
                    self.skipped_samples += skipped
                if batch["_valid_count"] == 0:
                    if self.controller and not self.controller.batch_boundary(self.save):
                        return
                    if self.next_batch_index % self.checkpoint_interval == 0:
                        self.save()
                    if max_train_batches is not None and processed_this_call >= max_train_batches:
                        break
                    continue
                self.optimizer.zero_grad(set_to_none=True)
                loss, components = self._batch_loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at global step {self.global_step}")
                # GradScaler protects mixed-precision gradients from underflow.
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.global_step += 1
                count = batch["_valid_count"]
                self.epoch_loss_sum += float(loss.detach()) * count
                self.epoch_examples += count

                if self.controller and not self.controller.batch_boundary(self.save):
                    return
                if self.global_step % self.checkpoint_interval == 0:
                    self.save()
                if self.global_step % int(self.config["log_interval_steps"]) == 0:
                    print(
                        f"epoch={epoch + 1} batch={self.next_batch_index}/{batches_per_epoch} "
                        f"step={self.global_step} loss={float(loss.detach()):.5f} "
                        f"mse={float(components['mse']):.5f} "
                        f"cos={float(components['cosine_distance']):.5f}",
                        flush=True,
                    )

                if max_train_batches is not None and processed_this_call >= max_train_batches:
                    break

            if max_train_batches is not None and self.next_batch_index < batches_per_epoch:
                # A deliberately bounded smoke run checks the validation path,
                # but keeps the epoch cursor so a later full run can continue.
                validation_loss = self.validate(max_batches=max_validation_batches)
                self.save()
                print(
                    f"bounded smoke segment saved: train_batches={processed_this_call}, "
                    f"validation_temporal_loss={validation_loss:.6f}, "
                    f"resume_cursor=({self.epoch},{self.next_batch_index})",
                    flush=True,
                )
                return

            # DataLoader with an exact tail slice visits each remaining sample once.
            if self.next_batch_index < batches_per_epoch:
                raise RuntimeError("Epoch ended before the configured batch count")
            self._finish_epoch(epoch)

