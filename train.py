"""Configure and run VISTA-DF temporal representation training.

Builds the datasets and pretrained feature extractors, then delegates the
training and checkpoint lifecycle to ``TemporalTrainer``.
"""

import argparse
import random
from pathlib import Path
import shutil

import numpy as np
import torch

from datasets.temporal_dataset import KineticsWindowDataset
from models.audio_network import WhisperFeatureExtractor
from models.vista_model import VistaTemporalModel
from training.trainer import TemporalTrainer
from training.training_controller import TrainingController


def parse_args():
    """Read paths and training options supplied on the command line."""
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default=str(project_root / "pretrained_models/videomae-base"),
                        help="Local VideoMAE checkpoint directory")
    parser.add_argument("--windows-csv", default=str(project_root / "dataset/K700/metadata/windows.csv"),
                        help="Kinetics windows.csv")
    parser.add_argument("--video-root", default=str(project_root / "dataset/K700/videos"),
                        help="Directory containing class/video.mp4 paths")
    parser.add_argument("--output-dir", default=str(project_root / "runs/vista_df"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1, help="Use 1 for GPUs with 4 GiB VRAM")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=20)
    parser.add_argument("--log-interval-steps", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="Stop after this many batches and save mid-epoch (smoke runs)")
    parser.add_argument("--max-val-batches", type=int, default=None,
                        help="Cap validation batches for bounded smoke runs")
    parser.add_argument("--command-file", default=None, help="Optional file containing RUN, PAUSE, or STOP")
    parser.add_argument("--resume", action="store_true", help="Restore latest.pt in output-dir")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA float16 autocast")
    parser.add_argument("--use-audio", action="store_true", help="Fuse frozen Whisper features from each video window")
    parser.add_argument("--audio-checkpoint-dir", default=str(project_root / "pretrained_models/whisper-small"))
    return parser.parse_args()


def main():
    """Validate the runtime, build the pipeline, and start or resume training."""
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if args.checkpoint_interval_steps < 1 or args.log_interval_steps < 1:
        raise ValueError("checkpoint/log intervals must be positive")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        raise ValueError("max-train-batches must be positive")
    if args.max_val_batches is not None and args.max_val_batches < 1:
        raise ValueError("max-val-batches must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this training entry point. Check the local NVIDIA driver and myenv PyTorch build."
        )

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    disk = shutil.disk_usage(Path(args.output_dir).expanduser().resolve().parent)
    print(
        f"device={properties.name} vram_gib={total_bytes / 1024**3:.2f} "
        f"free_vram_gib={free_bytes / 1024**3:.2f} "
        f"free_disk_gib={disk.free / 1024**3:.2f}",
        flush=True,
    )

    audio_model = WhisperFeatureExtractor(args.audio_checkpoint_dir) if args.use_audio else None
    model = VistaTemporalModel(
        args.checkpoint_dir,
        encoder_trainable=False,
        audio_dim=audio_model.hidden_size if audio_model is not None else None,
    )
    train_data = KineticsWindowDataset(
        args.windows_csv, args.video_root, model.video.image_processor, split="train",
        num_frames=model.video.num_frames,
        include_audio=args.use_audio,
    )
    val_data = KineticsWindowDataset(
        args.windows_csv, args.video_root, model.video.image_processor, split="val",
        num_frames=model.video.num_frames,
        include_audio=args.use_audio,
    )
    print(f"train_windows={len(train_data)} val_windows={len(val_data)}", flush=True)

    config = {
        "checkpoint_dir": str(Path(args.checkpoint_dir)),
        "windows_csv": str(Path(args.windows_csv)),
        "video_root": str(Path(args.video_root)),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": seed,
        "checkpoint_interval_steps": args.checkpoint_interval_steps,
        "log_interval_steps": args.log_interval_steps,
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
        "amp": not args.no_amp,
        "encoder_trainable": False,
        "use_audio": args.use_audio,
        "audio_checkpoint_dir": str(Path(args.audio_checkpoint_dir)) if args.use_audio else None,
    }
    dataset_info = {
        "source": "Kinetics-700 subset",
        "labels": "action class only; no real/fake labels used",
        "window_csv": str(Path(args.windows_csv)),
        "train_windows": len(train_data),
        "validation_windows": len(val_data),
        "holdout_used": False,
    }
    controller = TrainingController(args.command_file) if args.command_file else None
    trainer = TemporalTrainer(
        model=model,
        train_dataset=train_data,
        val_dataset=val_data,
        output_dir=args.output_dir,
        config=config,
        dataset_info=dataset_info,
        device=device,
        controller=controller,
        audio_model=audio_model,
    )
    if args.resume:
        trainer.resume()
    trainer.fit(
        args.epochs,
        max_train_batches=args.max_train_batches,
        max_validation_batches=args.max_val_batches,
    )


if __name__ == "__main__":
    main()

