# VISTA-DF

VISTA-DF contains two separate research workflows: temporal representation learning on Kinetics-700 clips, and a binary real/deepfake baseline using folder-derived labels. The binary classifier freezes the trained temporal model and trains a small classification head; its measured performance and dataset limitations are documented below after the run is verified.

## Verified status

The repository includes a working local temporal-pretraining pipeline and a separate binary-classification pipeline:

- Video windows are read from dataset/K700/metadata/windows.csv.
- Each window is decoded to a fixed number of RGB frames and preprocessed with the local VideoMAE image processor.
- A frozen local VideoMAE checkpoint provides frame features.
- HVCN, temporal transformer, fusion, and prediction heads are trained to predict the next feature vector.
- The trainer saves atomic checkpoints and resumable batch progress.
- Optional Whisper Small audio features can be fused with visual features.
- Corrupted or undecodable videos are skipped instead of crashing training.
- `classifier.py` audits the real/deepfake folders, creates deterministic group-aware splits, and trains/evaluates a binary head over frozen temporal features.
- Classifier features are cached atomically under `runs/vista_classifier/features/` so resumed training does not needlessly decode previously processed videos again.

This is the current verified state of the repository and the latest saved training run.

## Important limitation

The Kinetics temporal-pretraining task remains unrelated to authenticity labels. The classifier's labels come only from the `real/` and `deepfake/` directory names. Local source-name metadata for part of the dataset was lost, and filename/perceptual-hash duplicate groups are heuristic rather than source-verified; this is a research baseline, not a validated general-purpose detector.

---

## 1) Project overview

### Purpose

This repository is structured to train and evaluate a temporal video representation model on Kinetics action videos. The main objective is:

- encode video windows with a local VideoMAE encoder,
- build temporal scene-consistency features across frames,
- predict the next frame feature from the previous context,
- optionally combine the same window with Whisper audio features,
- save and resume training at batch boundaries.

### Current implementation status

Verified in code:

- Training entry point exists in train.py.
- Main model is VistaTemporalModel in models/vista_model.py.
- Dataset is KineticsWindowDataset in datasets/temporal_dataset.py.
- Video decoding and preprocessing are in preprocessing/video_loader.py.
- Audio decoding is in preprocessing/audio_loader.py.
- Trainer logic and resume behavior are in training/trainer.py.
- Checkpoint discipline is in training/checkpoint_manager.py.
- The latest run history is saved at runs/vista_df/history.json.
- Frozen-backbone binary training, grouped splits, inference, and evaluation are implemented in classifier.py.
- The classifier head is in models/binary_classifier.py.

Not verified as a complete product:

- Generalization to external deepfake datasets or deployment use has not been established.
- Labels are folder-derived, and some numeric filename associations and perceptual duplicate links cannot be confirmed from original source metadata.

### Core architecture

The current pipeline is:

1. Build temporal windows from the Kinetics metadata CSV.
2. For each window, decode the video frames and optional audio.
3. Run the local VideoMAE encoder to get frozen per-frame token features.
4. Spatially pool tokens per frame.
5. Pass the pooled frame features through HVCN and a temporal transformer.
6. Fuse video + optional audio features.
7. Predict the next frame feature from the preceding context.
8. Compute a temporal prediction loss based on MSE + cosine distance.
9. Save progress checkpoints and resume from the last completed batch if needed.

---

## 2) File-by-file guide

| File | Purpose | Verified role |
|---|---|---|
| train.py | Main training entry point; parses CLI args and launches the real-data training loop. | Builds dataset, model, trainer, and executes fit(). |
| main.py | Convenience alias for train.py. | Thin wrapper only. |
| requirements.txt | Python dependencies needed for the local project environment. | Includes torch, transformers, opencv-python-headless, pandas, numpy. |
| datasets/temporal_dataset.py | Loads a split from windows.csv and decodes video windows into model-ready tensors. | Produces KineticsWindowDataset. |
| preprocessing/video_loader.py | Video decode, frame sampling, safety checks, retry logic, and VideoMAE preprocessing. | Handles corrupt/invalid videos by skipping with explicit errors. |
| preprocessing/audio_loader.py | FFmpeg-based mono 16 kHz decoding of audio windows. | Used only when --use-audio is enabled. |
| models/video_network.py | Local VideoMAE checkpoint wrapper. | Provides encoder, hidden size, frame count, and image processor. |
| models/hvcn.py | Holistic visual consistency network. | Processes pooled frame features and delta features. |
| models/temporal_transformer.py | Temporal self-attention block over frame order. | Adds temporal context with causal masking. |
| models/fusion.py | Fuses video, HVCN, and optional audio features. | Weighted modality fusion. |
| models/audio_network.py | Wrapper for WhisperModel + feature extractor. | Extracts frozen encoder features from waveforms. |
| models/vista_model.py | Main model definition. | Connects VideoMAE, HVCN, transformer, fusion, predictor. |
| classifier.py | Real/deepfake split, frozen feature extraction, head training, evaluation, and inference CLI. | Uses the folder mapping `real=0`, `deepfake=1`; saves runs separately in `runs/vista_classifier/`. |
| models/binary_classifier.py | Small binary classification head. | LayerNorm, Linear(768,256), GELU, dropout, and two output logits. |
| losses/intra_modal_loss.py | Temporal prediction objective. | MSE + cosine distance. |
| losses/cross_modal_loss.py | Optional paired InfoNCE loss. | Not used in current training stage. |
| training/trainer.py | Training loop, optimizer, scheduler, validation, resume, and checkpointing. | Current training implementation. |
| training/checkpoint_manager.py | Atomic save/load helpers for latest.pt, best.pt, and history.json. | Records RNG state and batch cursor. |
| training/training_controller.py | File-based RUN/PAUSE/STOP control. | Checked on batch boundaries. |
| scripts/create_videos_csv.py | Generate dataset/K700/metadata/videos.csv from the local K700 videos folder. | Metadata inventory for videos. |
| scripts/create_dataset_splits.py | Split videos.csv into train/val/holdout splits. | Creates train.csv / val.csv / holdout.csv. |
| scripts/create_windows_csv.py | Create temporal windows from each split. | Generates dataset/K700/metadata/windows.csv. |
| scripts/download_k700.py | Kinetics downloader helper. | Not used in the current local run. |
| scripts/test_temporal_window.py | Inspect temporal windows generated from the metadata CSV. | Small diagnostic script. |
| scripts/analyze_videos_csv.py | Metadata analysis helper. | Used to inspect video-level metadata. |
| tests/test_video_failure_recovery.py | Regression tests for corrupt videos, missing files, and resume behavior. | Verified in the local environment. |
| runs/vista_df/latest.pt | Latest saved checkpoint with model/optimizer/scheduler/state. | Active resume artifact. |
| runs/vista_df/best.pt | Best validation checkpoint based on validation loss. | Saved separately. |
| runs/vista_df/history.json | Training summary; includes epoch, step, losses, config, dataset info. | Current saved run metadata. |
| dataset/K700/metadata/windows.csv | Final window-level dataset table. | Current dataset used by training. |
| dataset/K700/metadata/train.csv | Video split metadata for training. | Used to build windows. |
| dataset/K700/metadata/val.csv | Validation split metadata. | Used for validation. |
| dataset/K700/metadata/holdout.csv | Holdout split metadata. | Present but not used in this run. |

---

## 3) Execution workflow

### Data generation workflow

1. Build a video inventory from the K700 videos folder:
   - scripts/create_videos_csv.py
2. Split the video metadata into train/val/holdout:
   - scripts/create_dataset_splits.py
3. Generate temporal windows from each split:
   - scripts/create_windows_csv.py
4. Train using the generated windows.csv and the local model checkpoints:
   - train.py

### Runtime flow during training

1. train.py parses arguments.
2. It requires CUDA and loads the local VideoMAE checkpoint directory and optional Whisper checkpoint directory.
3. It builds VistaTemporalModel and the train/validation datasets.
4. It instantiates TemporalTrainer.
5. If --resume is set, trainer.resume() restores latest.pt.
6. trainer.fit() iterates epoch batches using a deterministic shuffled sampler.
7. For each valid window:
   - decode video frames,
   - preprocess with the VideoMAE processor,
   - optionally decode audio and convert it to Whisper features,
   - run the model,
   - compute temporal_prediction_loss,
   - backward pass + optimizer step,
   - save checkpoint every N steps.
8. Invalid windows are skipped by KineticsWindowDataset and counted in batch metadata.
9. At epoch end, validation runs and the epoch summary is written to history.json.

### Data flow

```
train.py
  -> KineticsWindowDataset
      -> decode_video_frames() in preprocessing/video_loader.py
      -> preprocess_video_frames()
      -> model.video(...) in models/video_network.py
  -> VistaTemporalModel
      -> hvcn + temporal_transformer + fusion + predictor
  -> temporal_prediction_loss() in losses/intra_modal_loss.py
  -> TemporalTrainer.fit()
      -> save() / resume() via CheckpointManager
```

---

## 4) Model architecture

### Main model: VistaTemporalModel

Source: models/vista_model.py

The model takes input pixel values shaped approximately `(B, T, C, H, W)` and returns a dictionary with prediction and target tensors.

#### Components

1. VideoFeatureExtractor
   - Source: models/video_network.py
   - Loads a local Hugging Face VideoMAE checkpoint from `pretrained_models/videomae-base`.
   - Uses `VideoMAEModel.from_pretrained(..., local_files_only=True)`.
   - Returns patch tokens of shape `(B, N, D)`.
   - The encoder is frozen by default (`encoder_trainable=False`).

2. HolisticVisualConsistencyNetwork (HVCN)
   - Source: models/hvcn.py
   - Receives pooled frame-level features `(B, T, D)`.
   - Computes content features plus frame-to-frame delta information.
   - Returns per-frame consistency features `(B, T, D)`.

3. TemporalTransformer
   - Source: models/temporal_transformer.py
   - Applies self-attention over ordered frames.
   - Uses a learned positional embedding and optional causal masking.
   - Returns temporal context features `(B, T, D)`.

4. EvidenceFusion
   - Source: models/fusion.py
   - Fuses video temporal features and HVCN features.
   - If audio is enabled, also fuses optional Whisper-derived features.
   - Produces fused temporal features with modality gating.

5. Predictor head
   - A small MLP: `Linear -> GELU -> Linear`.
   - Predicts the next step from the preceding fused context.

#### Forward pass

The current forward logic is approximately:

- `patch_tokens = self.video(pixel_values)`
- check token count divisible by frame count
- pool patch tokens per frame into `visual`: `(B, T, D)`
- `hvcn = self.hvcn(visual)`
- `temporal = self.temporal(visual)`
- optional audio features are expanded over time and fused
- `fused = self.fusion(temporal, hvcn, audio=audio_features)`
- `prediction = self.predictor(fused[:, :-1])`
- `target = visual[:, 1:].detach()`

This means the model predicts the feature at time `t+1` from the context up to `t`.

### Loss

Source: losses/intra_modal_loss.py

The current loss is:

- `mse = F.mse_loss(prediction, target)`
- `cosine = (1 - cosine_similarity(prediction, target, dim=-1)).mean()`
- `total = mse + 0.1 * cosine`

This is a representation-learning objective. It does not optimize a real/fake classification boundary.

---

## 5) Dataset

### Source data

- Kinetics-700 subset under `dataset/K700/videos`
- Metadata CSVs under `dataset/K700/metadata`
- Used in training: `train.csv`, `val.csv`, `holdout.csv`, `windows.csv`

### Split generation

The project generates metadata using the scripts in `scripts/`:

- `create_videos_csv.py` scans the videos directory and writes `videos.csv` with per-video metadata.
- `create_dataset_splits.py` groups by class and allocates 1 validation video + 1 holdout video per class, leaving the rest for training.
- `create_windows_csv.py` creates time windows using a 2-second window and 2-second stride.

Verified generation logic:

- `WINDOW_SIZE = 2.0`
- `STRIDE = 2.0`
- number_of_windows = floor((duration - WINDOW_SIZE) / STRIDE) + 1
- each row is a temporal window with `video_id`, `split`, `window_id`, `start_sec`, `end_sec`, `path`

### Temporal windows

The dataset rows are not one row per original video. They are one row per sampled temporal window. This is why the number of train windows can be much larger than the number of videos.

### Audio and video handling

- Video loader: preprocessing/video_loader.py
- Audio loader: preprocessing/audio_loader.py

The video loader:

- opens the video with OpenCV FFmpeg backend,
- samples evenly spaced frames within the requested time range,
- retries with a sequential decode fallback if random seek fails,
- raises `VideoDecodeError` if the requested frames cannot be decoded,
- preprocesses frames with the checkpoint’s `VideoMAEImageProcessor`.

The audio loader:

- runs FFmpeg with `-ac 1 -ar 16000`
- creates mono 16 kHz waveform data in float32 format
- used only when `--use-audio` is enabled

### Invalid video handling

This is an important, verified behavior in the current code:

- Corrupt, missing, or invalid videos are not silently replaced.
- Instead, they are treated as invalid samples and skipped.
- The dataset keeps a per-video `_bad_paths` map and logs warnings in the first few failures.
- `safe_collate()` removes invalid entries before batching and preserves a skip counter.
- The training loop continues without crashing the run.

This is necessary because the Kinetics set contains some broken or truncated video files; the warning "Could not decode requested frames ..." is consistent with that behavior.

---

## 6) Training pipeline

### Exact command used

The verified local command used in the current run is:

```bash
./myenv/bin/python train.py --epochs 1 --batch-size 1 --use-audio --resume
```

### CLI arguments and meaning

Key arguments from train.py:

- `--checkpoint-dir`: local VideoMAE checkpoint directory
- `--windows-csv`: path to training windows CSV
- `--video-root`: local K700 videos directory
- `--output-dir`: checkpoint directory, default `runs/vista_df`
- `--epochs`: number of epochs to run
- `--batch-size`: samples per batch; default is 1 in this project for 4 GiB VRAM
- `--learning-rate`: AdamW learning rate
- `--weight-decay`: AdamW weight decay
- `--seed`: deterministic seed
- `--checkpoint-interval-steps`: checkpoint save interval in optimizer steps
- `--log-interval-steps`: log interval in optimizer steps
- `--max-train-batches`: bounded smoke-run cap
- `--max-val-batches`: bounded validation cap
- `--command-file`: optional pause/run/stop file control
- `--resume`: load latest.pt and continue from next_batch_index
- `--no-amp`: disable GPU autocast
- `--use-audio`: enable Whisper audio feature fusion
- `--audio-checkpoint-dir`: local Whisper checkpoint directory

### Batch and epoch meaning

- One batch is one mini-batch of sampled windows.
- With `batch_size=1`, each batch contains one temporal window.
- The trainer computes `batches_per_epoch = ceil(len(train_dataset) / batch_size)`.
- A dataset of 16555 windows and batch size 1 gives roughly 16555 batches in one epoch.

### Resume and checkpoint logic

Verified from code and saved checkpoints:

- CheckpointManager saves to `runs/vista_df/latest.pt` atomically.
- It also saves `best.pt` when validation improves.
- It stores model, optimizer, scheduler, scaler, RNG state, epoch, `next_batch_index`, `global_step`, history, and config.
- `TemporalTrainer.resume()` restores those values and validates config compatibility.
- The current run resumes from the last `next_batch_index` rather than restarting the epoch.

### Optimizer and scheduler

From trainer.py:

- optimizer: `torch.optim.AdamW`
- scheduler: `torch.optim.lr_scheduler.CosineAnnealingLR`
- model weights are trained with `self.scaler = torch.amp.GradScaler("cuda")`
- if AMP is enabled, it uses CUDA float16 autocast

### Validation behavior

- Validation runs at epoch completion via `self.validate()`.
- It uses a separate DataLoader with no sampler and no `drop_last`.
- If a validation batch is invalid, it is skipped.
- The validation loss is reported as `validation_temporal_loss`.

### GPU usage

The project expects CUDA. The local training entry point raises if CUDA is unavailable. In the current run, GPU utilization was confirmed to be active; the saved run history includes a working local GPU execution state.

---

## 7) Latest training results

The current saved history file is:

`runs/vista_df/history.json`

It records:

- `epoch`: 1
- `global_step`: 16089
- `train_temporal_loss`: 0.01549790379146377
- `validation_temporal_loss`: 0.0106909559940784
- `best_validation_loss`: 0.0106909559940784

These values mean:

- the model reached the end of the first epoch in this run,
- the training objective decreased to about 0.0155,
- the validation loss decreased to about 0.0107,
- the best validation checkpoint is the current best saved model.

What these metrics do not prove:

- They do not prove the model can detect deepfakes.
- They do not prove general video classification performance.
- They do not prove all video windows are valid or that every Kinetics clip is usable.
- They are only evidence that the temporal prediction objective is being learned on the current real-video dataset.

This is an objective-value result, not a claim about authenticity discrimination.

---

## 8) Saved artifacts

### Important paths

- Checkpoints and history:
  - `/home/midstan/Documents/Python/VISTA-DF/runs/vista_df/latest.pt`
  - `/home/midstan/Documents/Python/VISTA-DF/runs/vista_df/best.pt`
  - `/home/midstan/Documents/Python/VISTA-DF/runs/vista_df/history.json`
- Model checkpoints:
  - `/home/midstan/Documents/Python/VISTA-DF/pretrained_models/videomae-base`
  - `/home/midstan/Documents/Python/VISTA-DF/pretrained_models/whisper-small`
- Dataset metadata and windows:
  - `/home/midstan/Documents/Python/VISTA-DF/dataset/K700/metadata/windows.csv`
  - `/home/midstan/Documents/Python/VISTA-DF/dataset/K700/metadata/train.csv`
  - `/home/midstan/Documents/Python/VISTA-DF/dataset/K700/metadata/val.csv`
  - `/home/midstan/Documents/Python/VISTA-DF/dataset/K700/metadata/holdout.csv`
  - `/home/midstan/Documents/Python/VISTA-DF/dataset/K700/metadata/videos.csv`

### What is needed to resume training

To resume the current run, the project needs:

- the checkpoint at `runs/vista_df/latest.pt`,
- the same config values used when saving, especially batch size, learning rate, seed, and audio flag,
- the local model directories under `pretrained_models/`,
- the same windows.csv and video_root path.

### What is needed to run inference

The repository currently includes no explicit verified inference script or CLI. The model architecture and checkpoint format are present, but no end-to-end inference entry point was verified in the source tree beyond the training loop.

---

## 9) Current status and limitations

### Implemented and verified

- Local VideoMAE and Whisper checkpoint loading from local directories
- Kinetics video sampling into fixed windows
- Video and audio decoding with FFmpeg/OpenCV
- Invalid-file handling and skip logic
- Temporal representation model with HVCN + transformer + fusion
- Optimizer, scheduler, validation, resume, checkpointing
- Current training metrics saved to `history.json`

### Incomplete or not verified

- External deepfake benchmark and independent dataset evaluation: not performed
- Full dataset cleaning and source-level identity verification: incomplete
- Strong assumptions about downstream performance remain unverified

### Known issues and behavior

- Some MP4/VP9 clips fail to decode cleanly due to corrupt or truncated video data.
- The loader intentionally retries and then skips the bad sample; it does not fabricate frames.
- The run logs can show FFmpeg/VP9 warning lines for those files, but they are handled as invalid samples.
- Kinetics action labels are not mapped to real/fake classes; the separate classifier uses only its two folder names.

---

## 10) How to run

### Environment setup

This project expects the local environment under `myenv` and a CUDA-capable GPU.

```bash
cd /home/midstan/Documents/Python/VISTA-DF
./myenv/bin/python -m pip install -r requirements.txt
```

### Smoke test

```bash
./myenv/bin/python train.py \
  --epochs 1 \
  --batch-size 1 \
  --checkpoint-interval-steps 1 \
  --max-train-batches 1 \
  --max-val-batches 1
```

With audio:

```bash
./myenv/bin/python train.py \
  --epochs 1 \
  --batch-size 1 \
  --use-audio \
  --checkpoint-interval-steps 1 \
  --max-train-batches 1 \
  --max-val-batches 1
```

### Full training

```bash
./myenv/bin/python train.py --epochs 1 --batch-size 1 --use-audio
```

### Resume training

```bash
./myenv/bin/python train.py --epochs 1 --batch-size 1 --use-audio --resume
```

### Pause / resume controls

The code supports a command file if `--command-file` is supplied.

Example contents for the command file:

```text
RUN
PAUSE
STOP
```

Behavior:

- `RUN`: continue training
- `PAUSE`: save checkpoint and wait
- `STOP`: save checkpoint and exit cleanly at the next batch boundary

---

## 11) File interaction map

```text
train.py
  ├─ parses CLI args
  ├─ creates VideoFeatureExtractor
  ├─ creates WhisperFeatureExtractor (optional)
  ├─ creates VistaTemporalModel
  ├─ creates KineticsWindowDataset (train + val)
  └─ creates TemporalTrainer

TemporalTrainer
  ├─ uses DataLoader + OrderedEpochSampler
  ├─ calls dataset.__getitem__
  │    ├─ decode_video_frames()
  │    ├─ preprocess_video_frames()
  │    └─ decode_audio_window() [optional]
  ├─ runs VistaTemporalModel.forward()
  ├─ computes temporal_prediction_loss()
  ├─ optimizer step + scheduler step
  └─ saves state via CheckpointManager

CheckpointManager
  ├─ writes latest.pt
  ├─ writes best.pt when validation improves
  └─ writes history.json
```

---

## 12) Important technical decisions and trade-offs

### Why this design

- Local checkpoints are used to avoid network dependence and keep training reproducible in a local environment.
- Batch size 1 is used because this project is tuned for a small CUDA GPU with limited VRAM.
- The video loader is intentionally conservative: invalid decode results are treated as invalid samples instead of substituted with fake frames.
- The strategy uses a deterministic per-epoch sampler plus a `next_batch_index` resume cursor to make checkpoint recovery stable.
- The model is trained to predict features rather than labels, which matches the repository’s representation-learning goal.

### Trade-offs

- Using a frozen VideoMAE encoder reduces training cost but limits task-specific feature learning.
- Temporal windows offer richer local motion context but make the dataset much larger than the raw video count.
- Skipping corrupt videos preserves stability but reduces effective sample count and makes training data less uniform.
- The current code is a research/training pipeline, not a production deepfake detector or evaluation system.

---

## Verified facts vs assumptions

### Verified in code and saved history

- The model architecture is implemented as described in `models/vista_model.py`.
- The dataset uses windows generated from Kinetics metadata.
- The training objective is MSE + cosine distance.
- The latest saved run history is in `runs/vista_df/history.json`.
- The current saved loss values are 0.0154979 train and 0.0106909 validation.
- The saved temporal checkpoint is a representation model; classifier results are tracked separately below.

### Unverified / not currently supported

- Any claim that the model performs deepfake detection.
- Any claim that the current architecture is a general-purpose authenticity detector.
- Any claim that all original Kinetics videos are valid; the code explicitly handles invalid items by skipping them.

---

## Summary

This repository is currently a local Kinetics-based temporal feature learning pipeline with a frozen VideoMAE backbone, HVCN, temporal transformer, optional Whisper audio fusion, and checkpointed resume support. The code is working as a training and resume pipeline. The latest verified checkpoint shows epoch 1 completed with `train_temporal_loss = 0.0155` and `validation_temporal_loss = 0.01069`. It does not, by itself, prove authenticity classification or deepfake detection capability.

## Real vs Deepfake Dataset

The organized videos are in `Real vs deepfake/real/` and `Real vs deepfake/deepfake/`. The parent directory supplies the working labels (`real=0`, `deepfake=1`); the numeric filenames alone are not a source manifest.

### Provenance and pair evidence

- [UniData's Kaggle preview](https://www.kaggle.com/datasets/unidpro/deepfake-videos-dataset) contains a CSV with `id`, `deepfake`, `image`, and `video` columns. It maps IDs 1–5 between `video/<id>` and `deepfake/<id>` (ID 5 uses `.mov`). The local five real/fake pairs match the Kaggle file-list sizes and extensions exactly. The source CSV establishes these five label/path mappings; exact byte identity was not verified.
- [The DFD Kaggle mirror](https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset) says it was downloaded from the official FaceForensics server. The [official FaceForensics dataset documentation](https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md) describes DFD original names as `<actor>__<scene>` and manipulated names as `<target actor>_<source actor>__<scene>__<experiment id>`. This gives the target-original association when original names are retained.
- Kaggle's DFD file listing has 3,068 manipulated video entries and 364 original-path entries (one is a zero-byte temporary entry); the upstream docs describe 363 DFD originals and more than 3,000 manipulations. The local folders have 3,078 real and 3,073 deepfake files, including the five UniData preview pairs.
- DFD's local files were numerically renamed, and no local source-name manifest or CSV remains. Kaggle file sizes uniquely suggest source filenames for all 3,068 local DFD fake files and 2,961 local DFD real files; 112 real files have ambiguous same-size source candidates. Size is not a content hash, so these are metadata matches, not verified per-file identities. The same numeric ID across local folders is therefore not sufficient proof of every DFD original/fake mapping.

The local inventory has 3,073 shared numeric IDs, five real-only IDs (3074–3078), and no fake-only numeric IDs. Treat these as filename-associated pairs, not all source-verified pairs. The DFD original-to-fake links need a retained source-name manifest or direct content hashes before their identity can be called verified.

This dataset addition remains separate from temporal pretraining. The temporal model is the frozen feature backbone; `classifier.py` supplies the separate binary classification head and training workflow.

## Historical pre-classifier audit (2026-10-28)

The following records the state before the classifier pipeline was implemented and is retained as provenance context. Its unaccepted split and "blocked" conclusion were superseded by the current split run and classifier implementation documented in the next section.

The requested classification work is **blocked pending dataset verification**. No classifier, classifier checkpoint, split manifest, or evaluation command has been added. The original temporal model and checkpoints were not modified.

### Checks completed

- `runs/vista_df/latest.pt` loaded into `VistaTemporalModel` with `strict=True` (all keys matched); saved state is epoch 1, global step 16089, audio enabled, VideoMAE frozen.
- Real and deepfake `1.mp4` passed the production video decoder, VideoMAE preprocessing, and GPU forward pass with finite outputs: frame features `(1,16,768)` and next-frame predictions `(1,15,768)`. Both also passed Whisper extraction/fusion with finite features. This verifies pipeline operation, not class separation.
- UniData IDs 1–5 have an explicit public CSV mapping and local path/extension/file-size matches. DFD provenance is supported by Kaggle's stated source and official FaceForensics naming documentation. The loss of original names in the numeric copy means source-level DFD pair identity remains partial: file-size matching is unique for 3,068 fake and 2,961 real clips, ambiguous for 112 real clips, and is not cryptographic verification.
- All 6,151 local files passed `ffprobe` stream/metadata checks and yielded a first-frame thumbnail for the perceptual screen. A seeded full-decode sample of 20 real + 20 fake videos had zero failures. This is representative sampling, not a full decode of all files. Seventeen candidate videos did not yield the second sampled frame; these failures are not counted as full-video decode failures.
- Sparse pHash screening: first frame per video at 32x32; candidate pairs required first-frame Hamming distance <=6, then a second early frame was checked and retained at distance <=8. This found 63,842 candidate edges and 278 connected components (largest 400 videos). These are **similarity-screen candidates**, not manually verified duplicates; shared scenes/appearance can cause false positives, and later-frame-only duplicates can be missed.
- A deterministic in-memory group allocation joined same numeric IDs and all pHash candidate edges. It had zero pair edges or detected pHash edges crossing splits, but its actual counts were train 1,244 real + 1,244 fake, validation 918 + 916, test 916 + 913 (about 40/30/30, not the intended 70/15/15). It was not saved and is **not an accepted split**; refine the allocator and manually/source-verify ambiguous pair and duplicate links before finalizing splits.
- Full-file SHA-256 previously found 343 exact-duplicate groups containing 3,060 files, all in `real/`, with no cross-label exact duplicates. Existing regression suite: `./myenv/bin/python -m unittest discover -s tests -v` passed all five tests.
- Environment observed: GTX 1650 (4 GiB VRAM), 5.7 GiB system RAM, 24 GiB free disk, CUDA available (PyTorch 2.14.1+cu130). The labeled data occupies about 42.53 GiB; the initial three-random-seek hash method was stopped after low throughput, and replaced by sparse FFmpeg thumbnails.

### Gate and next steps

The current architecture has no classification head. A future baseline could pool existing `fused_features` and use a binary logit head with `BCEWithLogitsLoss`; this is **not implemented or tested**. Before classifier training, recover/retain DFD source filenames, resolve the 112 size-ambiguous real files, improve/validate perceptual groups (including later-frame near duplicates), and produce a reproducible group split with the requested proportions. Keep every verified pair/group wholly inside one split and keep test data out of model selection. No classifier training/evaluation command is currently supported.

## Binary classifier implementation and run (2026-10-29)

This section supersedes the historical implementation-status statements above. The temporal model remains a representation learner; the new classifier freezes it and trains a separate binary head using the folder-derived labels:

- `real/` is label `0`; `deepfake/` is label `1`.
- The head pools the temporal model's 16 fused frame features by mean, then applies `LayerNorm(768)`, `Linear(768,256)`, GELU, dropout, and a two-logit output layer.
- The classifier run uses the saved VISTA temporal checkpoint with audio disabled (`--no-audio`). Although the temporal checkpoint was trained with audio, one labeled video (`deepfake/2185.mp4`) has a valid H.264 video stream but no audio stream. Video-only features retain that sample and keep the classifier input consistent. The temporal backbone stays frozen; AdamW and class-weighted cross-entropy update only the binary head.
- Training and validation features are cached atomically under `runs/vista_classifier/features/`. Cache keys include source-file metadata and checkpoint/config identity, avoiding repeat decoding after resume.
- The split workflow groups matching numeric stems across the two folders and joins sparse perceptual-hash candidates. These are conservative leakage-prevention heuristics, not source-verified duplicate identities. Similarity screening samples only the beginning of each clip, so later-only duplicates can be missed.

### Commands

```bash
# Create the seeded train/validation/test manifest and save its complete log.
./myenv/bin/python classifier.py prepare-splits --seed 42

# Start or resume training only the binary head (batch size 1 for the local GPU).
./myenv/bin/python classifier.py train --epochs 10 --batch-size 1 --no-audio
./myenv/bin/python classifier.py train --epochs 10 --batch-size 1 --no-audio --resume

# Evaluate the held-out test split after training has completed.
./myenv/bin/python classifier.py evaluate --split test
```

### Verified split manifest

The dataset contains 6,151 supported videos: 3,078 `real` and 3,073 `deepfake`. The seed-42 manifest keeps every heuristic group within one split and reaches the requested proportions:

| Split | Real (0) | Deepfake (1) | Total | Observed fraction |
|---|---:|---:|---:|---:|
| Train | 2,156 | 2,151 | 4,307 | 70.02% |
| Validation | 461 | 461 | 922 | 14.99% |
| Test | 461 | 461 | 922 | 14.99% |

The audit found 78,681 first-frame candidate pairs, 64,163 retained perceptual-similarity edges, and 278 connected components (276 multi-file groups; largest group 400). It found zero group or retained similarity edges crossing splits. Seventeen videos had no usable second sampled frame; they were retained using first-frame-only associations rather than silently dropped. Those links and all group member paths are recorded in the split manifest and log. Similarity and same-numeric-stem relationships remain heuristic, not verified provenance.

The completed run took about 26 minutes because the prior implementation launched a separate FFmpeg decode for the first frame and a second decode for frame 25 of nearly every video. The implementation now samples both frames in one FFmpeg invocation per video for future split generation; this manifest was preserved rather than recomputed.

### Classifier artifacts

- `runs/vista_classifier/splits.json`: seeded assignments, folder labels, group IDs, class counts, and grouping metadata.
- `runs/vista_classifier/prepare-splits.log`: split-generation progress and invalid/uncertain-group audit.
- `runs/vista_classifier/latest.pt`: atomic resumable classifier-head and optimizer checkpoint.
- `runs/vista_classifier/best.pt`: best completed-epoch validation checkpoint.
- `runs/vista_classifier/history.json`: epoch validation metrics and in-progress cursor.
- `runs/vista_classifier/features/`: reusable frozen representations.
- `runs/vista_classifier/errors.jsonl`: invalid media and decoder diagnostics; samples are not silently dropped.
- `runs/vista_classifier/train.log`: training and retry output.
- `runs/vista_classifier/metrics_test.json`: held-out test metrics after a complete evaluation.

All classifier artifacts are isolated from the original temporal checkpoints in `runs/vista_df/`.

### Current training status

The audio-enabled attempt stopped before its first optimizer update because `deepfake/2185.mp4` has no audio stream. `ffprobe` confirmed that its H.264 video stream is valid. The sample was retained and training restarted with `--no-audio`; the checkpoint records `use_audio=false`. The latest saved progress is epoch 1, batch cursor 40, global step 40. No epoch-level validation, best checkpoint, held-out test evaluation, or final classifier metrics are available yet. Continue without restarting completed batches with:

```bash
./myenv/bin/python classifier.py train --epochs 10 --batch-size 1 --no-audio --resume
```
