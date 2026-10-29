"""Train and evaluate a binary classifier on frozen VISTA-DF features."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import subprocess
import tempfile

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.audio_network import WhisperFeatureExtractor
from models.binary_classifier import BinaryClassificationHead
from models.vista_model import VistaTemporalModel
from preprocessing.audio_loader import decode_audio_window
from preprocessing.video_loader import decode_video_frames, preprocess_video_frames


LOGGER = logging.getLogger("vista_classifier")
LABELS = {"real": 0, "deepfake": 1}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg"}


def _atomic_json_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _atomic_torch_save(payload: dict | torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _phash(frame: np.ndarray | None) -> int | None:
    if frame is None or frame.size == 0 or float(frame.std()) < 4.0:
        return None
    coefficients = cv2.dct(frame.astype(np.float32))[:8, :8].reshape(-1)
    threshold = float(np.median(coefficients[1:]))
    value = 0
    for bit in coefficients > threshold:
        value = (value << 1) | int(bit)
    return value


def _sample_frame_pair(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
        "-vf", "select='eq(n,0)+eq(n,25)',scale=32:32,format=gray",
        "-vsync", "0", "-frames:v", "2", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode or len(result.stdout) < 1024 or len(result.stdout) % 1024:
        return None, None
    frames = [
        np.frombuffer(result.stdout[offset:offset + 1024], dtype=np.uint8).reshape(32, 32).copy()
        for offset in range(0, len(result.stdout), 1024)
    ]
    if len(frames) > 2:
        return None, None
    return frames[0], frames[1] if len(frames) == 2 else None


def _collect_video_records(dataset_root: Path) -> list[dict]:
    records = []
    invalid_paths = []
    for class_name, label in LABELS.items():
        class_root = dataset_root / class_name
        if not class_root.is_dir():
            raise FileNotFoundError(f"Required class folder is missing: {class_root}")
        for path in sorted(class_root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in VIDEO_SUFFIXES:
                invalid_paths.append((path, "unsupported file extension"))
                continue
            records.append({
                "path": path.relative_to(dataset_root).as_posix(),
                "label": label,
                "label_name": class_name,
                "numeric_stem": path.stem,
            })
    for path, reason in invalid_paths:
        LOGGER.warning("Ignoring non-video dataset file %s: %s", path, reason)
    if not records:
        raise ValueError(f"No supported videos found under {dataset_root}")
    counts = Counter(record["label_name"] for record in records)
    if set(counts) != set(LABELS) or min(counts.values()) == 0:
        raise ValueError(f"Expected non-empty real/deepfake folders, found {dict(counts)}")
    return records


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _candidate_pairs(records: list[dict], hashes: list[int | None], threshold: int) -> list[tuple[int, int, int]]:
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, value in enumerate(hashes):
        if value is None:
            continue
        for shift in range(0, 64, 8):
            buckets[(shift, (value >> shift) & 255)].append(index)
    possible = set()
    for members in buckets.values():
        if len(members) > 1:
            for offset, left in enumerate(members[:-1]):
                for right in members[offset + 1:]:
                    possible.add((left, right) if left < right else (right, left))
    edges = []
    for left, right in possible:
        distance = _hamming(hashes[left], hashes[right])
        if distance <= threshold:
            edges.append((left, right, distance))
    return edges


def _union_find(size: int):
    parents = list(range(size))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    return find, union


def _assign_group_splits(
    records: list[dict], groups: dict[str, list[int]], fractions: dict[str, float], seed: int
) -> dict[int, str]:
    total_by_label = Counter(record["label"] for record in records)
    counts = {split: Counter() for split in fractions}
    grouped = list(groups.items())
    random.Random(seed).shuffle(grouped)
    grouped.sort(key=lambda item: -len(item[1]))
    assigned: dict[int, str] = {}

    def fill_score(split: str, additions: Counter) -> tuple[float, float]:
        ratios = [
            (counts[split][label] + additions[label]) / max(1.0, total * fractions[split])
            for label, total in total_by_label.items()
        ]
        return max(ratios), sum(ratios)

    for group_id, indices in grouped:
        additions = Counter(records[index]["label"] for index in indices)
        selected_split = min(fractions, key=lambda split: fill_score(split, additions))
        for index in indices:
            assigned[index] = selected_split
        counts[selected_split].update(additions)
    return assigned


def prepare_splits(
    dataset_root: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    first_hash_threshold: int = 6,
    second_hash_threshold: int = 8,
    fractions: dict[str, float] | None = None,
) -> Path:
    fractions = fractions or {"train": 0.70, "validation": 0.15, "test": 0.15}
    if set(fractions) != {"train", "validation", "test"} or not math.isclose(sum(fractions.values()), 1.0):
        raise ValueError("Split fractions must define train/validation/test and sum to 1")
    records = _collect_video_records(dataset_root)
    LOGGER.info("Computing first/second sampled-frame hashes for %d files", len(records))
    hashes = []
    second_hashes: dict[int, int | None] = {}
    decode_failures = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(_sample_frame_pair, dataset_root / record["path"]): index
            for index, record in enumerate(records)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            index = futures[future]
            record = records[index]
            first_frame, second_frame = future.result()
            if first_frame is None:
                decode_failures.append((record["path"], "failed to decode first frame for pHash"))
                hashes.append((index, None))
            else:
                hashes.append((index, _phash(first_frame)))
                second_hashes[index] = _phash(second_frame)
            if completed % 500 == 0:
                LOGGER.info("Hashed sampled frames for %d/%d videos", completed, len(records))
    hashes = [value for _, value in sorted(hashes)]
    if decode_failures:
        for relative_path, reason in decode_failures:
            LOGGER.error("Dataset audit decode failure %s: %s", relative_path, reason)
        raise RuntimeError(f"Split generation aborted: {len(decode_failures)} videos failed first-frame decoding")

    first_candidates = _candidate_pairs(records, hashes, first_hash_threshold)
    candidate_indices = sorted({index for left, right, _ in first_candidates for index in (left, right)})
    LOGGER.info("Comparing second sampled frames for %d candidate videos", len(candidate_indices))
    second_failures = [
        records[index]["path"]
        for index in candidate_indices
        if second_hashes.get(index) is None
    ]

    near_edges = []
    for left, right, first_distance in first_candidates:
        left_hash, right_hash = second_hashes.get(left), second_hashes.get(right)
        if left_hash is None or right_hash is None:
            near_edges.append((left, right, first_distance, None, "first_frame_only"))
            continue
        second_distance = _hamming(left_hash, right_hash)
        if second_distance <= second_hash_threshold:
            near_edges.append((left, right, first_distance, second_distance, "two_sampled_frames"))

    find, union = _union_find(len(records))
    by_stem: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_stem[record["numeric_stem"]].append(index)
    pair_edges = 0
    for indices in by_stem.values():
        real = [i for i in indices if records[i]["label"] == LABELS["real"]]
        fake = [i for i in indices if records[i]["label"] == LABELS["deepfake"]]
        if real and fake:
            union(real[0], fake[0])
            pair_edges += 1
    for left, right, _, _, _ in near_edges:
        union(left, right)

    component_indices: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        component_indices[find(index)].append(index)
    sorted_components = sorted(
        component_indices.values(),
        key=lambda indices: min(records[index]["path"] for index in indices),
    )
    groups = {}
    for group_number, indices in enumerate(sorted_components, 1):
        group_id = f"group_{group_number:05d}"
        groups[group_id] = indices
        for index in indices:
            records[index]["group_id"] = group_id
    uncertain_groups = {
        group_id: [records[index]["path"] for index in indices]
        for group_id, indices in groups.items()
        if len(indices) > 1
    }
    for group_id, members in uncertain_groups.items():
        LOGGER.warning(
            "Heuristic duplicate group %s (%d files; not source-verified): %s",
            group_id,
            len(members),
            ", ".join(members),
        )
    assignments = _assign_group_splits(records, groups, fractions, seed)
    for index, split in assignments.items():
        records[index]["split"] = split

    split_counts = {split: dict(Counter(record["label_name"] for record in records if record["split"] == split)) for split in fractions}
    component_split_leaks = sum(
        records[left]["split"] != records[right]["split"]
        for left, right, _, _, _ in near_edges
    )
    if component_split_leaks:
        raise RuntimeError(f"Internal split error: {component_split_leaks} similarity edges cross splits")
    manifest = {
        "version": 1,
        "seed": seed,
        "label_mapping": LABELS,
        "dataset_root": str(dataset_root.resolve()),
        "split_fractions": fractions,
        "grouping": {
            "pair_rule": "same numeric stem across real/deepfake folders (conservative association, not source-verified)",
            "phash": "32x32 grayscale DCT pHash; first-frame Hamming <=6 then second sampled frame <=8",
            "first_frame_candidates": len(first_candidates),
            "near_duplicate_edges": len(near_edges),
            "first_frame_only_edges_due_to_second_sample_failure": sum(edge[4] == "first_frame_only" for edge in near_edges),
            "second_sample_failures": second_failures,
            "components": len(groups),
            "largest_component": max(map(len, groups.values())),
            "numeric_stem_pair_groups": pair_edges,
            "cross_split_similarity_edges": component_split_leaks,
            "uncertain_group_count": len(uncertain_groups),
            "uncertain_groups": uncertain_groups,
            "uncertainty_note": "Filename and sparse perceptual-hash associations are heuristic; source-level duplicate identity is not verified.",
        },
        "split_counts": split_counts,
        "records": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "splits.json"
    _atomic_json_save(manifest, manifest_path)
    _atomic_json_save({"seed": seed, "fractions": fractions, "dataset_root": str(dataset_root.resolve())}, output_dir / "config.json")
    LOGGER.info("Saved split manifest %s; counts=%s", manifest_path, split_counts)
    return manifest_path


def _load_manifest(manifest_path: Path, dataset_root: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_root = str(dataset_root.resolve())
    if manifest.get("dataset_root") != expected_root:
        raise ValueError(f"Manifest dataset root mismatch: {manifest.get('dataset_root')} != {expected_root}")
    records = manifest.get("records", [])
    valid_splits = {"train", "validation", "test"}
    group_splits = {}
    actual_counts = {split: Counter() for split in valid_splits}
    for record in records:
        expected_label = LABELS.get(Path(record["path"]).parts[0])
        if expected_label is None or expected_label != record.get("label"):
            raise ValueError(f"Label/path mismatch in split manifest: {record}")
        if record.get("label_name") != Path(record["path"]).parts[0]:
            raise ValueError(f"Label folder/name mismatch in split manifest: {record}")
        split, group_id = record.get("split"), record.get("group_id")
        if split not in valid_splits or not group_id:
            raise ValueError(f"Missing or invalid split/group assignment in manifest: {record}")
        prior_split = group_splits.setdefault(group_id, split)
        if prior_split != split:
            raise ValueError(f"Duplicate group {group_id} crosses splits {prior_split!r} and {split!r}")
        actual_counts[split][record["label_name"]] += 1
    expected_counts = manifest.get("split_counts")
    if expected_counts is not None:
        normalized_counts = {
            split: dict(actual_counts[split])
            for split in valid_splits
        }
        if expected_counts != normalized_counts:
            raise ValueError(f"Manifest split counts are inconsistent: {expected_counts} != {normalized_counts}")
    return manifest


def _load_temporal_backbone(checkpoint_path: Path, video_checkpoint: Path, audio_checkpoint: Path, device: torch.device, use_audio: bool):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    model_state = state["model"]
    audio_weight = model_state.get("fusion.audio_projection.weight")
    audio_dim = int(audio_weight.shape[1]) if audio_weight is not None else None
    model = VistaTemporalModel(str(video_checkpoint), encoder_trainable=False, audio_dim=audio_dim)
    model.load_state_dict(model_state, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    audio_model = None
    saved_config = state.get("config", {})
    if use_audio:
        if audio_dim is None:
            raise ValueError("The selected temporal checkpoint has no audio projection")
        if not saved_config.get("use_audio", False):
            LOGGER.warning("Audio enabled for classification although the temporal checkpoint was trained without audio")
        audio_model = WhisperFeatureExtractor(str(audio_checkpoint)).to(device).eval()
        for parameter in audio_model.parameters():
            parameter.requires_grad_(False)
    return model, audio_model, state


def extract_video_feature(
    record: dict,
    dataset_root: Path,
    model: VistaTemporalModel,
    audio_model: WhisperFeatureExtractor | None,
    device: torch.device,
    *,
    window_seconds: float = 2.0,
    amp: bool = True,
) -> torch.Tensor:
    path = Path(record.get("absolute_path", dataset_root / record["path"]))
    if "label" in record and (path.parent.name not in LABELS or LABELS[path.parent.name] != record["label"]):
        raise ValueError(f"Label/path mismatch; refusing sample: {record}")
    if not path.is_file():
        raise FileNotFoundError(f"Manifest video is missing: {path}")
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open labeled video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    duration = frame_count / fps if frame_count > 0 and math.isfinite(fps) and fps > 0 else 0.0
    if duration <= 0:
        raise RuntimeError(f"Invalid video metadata: {path}")
    clip_duration = min(window_seconds, duration)
    start = max(0.0, (duration - clip_duration) / 2.0)
    frames = decode_video_frames(
        path,
        start_sec=start,
        end_sec=start + clip_duration,
        num_frames=model.video.num_frames,
    )
    pixels = preprocess_video_frames(frames, model.video.image_processor).unsqueeze(0).to(device)
    audio_features = None
    if audio_model is not None:
        waveform = decode_audio_window(path, start_sec=start, duration_sec=clip_duration)
        mel = audio_model.waveform_to_features([waveform.numpy()]).to(device)
    else:
        mel = None
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp)
        if device.type == "cuda" else nullcontext()
    )
    with torch.no_grad(), autocast:
        if mel is not None:
            audio_features = audio_model(mel)
        output = model(pixels, audio_features=audio_features)
        feature = output["fused_features"].float().mean(dim=1).squeeze(0)
    if feature.shape != (model.fusion.output[1].out_features,):
        raise RuntimeError(f"Unexpected pooled VISTA-DF feature shape {tuple(feature.shape)} for {path}")
    if not torch.isfinite(feature).all():
        raise FloatingPointError(f"Non-finite representation for {path}")
    return feature.detach()


def _metrics(labels: list[int], probabilities: list[float]) -> dict:
    confusion = [[0, 0], [0, 0]]
    predictions = [int(score >= 0.5) for score in probabilities]
    for expected, predicted in zip(labels, predictions):
        confusion[expected][predicted] += 1
    tn, fp = confusion[0]
    fn, tp = confusion[1]
    accuracy = (tn + tp) / max(1, len(labels))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    positives = [score for label, score in zip(labels, probabilities) if label == 1]
    negatives = [score for label, score in zip(labels, probabilities) if label == 0]
    auc = None
    if positives and negatives:
        wins = sum(1.0 if positive > negative else 0.5 if positive == negative else 0.0
                   for positive in positives for negative in negatives)
        auc = wins / (len(positives) * len(negatives))
    return {
        "count": len(labels),
        "accuracy": accuracy,
        "precision_deepfake": precision,
        "recall_deepfake": recall,
        "f1_deepfake": f1,
        "roc_auc": auc,
        "confusion_matrix_real_deepfake": confusion,
    }


def _load_run_models(args, device: torch.device):
    base_checkpoint = Path(args.temporal_checkpoint)
    video_checkpoint = Path(args.video_checkpoint)
    audio_checkpoint = Path(args.audio_checkpoint)
    model, audio_model, temporal_state = _load_temporal_backbone(
        base_checkpoint, video_checkpoint, audio_checkpoint, device, args.use_audio
    )
    head = BinaryClassificationHead(input_dim=model.fusion.output[1].out_features).to(device)
    return model, audio_model, temporal_state, head


def _append_error(output_dir: Path, record: dict, error: Exception) -> None:
    path = output_dir / "errors.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "path": record.get("path"),
        "label": record.get("label"),
        "error": str(error),
    }
    stderr = getattr(error, "stderr", None)
    if stderr:
        entry["stderr"] = stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else str(stderr)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    LOGGER.error(
        "Invalid labeled sample %s (label=%s): %s%s",
        record.get("path"),
        record.get("label_name"),
        error,
        f"; stderr={entry['stderr']}" if "stderr" in entry else "",
    )


def _checkpoint_signature(path: Path) -> dict:
    path = path.resolve()
    if path.is_file():
        stat = path.stat()
        return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if path.is_dir():
        files = []
        for item in sorted(path.rglob("*")):
            if item.is_file():
                stat = item.stat()
                files.append((item.relative_to(path).as_posix(), stat.st_size, stat.st_mtime_ns))
        return {"path": str(path), "files": files}
    raise FileNotFoundError(f"Feature checkpoint does not exist: {path}")


def _feature_cache_identity(args, amp: bool) -> dict:
    paths = [Path(args.temporal_checkpoint), Path(args.video_checkpoint)]
    if args.use_audio:
        paths.append(Path(args.audio_checkpoint))
    return {
        "version": 1,
        "checkpoints": [_checkpoint_signature(path) for path in paths],
        "use_audio": bool(args.use_audio),
        "amp": bool(amp),
        "window_seconds": 2.0,
        "pooling": "mean_fused_features",
    }


def _feature_cache_path(
    record: dict, dataset_root: Path, cache_dir: Path, cache_identity: dict
) -> Path:
    source = dataset_root / record["path"]
    stat = source.stat()
    key = json.dumps({
        "identity": cache_identity,
        "dataset_root": str(dataset_root.resolve()),
        "source": record["path"],
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return cache_dir / digest[:2] / f"{digest}.pt"


def _load_or_extract_feature(cache_path: Path, extractor) -> torch.Tensor:
    if cache_path.is_file():
        feature = torch.load(cache_path, map_location="cpu", weights_only=True)
        if not isinstance(feature, torch.Tensor) or feature.shape != (768,):
            raise ValueError(f"Invalid cached feature at {cache_path}")
        if not torch.isfinite(feature).all():
            raise FloatingPointError(f"Non-finite cached feature at {cache_path}")
        return feature
    feature = extractor().detach().float().cpu()
    if feature.shape != (768,) or not torch.isfinite(feature).all():
        raise ValueError(f"Invalid extracted feature for cache {cache_path}")
    _atomic_torch_save(feature, cache_path)
    return feature


def _predict_records(
    records, dataset_root, model, audio_model, head, device, amp, max_batches=None,
    feature_cache_dir: Path | None = None, cache_identity: dict | None = None,
    error_output_dir: Path | None = None,
):
    labels, fake_probabilities = [], []
    for index, record in enumerate(records):
        if max_batches is not None and index >= max_batches:
            break
        try:
            if feature_cache_dir is not None and cache_identity is not None:
                cache_path = _feature_cache_path(record, dataset_root, feature_cache_dir, cache_identity)
                feature = _load_or_extract_feature(
                    cache_path,
                    lambda: extract_video_feature(record, dataset_root, model, audio_model, device, amp=amp),
                )
            else:
                feature = extract_video_feature(record, dataset_root, model, audio_model, device, amp=amp)
        except Exception as exc:
            if error_output_dir is not None:
                _append_error(error_output_dir, record, exc)
            raise
        with torch.no_grad():
            logits = head(feature.unsqueeze(0).to(device))
            probability = float(BinaryClassificationHead.probabilities(logits)[0, 1])
        labels.append(int(record["label"]))
        fake_probabilities.append(probability)
    return labels, fake_probabilities


def _save_classifier_state(path: Path, head, optimizer, *, epoch: int, next_batch: int,
                           global_step: int, history: list, config: dict, best_loss: float) -> None:
    _atomic_torch_save({
        "head": head.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "next_batch": next_batch,
        "global_step": global_step,
        "history": history,
        "config": config,
        "best_validation_loss": best_loss,
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, path)


def train(args) -> None:
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.split_manifest)
    dataset_root = Path(args.dataset_root)
    manifest = _load_manifest(manifest_path, dataset_root)
    records = [row for row in manifest["records"] if row["split"] in {"train", "validation"}]
    train_records = [row for row in records if row["split"] == "train"]
    val_records = [row for row in records if row["split"] == "validation"]
    if not train_records or not val_records:
        raise ValueError("Manifest must have non-empty train and validation splits")
    seed = int(args.seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, audio_model, temporal_state, head = _load_run_models(args, device)
    config = {
        "dataset_root": str(dataset_root.resolve()),
        "split_manifest": str(manifest_path.resolve()),
        "temporal_checkpoint": str(Path(args.temporal_checkpoint).resolve()),
        "temporal_global_step": temporal_state.get("global_step"),
        "video_checkpoint": str(Path(args.video_checkpoint).resolve()),
        "audio_checkpoint": str(Path(args.audio_checkpoint).resolve()) if args.use_audio else None,
        "use_audio": args.use_audio,
        "amp": bool(args.amp and device.type == "cuda"),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": seed,
        "head": {"input_dim": 768, "hidden_dim": args.hidden_dim, "dropout": args.dropout},
        "objective": "CrossEntropyLoss over REAL=0, DEEPFAKE=1 logits",
        "window_seconds": 2.0,
    }
    _atomic_json_save(config, output_dir / "config.json")
    feature_cache_dir = output_dir / "features"
    cache_identity = _feature_cache_identity(args, config["amp"])
    class_counts = Counter(row["label"] for row in train_records)
    weights = torch.tensor(
        [len(train_records) / (2 * max(1, class_counts[label])) for label in (0, 1)],
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    start_epoch, next_batch, global_step, history, best_loss = 0, 0, 0, [], math.inf
    latest_path, best_path = output_dir / "latest.pt", output_dir / "best.pt"
    if args.resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"No classifier checkpoint to resume: {latest_path}")
        saved = torch.load(latest_path, map_location="cpu", weights_only=False)
        for key in ("dataset_root", "split_manifest", "temporal_checkpoint", "use_audio", "batch_size", "seed"):
            if saved["config"].get(key) != config.get(key):
                raise ValueError(f"Cannot resume with changed {key}: {saved['config'].get(key)} != {config.get(key)}")
        head.load_state_dict(saved["head"], strict=True)
        if saved.get("optimizer"):
            optimizer.load_state_dict(saved["optimizer"])
        start_epoch, next_batch = int(saved["epoch"]), int(saved["next_batch"])
        global_step, history = int(saved["global_step"]), list(saved.get("history", []))
        best_loss = float(saved.get("best_validation_loss", math.inf))
        random.setstate(saved["rng_python"]); np.random.set_state(saved["rng_numpy"])
        torch.set_rng_state(saved["rng_torch"])
        if torch.cuda.is_available() and saved.get("rng_cuda"):
            torch.cuda.set_rng_state_all(saved["rng_cuda"])
        LOGGER.info("Resumed classifier epoch=%d next_batch=%d step=%d", start_epoch, next_batch, global_step)

    _save_classifier_state(latest_path, head, optimizer, epoch=start_epoch, next_batch=next_batch,
                           global_step=global_step, history=history, config=config, best_loss=best_loss)
    history_path = output_dir / "history.json"
    _atomic_json_save({
        "history": history,
        "in_progress": {
            "epoch": start_epoch + 1,
            "next_batch": next_batch,
            "global_step": global_step,
        },
    }, history_path)
    max_epochs = args.epochs
    for epoch in range(start_epoch, max_epochs):
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(train_records), generator=generator).tolist()
        if epoch == start_epoch:
            order = order[next_batch * args.batch_size:]
        head.train()
        batches_processed = 0
        epoch_loss_sum = 0.0
        epoch_sample_count = 0
        for start in range(0, len(order), args.batch_size):
            batch_indices = order[start:start + args.batch_size]
            features, labels = [], []
            for index in batch_indices:
                record = train_records[index]
                try:
                    cache_path = _feature_cache_path(record, dataset_root, feature_cache_dir, cache_identity)
                    features.append(_load_or_extract_feature(
                        cache_path,
                        lambda: extract_video_feature(
                            record, dataset_root, model, audio_model, device, amp=config["amp"]
                        ),
                    ))
                except Exception as exc:
                    _append_error(output_dir, record, exc)
                    _save_classifier_state(latest_path, head, optimizer, epoch=epoch,
                                           next_batch=next_batch, global_step=global_step,
                                           history=history, config=config, best_loss=best_loss)
                    raise
                labels.append(int(record["label"]))
            batch_features = torch.stack(features).to(device)
            batch_labels = torch.tensor(labels, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(batch_features)
            loss = F.cross_entropy(logits, batch_labels, weight=weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite classifier loss at step {global_step}")
            loss.backward()
            optimizer.step()
            global_step += 1
            next_batch += 1
            batches_processed += 1
            epoch_loss_sum += float(loss.detach()) * len(labels)
            epoch_sample_count += len(labels)
            if global_step % args.checkpoint_interval == 0:
                _save_classifier_state(latest_path, head, optimizer, epoch=epoch, next_batch=next_batch,
                                       global_step=global_step, history=history, config=config, best_loss=best_loss)
                _atomic_json_save({
                    "history": history,
                    "in_progress": {
                        "epoch": epoch + 1,
                        "next_batch": next_batch,
                        "global_step": global_step,
                    },
                }, history_path)
            if global_step % args.log_interval == 0:
                LOGGER.info("epoch=%d step=%d loss=%.5f", epoch + 1, global_step, float(loss.detach()))
            if args.max_train_batches is not None and batches_processed >= args.max_train_batches:
                break

        head.eval()
        try:
            val_labels, val_scores = _predict_records(
                val_records, dataset_root, model, audio_model, head, device, config["amp"], args.max_val_batches,
                feature_cache_dir, cache_identity, output_dir,
            )
        except Exception as exc:
            LOGGER.error("Validation feature extraction failed: %s", exc)
            _save_classifier_state(latest_path, head, optimizer, epoch=epoch, next_batch=next_batch,
                                   global_step=global_step, history=history, config=config, best_loss=best_loss)
            raise
        validation = _metrics(val_labels, val_scores)
        train_loss = epoch_loss_sum / max(1, epoch_sample_count)
        entry = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_batches": batches_processed,
            "last_train_loss": train_loss,
            "validation": validation,
            "bounded_smoke": args.max_train_batches is not None,
        }
        history.append(entry)
        val_loss = 1.0 - validation["accuracy"]
        improved = val_loss < best_loss
        best_loss = min(best_loss, val_loss)
        if args.max_train_batches is not None and (next_batch * args.batch_size) < len(train_records):
            _save_classifier_state(latest_path, head, optimizer, epoch=epoch, next_batch=next_batch,
                                   global_step=global_step, history=history, config=config, best_loss=best_loss)
            _atomic_json_save({"history": history}, history_path)
            LOGGER.info("Bounded smoke run saved at epoch=%d next_batch=%d; validation=%s", epoch, next_batch, validation)
            return
        next_batch = 0
        _save_classifier_state(latest_path, head, optimizer, epoch=epoch + 1, next_batch=0,
                               global_step=global_step, history=history, config=config, best_loss=best_loss)
        if improved:
            _save_classifier_state(best_path, head, optimizer, epoch=epoch + 1, next_batch=0,
                                   global_step=global_step, history=history, config=config, best_loss=best_loss)
        _atomic_json_save({"history": history}, history_path)
        LOGGER.info("Completed epoch %d validation=%s", epoch + 1, validation)


def evaluate(args) -> dict:
    output_dir, dataset_root = Path(args.output_dir), Path(args.dataset_root)
    manifest = _load_manifest(Path(args.split_manifest), dataset_root)
    records = [row for row in manifest["records"] if row["split"] == args.split]
    if not records:
        raise ValueError(f"Split {args.split!r} is empty")
    checkpoint_path = Path(args.classifier_checkpoint or (output_dir / "best.pt"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_audio = bool(checkpoint["config"]["use_audio"])
    args.use_audio = use_audio
    model, audio_model, _, head = _load_run_models(args, device)
    head.load_state_dict(checkpoint["head"], strict=True); head.eval()
    labels, scores = _predict_records(records, dataset_root, model, audio_model, head, device,
                                      bool(checkpoint["config"]["amp"]), args.max_batches,
                                      output_dir / "features",
                                      _feature_cache_identity(args, bool(checkpoint["config"]["amp"])),
                                      output_dir)
    metrics = _metrics(labels, scores)
    _atomic_json_save(metrics, output_dir / f"metrics_{args.split}.json")
    print(json.dumps(metrics, indent=2))
    return metrics


def predict(args) -> dict:
    output_dir, dataset_root = Path(args.output_dir), Path(args.dataset_root)
    checkpoint_path = Path(args.classifier_checkpoint or (output_dir / "best.pt"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    input_path = Path(args.video).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Inference video does not exist: {input_path}")
    record = {"path": str(input_path), "absolute_path": str(input_path)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.use_audio = bool(checkpoint["config"]["use_audio"])
    model, audio_model, _, head = _load_run_models(args, device)
    head.load_state_dict(checkpoint["head"], strict=True); head.eval()
    feature = extract_video_feature(record, dataset_root, model, audio_model, device,
                                    amp=bool(checkpoint["config"]["amp"]))
    with torch.no_grad():
        probabilities = BinaryClassificationHead.probabilities(head(feature.unsqueeze(0).to(device)))[0]
    result = {
        "video": str(input_path),
        "predicted_label": "DEEPFAKE" if int(probabilities.argmax()) == 1 else "REAL",
        "probabilities": {"REAL": float(probabilities[0]), "DEEPFAKE": float(probabilities[1])},
    }
    print(json.dumps(result, indent=2))
    return result


def _add_model_arguments(parser):
    root = Path(__file__).resolve().parent
    parser.add_argument("--dataset-root", default=str(root / "Real vs deepfake"))
    parser.add_argument("--output-dir", default=str(root / "runs/vista_classifier"))
    parser.add_argument("--split-manifest", default=str(root / "runs/vista_classifier/splits.json"))
    parser.add_argument("--temporal-checkpoint", default=str(root / "runs/vista_df/best.pt"))
    parser.add_argument("--video-checkpoint", default=str(root / "pretrained_models/videomae-base"))
    parser.add_argument("--audio-checkpoint", default=str(root / "pretrained_models/whisper-small"))
    parser.add_argument("--no-audio", dest="use_audio", action="store_false")
    parser.set_defaults(use_audio=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare-splits", help="Audit videos and create reproducible grouped splits")
    _add_model_arguments(prepare_parser)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument("--first-hash-threshold", type=int, default=6)
    prepare_parser.add_argument("--second-hash-threshold", type=int, default=8)

    train_parser = subparsers.add_parser("train", help="Train only the binary classification head")
    _add_model_arguments(train_parser)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--hidden-dim", type=int, default=256)
    train_parser.add_argument("--dropout", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--checkpoint-interval", type=int, default=10)
    train_parser.add_argument("--log-interval", type=int, default=10)
    train_parser.add_argument("--max-train-batches", type=int)
    train_parser.add_argument("--max-val-batches", type=int)
    train_parser.add_argument("--no-amp", dest="amp", action="store_false")
    train_parser.set_defaults(amp=True)
    train_parser.add_argument("--resume", action="store_true")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the held-out split")
    _add_model_arguments(eval_parser)
    eval_parser.add_argument("--split", choices=("validation", "test"), default="test")
    eval_parser.add_argument("--classifier-checkpoint")
    eval_parser.add_argument("--max-batches", type=int)

    predict_parser = subparsers.add_parser("predict", help="Predict REAL/DEEPFAKE probabilities for one labeled-folder video")
    _add_model_arguments(predict_parser)
    predict_parser.add_argument("--video", required=True)
    predict_parser.add_argument("--classifier-checkpoint")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "prepare-splits":
        path = prepare_splits(Path(args.dataset_root), Path(args.output_dir), seed=args.seed,
                              first_hash_threshold=args.first_hash_threshold,
                              second_hash_threshold=args.second_hash_threshold)
        print(path)
    elif args.command == "train":
        if args.batch_size < 1 or args.epochs < 1:
            raise ValueError("batch-size and epochs must be positive")
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()