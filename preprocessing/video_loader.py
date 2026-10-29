"""Video decoding and VideoMAE preprocessing utilities.

This module is intentionally CPU-only. It decodes a temporal window, then
uses the image processor shipped with the VideoMAE checkpoint so resizing and
normalization follow that checkpoint's configuration.
"""

from pathlib import Path
import os
from contextlib import contextmanager
from typing import Union

import cv2
import numpy as np
import torch


PathLike = Union[str, Path]


class VideoDecodeError(RuntimeError):
    """Signal that the requested clip could not be decoded faithfully."""


cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)


@contextmanager
def _silence_native_stderr():
    """Suppress native FFmpeg diagnostics; Python reports one concise failure."""
    saved_stderr = os.dup(2)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, 2)
        yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(null_fd)
        os.close(saved_stderr)


def _valid_frame(frame: np.ndarray | None) -> bool:
    """Check that a decoded frame is a non-empty three-channel byte image."""
    return (
        frame is not None
        and frame.ndim == 3
        and frame.shape[2] == 3
        and frame.dtype == np.uint8
        and frame.size > 0
    )


def _decode_random(capture, path: Path, indices: np.ndarray) -> list[np.ndarray]:
    """Seek to requested frame indices and convert each BGR frame to RGB."""
    frames = []
    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, bgr = capture.read()
        if not ok or not _valid_frame(bgr):
            raise VideoDecodeError(f"random seek failed at frame {int(frame_index)}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return frames


def _decode_sequential(path: Path, indices: np.ndarray) -> list[np.ndarray]:
    """Decode from the start when seeking fails, keeping requested frames only."""
    with _silence_native_stderr():
        capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        capture.release()
        raise VideoDecodeError("sequential retry could not open video")
    selected: dict[int, np.ndarray] = {}
    wanted = set(int(index) for index in indices)
    try:
        for frame_index in range(max(wanted) + 1):
            ok, bgr = capture.read()
            if not ok or not _valid_frame(bgr):
                raise VideoDecodeError(f"sequential decode failed at frame {frame_index}")
            if frame_index in wanted:
                selected[frame_index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
    if len(selected) != len(wanted):
        raise VideoDecodeError("sequential retry did not produce every requested frame")
    return [selected[int(index)].copy() for index in indices]


def decode_video_frames(
    video_path: PathLike,
    *,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    num_frames: int = 16,
) -> list[np.ndarray]:
    """Decode evenly spaced RGB frames from a video window.

    Repeated indices are used for very short windows. This keeps the model
    input shape fixed and avoids silently sampling outside the requested clip.
    """
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: {path}")
    if num_frames < 1 or start_sec < 0:
        raise ValueError("num_frames must be positive and start_sec nonnegative")

    with _silence_native_stderr():
        capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise VideoDecodeError(f"OpenCV could not open video: {path}")
    try:
        with _silence_native_stderr():
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        if frame_count <= 0 or not np.isfinite(fps) or fps <= 0:
            raise RuntimeError(f"Video has invalid frame metadata: {path}")

        duration = frame_count / fps
        stop = duration if end_sec is None else min(float(end_sec), duration)
        start = min(float(start_sec), max(0.0, duration - 1.0 / fps))
        if stop <= start:
            raise ValueError(f"Empty video window [{start_sec}, {end_sec}] for {path}")

        # Include endpoints while keeping all sampled timestamps within the clip.
        timestamps = np.linspace(start, max(start, stop - 1.0 / fps), num_frames)
        indices = np.clip(np.rint(timestamps * fps).astype(np.int64), 0, frame_count - 1)
        try:
            # Some codecs cannot seek reliably; sequential decoding is the fallback.
            with _silence_native_stderr():
                frames = _decode_random(capture, path, indices)
        except VideoDecodeError as random_error:
            capture.release()
            try:
                with _silence_native_stderr():
                    frames = _decode_sequential(path, indices)
            except (VideoDecodeError, cv2.error) as sequential_error:
                raise VideoDecodeError(
                    f"Could not decode requested frames from {path}; "
                    f"random retry: {random_error}; sequential retry: {sequential_error}"
                ) from sequential_error
        if len(frames) != num_frames or any(not _valid_frame(frame) for frame in frames):
            raise VideoDecodeError(f"Decoder returned an invalid clip for {path}")
        dimensions = {frame.shape for frame in frames}
        if len(dimensions) != 1:
            raise VideoDecodeError(f"Inconsistent frame dimensions in {path}: {dimensions}")
        return frames
    except cv2.error as exc:
        raise VideoDecodeError(f"OpenCV failed to decode {path}: {exc}") from exc
    finally:
        capture.release()


def preprocess_video_frames(frames: list[np.ndarray], image_processor) -> torch.Tensor:
    """Normalize RGB frames with the checkpoint processor into ``(T,C,H,W)``."""
    if not frames:
        raise ValueError("At least one frame is required")
    encoded = image_processor(images=frames, return_tensors="pt")
    pixels = encoded["pixel_values"]
    if pixels.ndim == 5 and pixels.shape[0] == 1:
        pixels = pixels.squeeze(0)
    if pixels.ndim != 4 or pixels.shape[0] != len(frames) or pixels.shape[1] != 3:
        raise RuntimeError(f"Unexpected processor output shape: {tuple(pixels.shape)}")
    crop = getattr(image_processor, "crop_size", None) or getattr(image_processor, "size", None)
    if isinstance(crop, dict):
        expected_height = crop.get("height", crop.get("shortest_edge"))
        expected_width = crop.get("width", crop.get("shortest_edge"))
        if expected_height and expected_width and tuple(pixels.shape[-2:]) != (expected_height, expected_width):
            raise RuntimeError(f"Expected processed frame size {(expected_height, expected_width)}, got {tuple(pixels.shape[-2:])}")
    pixels = pixels.to(dtype=torch.float32)
    if not torch.isfinite(pixels).all():
        raise RuntimeError("Video processor returned non-finite normalized pixels")
    return pixels.contiguous()

