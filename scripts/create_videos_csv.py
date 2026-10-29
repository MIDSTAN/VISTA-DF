"""Scan class video folders and write technical metadata to ``videos.csv``.

Uses FFprobe for duration, stream, frame-rate, and resolution details; the
result is input to dataset splitting and window generation.
"""

import csv
import json
import subprocess
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

# Your videos folder
VIDEOS_DIR = Path(
    "/home/midstan/Documents/Python/VISTA-DF/dataset/K700/videos"
)

# Output CSV
OUTPUT_CSV = Path(
    "/home/midstan/Documents/Python/VISTA-DF/dataset/K700/metadata/videos.csv"
)

# Supported video extensions
VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".mpeg",
    ".mpg",
}


# ============================================================
# CHECK INPUT DIRECTORY
# ============================================================

if not VIDEOS_DIR.exists():
    print(f"ERROR: Videos directory does not exist:")
    print(VIDEOS_DIR)
    exit(1)


# Create metadata directory
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# GET VIDEO INFORMATION USING FFPROBE
# ============================================================

def get_video_info(video_path):
    """Return duration, video/audio stream metadata, and status for one file."""

    command = [
        "ffprobe",
        "-v", "error",

        "-show_entries",
        "format=duration",

        "-show_entries",
        "stream=index,codec_type,width,height,r_frame_rate",

        "-of",
        "json",

        str(video_path)
    ]

    try:

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        data = json.loads(result.stdout)

        duration = float(
            data.get("format", {}).get("duration", 0)
        )

        streams = data.get("streams", [])

        video_stream = None
        audio_stream = None

        for stream in streams:

            if stream.get("codec_type") == "video":
                if video_stream is None:
                    video_stream = stream

            elif stream.get("codec_type") == "audio":
                if audio_stream is None:
                    audio_stream = stream


        # -----------------------------
        # Video information
        # -----------------------------

        if video_stream:

            width = video_stream.get("width", 0)
            height = video_stream.get("height", 0)

            frame_rate = video_stream.get(
                "r_frame_rate",
                "0/1"
            )

            try:

                numerator, denominator = frame_rate.split("/")

                numerator = float(numerator)
                denominator = float(denominator)

                if denominator != 0:
                    fps = numerator / denominator
                else:
                    fps = 0

            except Exception:
                fps = 0

        else:

            width = 0
            height = 0
            fps = 0


        # -----------------------------
        # Audio information
        # -----------------------------

        has_audio = 1 if audio_stream else 0


        return {
            "duration": round(duration, 3),
            "fps": round(fps, 3),
            "width": width,
            "height": height,
            "has_audio": has_audio,
            "status": "OK"
        }


    except Exception as e:

        return {
            "duration": 0,
            "fps": 0,
            "width": 0,
            "height": 0,
            "has_audio": 0,
            "status": f"ERROR: {str(e)}"
        }


# ============================================================
# FIND ALL VIDEOS
# ============================================================

print("=" * 70)
print("VISTA-DF DATASET METADATA GENERATION")
print("=" * 70)

print(f"\nScanning:")
print(VIDEOS_DIR)

video_files = []

for path in VIDEOS_DIR.rglob("*"):

    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
        video_files.append(path)


# Sort for deterministic ordering
video_files.sort()


print(f"\nVideos found: {len(video_files)}")


# ============================================================
# PROCESS VIDEOS
# ============================================================

rows = []

error_count = 0
audio_count = 0
video_count = 0

for index, video_path in enumerate(video_files, start=1):

    # --------------------------------------------------------
    # Class name = immediate parent directory
    # --------------------------------------------------------

    class_name = video_path.parent.name

    # --------------------------------------------------------
    # Relative path
    # --------------------------------------------------------

    relative_path = video_path.relative_to(VIDEOS_DIR)

    # --------------------------------------------------------
    # File size
    # --------------------------------------------------------

    file_size_mb = video_path.stat().st_size / (1024 * 1024)

    # --------------------------------------------------------
    # Get metadata
    # --------------------------------------------------------

    info = get_video_info(video_path)

    # --------------------------------------------------------
    # Unique video ID
    # --------------------------------------------------------

    video_id = f"{index:06d}"

    # --------------------------------------------------------
    # Store row
    # --------------------------------------------------------

    row = {
        "video_id": video_id,
        "class": class_name,
        "path": str(relative_path),
        "duration_sec": info["duration"],
        "fps": info["fps"],
        "width": info["width"],
        "height": info["height"],
        "has_audio": info["has_audio"],
        "file_size_mb": round(file_size_mb, 2),
        "status": info["status"]
    }

    rows.append(row)


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    if info["status"] == "OK":
        video_count += 1
    else:
        error_count += 1

    if info["has_audio"] == 1:
        audio_count += 1


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    print(
        f"[{index}/{len(video_files)}] "
        f"{class_name} / {video_path.name} "
        f"| Audio: {info['has_audio']} "
        f"| Duration: {info['duration']:.2f}s"
    )


# ============================================================
# WRITE CSV
# ============================================================

fieldnames = [
    "video_id",
    "class",
    "path",
    "duration_sec",
    "fps",
    "width",
    "height",
    "has_audio",
    "file_size_mb",
    "status"
]


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(rows)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("METADATA GENERATION COMPLETE")
print("=" * 70)

print(f"Total videos found       : {len(video_files)}")
print(f"Readable videos          : {video_count}")
print(f"Videos with audio        : {audio_count}")
print(f"Videos without audio     : {len(video_files) - audio_count}")
print(f"Videos with errors       : {error_count}")

print(f"\nCSV saved to:")
print(OUTPUT_CSV)

print("=" * 70)
