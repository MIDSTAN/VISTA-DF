"""Download and verify candidate Kinetics clips while tracking progress.

Consumes the source annotation CSV, writes class-organized MP4 files, and
records successful and failed candidates in ``progress.csv`` for resumability.
"""

import os
import re
import subprocess
import time
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CSV_FILE = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "K700",
    "metadata",
    "train.csv"
)

VIDEO_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "K700",
    "videos"
)

PROGRESS_FILE = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "K700",
    "progress.csv"
)

VIDEOS_PER_CLASS = 7

# Maximum number of candidates to try for one class
MAX_ATTEMPTS_PER_CLASS = 100


# ============================================================
# SETUP
# ============================================================

os.makedirs(VIDEO_ROOT, exist_ok=True)

print("=" * 70)
print("Kinetics-700 Downloader")
print("=" * 70)

print(f"CSV      : {CSV_FILE}")
print(f"Videos   : {VIDEO_ROOT}")
print(f"Progress : {PROGRESS_FILE}")
print(f"Target   : {VIDEOS_PER_CLASS} videos per class")

print("=" * 70)


# ============================================================
# CHECK CSV
# ============================================================

if not os.path.exists(CSV_FILE):

    raise FileNotFoundError(
        f"\nCSV file not found:\n{CSV_FILE}\n\n"
        "Download train.csv first."
    )


df = pd.read_csv(CSV_FILE)

required_columns = [
    "label",
    "youtube_id",
    "time_start",
    "time_end"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing column: {column}\n"
            f"Available columns: {list(df.columns)}"
        )


# Remove incomplete rows
df = df.dropna(
    subset=required_columns
)

# Remove duplicates
df = df.drop_duplicates(
    subset=[
        "label",
        "youtube_id",
        "time_start",
        "time_end"
    ]
)


# ============================================================
# PROGRESS FILE
# ============================================================

if os.path.exists(PROGRESS_FILE):

    progress = pd.read_csv(
        PROGRESS_FILE
    )

    print(
        f"\nExisting progress found: "
        f"{len(progress)} records"
    )

else:

    progress = pd.DataFrame(
        columns=[
            "label",
            "youtube_id",
            "time_start",
            "time_end",
            "status",
            "filename"
        ]
    )

    progress.to_csv(
        PROGRESS_FILE,
        index=False
    )

    print(
        "\nNo previous progress file found."
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_class_name(name):
    """Convert a class label into a filesystem-safe directory name."""

    name = str(name)

    name = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        name
    )

    return name.strip("_")


def save_progress(
    label,
    youtube_id,
    start,
    end,
    status,
    filename=""
):
    """Replace one candidate's progress record and persist the updated table."""

    global progress

    # Remove old record for this exact candidate
    if not progress.empty:

        mask = ~(
            (progress["label"] == label)
            &
            (progress["youtube_id"] == youtube_id)
            &
            (progress["time_start"] == start)
            &
            (progress["time_end"] == end)
        )

        progress = progress[mask]

    new_row = pd.DataFrame([
        {
            "label": label,
            "youtube_id": youtube_id,
            "time_start": start,
            "time_end": end,
            "status": status,
            "filename": filename
        }
    ])

    progress = pd.concat(
        [progress, new_row],
        ignore_index=True
    )

    progress.to_csv(
        PROGRESS_FILE,
        index=False
    )


def successful_count(label):
    """Count candidates already marked successful for a class."""

    if progress.empty:
        return 0

    rows = progress[
        (progress["label"] == label)
        &
        (progress["status"] == "success")
    ]

    return len(rows)


def already_successful(
    label,
    youtube_id,
    start,
    end
):
    """Check whether this exact clip candidate has already succeeded."""

    if progress.empty:
        return False

    rows = progress[
        (progress["label"] == label)
        &
        (progress["youtube_id"] == youtube_id)
        &
        (progress["time_start"] == start)
        &
        (progress["time_end"] == end)
        &
        (progress["status"] == "success")
    ]

    return len(rows) > 0


def check_audio_video(filepath):
    """Return whether FFprobe finds both video and audio streams."""

    if not os.path.exists(filepath):
        return False

    # --------------------------------------------------------
    # Check video
    # --------------------------------------------------------

    video_command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "csv=p=0",
        filepath
    ]

    # --------------------------------------------------------
    # Check audio
    # --------------------------------------------------------

    audio_command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "csv=p=0",
        filepath
    ]

    video_result = subprocess.run(
        video_command,
        capture_output=True,
        text=True
    )

    audio_result = subprocess.run(
        audio_command,
        capture_output=True,
        text=True
    )

    has_video = bool(
        video_result.stdout.strip()
    )

    has_audio = bool(
        audio_result.stdout.strip()
    )

    print(
        f"Video stream: "
        f"{'YES' if has_video else 'NO'}"
    )

    print(
        f"Audio stream: "
        f"{'YES' if has_audio else 'NO'}"
    )

    return has_video and has_audio


def find_mp4(
    directory,
    prefix
):
    """Find a previously downloaded MP4 with the expected candidate prefix."""

    if not os.path.exists(directory):
        return None

    for filename in os.listdir(directory):

        if (
            filename.startswith(prefix)
            and filename.endswith(".mp4")
        ):

            return os.path.join(
                directory,
                filename
            )

    return None


# ============================================================
# CLASSES
# ============================================================

classes = sorted(
    df["label"].unique()
)

print()
print(
    f"Total classes: {len(classes)}"
)

print(
    f"Target videos: "
    f"{len(classes) * VIDEOS_PER_CLASS}"
)

print()


# ============================================================
# MAIN DOWNLOAD LOOP
# ============================================================

for class_number, label in enumerate(
    classes,
    start=1
):

    print()
    print("#" * 70)

    print(
        f"CLASS {class_number}/{len(classes)}"
    )

    print(
        f"Class: {label}"
    )

    print("#" * 70)

    # --------------------------------------------------------
    # Count successful videos already downloaded
    # --------------------------------------------------------

    current_count = successful_count(
        label
    )

    print(
        f"Existing successful videos: "
        f"{current_count}/{VIDEOS_PER_CLASS}"
    )

    # --------------------------------------------------------
    # Skip completed class
    # --------------------------------------------------------

    if current_count >= VIDEOS_PER_CLASS:

        print(
            "Class already complete. Skipping."
        )

        continue

    # --------------------------------------------------------
    # Candidates for this class
    # --------------------------------------------------------

    class_df = df[
        df["label"] == label
    ].copy()

    # Shuffle candidates
    class_df = class_df.sample(
        frac=1,
        random_state=class_number
    ).reset_index(
        drop=True
    )

    attempts = 0

    # --------------------------------------------------------
    # Try candidates
    # --------------------------------------------------------

    for _, row in class_df.iterrows():

        current_count = successful_count(
            label
        )

        if current_count >= VIDEOS_PER_CLASS:

            break

        if attempts >= MAX_ATTEMPTS_PER_CLASS:

            print()
            print(
                f"Maximum attempts reached: "
                f"{MAX_ATTEMPTS_PER_CLASS}"
            )

            break

        youtube_id = str(
            row["youtube_id"]
        ).strip()

        start = int(
            float(row["time_start"])
        )

        end = int(
            float(row["time_end"])
        )

        # ----------------------------------------------------
        # Skip candidate already successful
        # ----------------------------------------------------

        if already_successful(
            label,
            youtube_id,
            start,
            end
        ):

            continue

        attempts += 1

        # ----------------------------------------------------
        # Directory
        # ----------------------------------------------------

        safe_label = safe_class_name(
            label
        )

        class_directory = os.path.join(
            VIDEO_ROOT,
            safe_label
        )

        os.makedirs(
            class_directory,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Filename
        # ----------------------------------------------------

        prefix = (
            f"{safe_label}_"
            f"{youtube_id}_"
            f"{start:06d}_"
            f"{end:06d}"
        )

        output_template = os.path.join(
            class_directory,
            prefix + ".%(ext)s"
        )

        # ----------------------------------------------------
        # Check existing file
        # ----------------------------------------------------

        existing_file = find_mp4(
            class_directory,
            prefix
        )

        if existing_file:

            print()
            print(
                "Existing file found:"
            )

            print(
                existing_file
            )

            print(
                "Checking file..."
            )

            if check_audio_video(
                existing_file
            ):

                print(
                    "Existing file is valid."
                )

                save_progress(
                    label,
                    youtube_id,
                    start,
                    end,
                    "success",
                    existing_file
                )

                continue

            else:

                print(
                    "Existing file is invalid."
                )

                try:

                    os.remove(
                        existing_file
                    )

                except OSError:

                    pass

        # ----------------------------------------------------
        # Display information
        # ----------------------------------------------------

        print()
        print("-" * 70)

        print(
            f"Class       : {label}"
        )

        print(
            f"Progress    : "
            f"{current_count}/{VIDEOS_PER_CLASS}"
        )

        print(
            f"Attempt     : "
            f"{attempts}/{MAX_ATTEMPTS_PER_CLASS}"
        )

        print(
            f"YouTube ID  : {youtube_id}"
        )

        print(
            f"Time        : {start} - {end}"
        )

        print("-" * 70)

        # ----------------------------------------------------
        # URL
        # ----------------------------------------------------

        url = (
            "https://www.youtube.com/watch?v="
            + youtube_id
        )

        # ----------------------------------------------------
        # yt-dlp command
        # ----------------------------------------------------

        command = [
            "yt-dlp",

            "--no-playlist",

            # Prefer MP4 video + M4A audio
            "-f",
            "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",

            "--merge-output-format",
            "mp4",

            # Download only Kinetics segment
            "--download-sections",
            f"*{start}-{end}",

            # Retry
            "--retries",
            "3",

            "--fragment-retries",
            "3",

            "--file-access-retries",
            "3",

            # Continue partial files
            "--continue",

            # Output
            "-o",
            output_template,

            url
        ]

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        try:

            result = subprocess.run(
                command
            )

        except Exception as error:

            print()
            print(
                "ERROR running yt-dlp:"
            )

            print(error)

            save_progress(
                label,
                youtube_id,
                start,
                end,
                "failed",
                ""
            )

            continue

        # ----------------------------------------------------
        # Failed
        # ----------------------------------------------------

        if result.returncode != 0:

            print()
            print(
                "DOWNLOAD FAILED"
            )

            print(
                "Trying next candidate..."
            )

            save_progress(
                label,
                youtube_id,
                start,
                end,
                "failed",
                ""
            )

            continue

        # ----------------------------------------------------
        # Find MP4
        # ----------------------------------------------------

        downloaded_file = find_mp4(
            class_directory,
            prefix
        )

        if downloaded_file is None:

            print(
                "MP4 file was not found."
            )

            save_progress(
                label,
                youtube_id,
                start,
                end,
                "failed",
                ""
            )

            continue

        # ----------------------------------------------------
        # Verify audio/video
        # ----------------------------------------------------

        print()
        print(
            "Verifying audio and video..."
        )

        valid = check_audio_video(
            downloaded_file
        )

        # ----------------------------------------------------
        # Invalid file
        # ----------------------------------------------------

        if not valid:

            print()
            print(
                "INVALID FILE"
            )

            print(
                "Missing audio or video."
            )

            print(
                "Deleting and trying next candidate..."
            )

            try:

                os.remove(
                    downloaded_file
                )

            except OSError:

                pass

            save_progress(
                label,
                youtube_id,
                start,
                end,
                "failed_missing_audio_or_video",
                ""
            )

            continue

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        current_count += 1

        print()
        print("=" * 70)

        print(
            "SUCCESS"
        )

        print(
            f"Class: {label}"
        )

        print(
            f"Progress: "
            f"{current_count}/{VIDEOS_PER_CLASS}"
        )

        print(
            f"File: {downloaded_file}"
        )

        print("=" * 70)

        save_progress(
            label,
            youtube_id,
            start,
            end,
            "success",
            downloaded_file
        )

        time.sleep(1)

    # --------------------------------------------------------
    # CLASS SUMMARY
    # --------------------------------------------------------

    final_count = successful_count(
        label
    )

    print()
    print("=" * 70)

    if final_count >= VIDEOS_PER_CLASS:

        print(
            f"COMPLETED: {label}"
        )

        print(
            f"{final_count}/{VIDEOS_PER_CLASS}"
        )

    else:

        print(
            f"INCOMPLETE: {label}"
        )

        print(
            f"{final_count}/{VIDEOS_PER_CLASS}"
        )

        print(
            "You can run the script again later."
        )

    print("=" * 70)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("#" * 70)
print("FINAL DATASET SUMMARY")
print("#" * 70)

if os.path.exists(PROGRESS_FILE):

    final_progress = pd.read_csv(
        PROGRESS_FILE
    )

    successful = final_progress[
        final_progress["status"] == "success"
    ]

    failed = final_progress[
        final_progress["status"] != "success"
    ]

    completed_classes = 0

    for label in classes:

        count = len(
            successful[
                successful["label"] == label
            ]
        )

        if count >= VIDEOS_PER_CLASS:

            completed_classes += 1

    print(
        f"Successful videos : {len(successful)}"
    )

    print(
        f"Failed attempts   : {len(failed)}"
    )

    print(
        f"Completed classes : "
        f"{completed_classes}/{len(classes)}"
    )

    print()
    print(
        f"Progress file:"
    )

    print(
        PROGRESS_FILE
    )

print()
print(
    "Download process finished."
)