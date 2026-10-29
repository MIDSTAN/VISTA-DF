"""Convert train, validation, and holdout video metadata into temporal windows.

Writes fixed-duration start/end records to ``windows.csv`` for the dataset
loader and media validation tools.
"""

import pandas as pd
from pathlib import Path
import math


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    "/home/midstan/Documents/Python/VISTA-DF"
)

METADATA_DIR = (
    BASE_DIR
    / "dataset/K700/metadata"
)

TRAIN_CSV = METADATA_DIR / "train.csv"
VAL_CSV = METADATA_DIR / "val.csv"
HOLDOUT_CSV = METADATA_DIR / "holdout.csv"

OUTPUT_CSV = METADATA_DIR / "windows.csv"


# Temporal window configuration
WINDOW_SIZE = 2.0       # seconds
STRIDE = 2.0            # seconds


# ============================================================
# FUNCTION TO CREATE WINDOWS
# ============================================================

def create_windows(csv_path, split_name):
    """Build complete, non-overlapping time windows for one split CSV."""

    print("\n" + "=" * 70)
    print(f"PROCESSING {split_name.upper()}")
    print("=" * 70)

    df = pd.read_csv(csv_path)

    print(f"Videos in {split_name}: {len(df)}")

    windows = []

    skipped_short = 0
    skipped_invalid = 0

    total_windows = 0


    # --------------------------------------------------------
    # Process every video
    # --------------------------------------------------------

    for _, row in df.iterrows():

        video_id = row["video_id"]
        class_name = row["class"]
        video_path = row["path"]
        duration = float(row["duration_sec"])


        # ----------------------------------------------------
        # Check duration
        # ----------------------------------------------------

        if duration <= 0:

            skipped_invalid += 1
            continue


        # ----------------------------------------------------
        # Number of complete windows
        # ----------------------------------------------------

        if duration < WINDOW_SIZE:

            skipped_short += 1
            continue


        number_of_windows = int(
            math.floor(
                (duration - WINDOW_SIZE) / STRIDE
            ) + 1
        )


        # ----------------------------------------------------
        # Create windows
        # ----------------------------------------------------

        for window_id in range(number_of_windows):

            start_sec = window_id * STRIDE

            end_sec = start_sec + WINDOW_SIZE


            # Safety check
            if end_sec > duration:
                break


            windows.append({
                "video_id": video_id,
                "class": class_name,
                "split": split_name,
                "window_id": window_id,
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "window_duration_sec": WINDOW_SIZE,
                "path": video_path
            })


            total_windows += 1


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(f"Windows generated : {total_windows}")

    print(
        f"Videos skipped "
        f"(shorter than {WINDOW_SIZE}s): "
        f"{skipped_short}"
    )

    print(
        f"Videos skipped "
        f"(invalid duration): "
        f"{skipped_invalid}"
    )


    return windows


# ============================================================
# CREATE WINDOWS FOR ALL SPLITS
# ============================================================

all_windows = []


# TRAIN
train_windows = create_windows(
    TRAIN_CSV,
    "train"
)

all_windows.extend(train_windows)


# VALIDATION
val_windows = create_windows(
    VAL_CSV,
    "val"
)

all_windows.extend(val_windows)


# HOLDOUT
holdout_windows = create_windows(
    HOLDOUT_CSV,
    "holdout"
)

all_windows.extend(holdout_windows)


# ============================================================
# CREATE DATAFRAME
# ============================================================

windows_df = pd.DataFrame(
    all_windows
)


# ============================================================
# SORT DATA
# ============================================================

if not windows_df.empty:

    windows_df = windows_df.sort_values(
        by=[
            "split",
            "video_id",
            "window_id"
        ]
    ).reset_index(drop=True)


# ============================================================
# SAVE CSV
# ============================================================

windows_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("WINDOW GENERATION COMPLETE")
print("=" * 70)

print(
    f"\nTotal windows: "
    f"{len(windows_df)}"
)


if not windows_df.empty:

    print("\nWindows by split:")

    split_counts = (
        windows_df["split"]
        .value_counts()
    )

    for split, count in split_counts.items():

        print(
            f"{split:12} : {count}"
        )


    print("\nVideos represented in each split:")

    video_counts = (
        windows_df
        .groupby("split")["video_id"]
        .nunique()
    )

    for split, count in video_counts.items():

        print(
            f"{split:12} : {count}"
        )


    print("\nAverage windows per video:")

    average_windows = (
        windows_df
        .groupby("split")
        .size()
        /
        windows_df
        .groupby("split")["video_id"]
        .nunique()
    )

    for split, value in average_windows.items():

        print(
            f"{split:12} : {value:.2f}"
        )


# ============================================================
# FILE INFORMATION
# ============================================================

print("\n")
print("=" * 70)
print("OUTPUT")
print("=" * 70)

print("\nwindows.csv saved to:")

print(OUTPUT_CSV)

print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)
