"""Create reproducible, class-stratified train, validation, and holdout CSVs.

Reads video metadata, excludes records not marked valid, and checks that each
video appears in only one output split.
"""

import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    "/home/midstan/Documents/Python/VISTA-DF"
)

INPUT_CSV = (
    BASE_DIR
    / "dataset/K700/metadata/videos.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "dataset/K700/metadata"
)

TRAIN_CSV = OUTPUT_DIR / "train.csv"
VAL_CSV = OUTPUT_DIR / "val.csv"
HOLDOUT_CSV = OUTPUT_DIR / "holdout.csv"


# Fixed random seed
# This guarantees the same split every time the script runs.
RANDOM_SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("VISTA-DF DATASET SPLITTING")
print("=" * 70)

print(f"\nReading:")
print(INPUT_CSV)

df = pd.read_csv(INPUT_CSV)

print(f"\nTotal videos: {len(df)}")


# ============================================================
# CHECK REQUIRED COLUMN
# ============================================================

if "class" not in df.columns:

    raise ValueError(
        "ERROR: 'class' column not found in videos.csv"
    )


# ============================================================
# REMOVE INVALID RECORDS
# ============================================================

if "status" in df.columns:

    invalid = df[
        df["status"].astype(str) != "OK"
    ]

    if len(invalid) > 0:

        print(
            f"\nWARNING: "
            f"{len(invalid)} invalid videos found."
        )

        print(
            "These videos will NOT be included "
            "in the splits."
        )

        df = df[
            df["status"].astype(str) == "OK"
        ]


# ============================================================
# SHUFFLE EACH CLASS AND SPLIT
# ============================================================

train_parts = []
val_parts = []
holdout_parts = []


for class_name, class_df in df.groupby(
    "class",
    sort=True
):

    # Shuffle videos inside this class
    class_df = class_df.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)


    number_of_videos = len(class_df)


    # --------------------------------------------------------
    # Need at least 3 videos
    # --------------------------------------------------------

    if number_of_videos < 3:

        raise ValueError(
            f"Class '{class_name}' has only "
            f"{number_of_videos} videos. "
            f"At least 3 are required."
        )


    # --------------------------------------------------------
    # Always reserve:
    #
    # 1 video → validation
    # 1 video → holdout
    #
    # Everything else → training
    # --------------------------------------------------------

    train_class = class_df.iloc[:-2]

    val_class = class_df.iloc[-2:-1]

    holdout_class = class_df.iloc[-1:]


    train_parts.append(train_class)

    val_parts.append(val_class)

    holdout_parts.append(holdout_class)


# ============================================================
# COMBINE ALL CLASSES
# ============================================================

train_df = pd.concat(
    train_parts,
    ignore_index=True
)

val_df = pd.concat(
    val_parts,
    ignore_index=True
)

holdout_df = pd.concat(
    holdout_parts,
    ignore_index=True
)


# ============================================================
# SHUFFLE FINAL DATASETS
# ============================================================

train_df = train_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

val_df = val_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

holdout_df = holdout_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)


# ============================================================
# SAVE CSV FILES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

train_df.to_csv(
    TRAIN_CSV,
    index=False
)

val_df.to_csv(
    VAL_CSV,
    index=False
)

holdout_df.to_csv(
    HOLDOUT_CSV,
    index=False
)


# ============================================================
# VERIFY SPLIT
# ============================================================

print("\n")
print("=" * 70)
print("SPLIT RESULTS")
print("=" * 70)

print(
    f"\nTraining videos   : {len(train_df)}"
)

print(
    f"Validation videos : {len(val_df)}"
)

print(
    f"Holdout videos    : {len(holdout_df)}"
)

print(
    f"Total              : "
    f"{len(train_df) + len(val_df) + len(holdout_df)}"
)


# ============================================================
# CLASS DISTRIBUTION CHECK
# ============================================================

print("\n")
print("=" * 70)
print("CLASS DISTRIBUTION CHECK")
print("=" * 70)

train_counts = train_df["class"].value_counts()

val_counts = val_df["class"].value_counts()

holdout_counts = holdout_df["class"].value_counts()


print(
    f"\nClasses in training   : "
    f"{train_df['class'].nunique()}"
)

print(
    f"Classes in validation : "
    f"{val_df['class'].nunique()}"
)

print(
    f"Classes in holdout    : "
    f"{holdout_df['class'].nunique()}"
)


# ============================================================
# VERIFY NO VIDEO OVERLAP
# ============================================================

train_ids = set(
    train_df["video_id"]
)

val_ids = set(
    val_df["video_id"]
)

holdout_ids = set(
    holdout_df["video_id"]
)


train_val_overlap = train_ids & val_ids

train_holdout_overlap = (
    train_ids & holdout_ids
)

val_holdout_overlap = (
    val_ids & holdout_ids
)


print("\n")
print("=" * 70)
print("VIDEO OVERLAP CHECK")
print("=" * 70)

print(
    f"\nTrain ↔ Validation overlap : "
    f"{len(train_val_overlap)}"
)

print(
    f"Train ↔ Holdout overlap    : "
    f"{len(train_holdout_overlap)}"
)

print(
    f"Validation ↔ Holdout       : "
    f"{len(val_holdout_overlap)}"
)


# ============================================================
# FINAL VALIDATION
# ============================================================

total_split = (
    len(train_df)
    + len(val_df)
    + len(holdout_df)
)

if total_split != len(df):

    print("\nWARNING:")
    print(
        "Split total does not equal "
        "the number of valid videos."
    )

else:

    print("\nAll valid videos are accounted for.")


if (
    len(train_val_overlap) == 0
    and
    len(train_holdout_overlap) == 0
    and
    len(val_holdout_overlap) == 0
):

    print(
        "No video appears in more than one split."
    )

else:

    raise RuntimeError(
        "ERROR: Video overlap detected!"
    )


# ============================================================
# FILE LOCATIONS
# ============================================================

print("\n")
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print(f"\nTrain:")
print(TRAIN_CSV)

print(f"\nValidation:")
print(VAL_CSV)

print(f"\nHoldout:")
print(HOLDOUT_CSV)

print("\n")
print("=" * 70)
print("DATASET SPLITTING COMPLETE")
print("=" * 70)
