"""Run a standalone FFmpeg/FFprobe sanity check on sampled CSV windows.

Checks representative source clips and extracted media; this script is not a
pytest test and does not exercise the production dataset loader.
"""

import pandas as pd
import subprocess
import json
from pathlib import Path
import tempfile
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    "/home/midstan/Documents/Python/VISTA-DF"
)

VIDEOS_DIR = (
    BASE_DIR
    / "dataset/K700/videos"
)

WINDOWS_CSV = (
    BASE_DIR
    / "dataset/K700/metadata/windows.csv"
)

# Number of windows to test
NUM_WINDOWS_TO_TEST = 10

# Expected temporal window
EXPECTED_DURATION = 2.0

# Number of video frames we want to sample
# This is ONLY for testing the loader.
NUM_TEST_FRAMES = 16


# ============================================================
# CHECK FILES
# ============================================================

if not WINDOWS_CSV.exists():

    raise FileNotFoundError(
        f"windows.csv not found:\n{WINDOWS_CSV}"
    )


if not VIDEOS_DIR.exists():

    raise FileNotFoundError(
        f"Videos directory not found:\n{VIDEOS_DIR}"
    )


# ============================================================
# LOAD WINDOWS
# ============================================================

windows_df = pd.read_csv(WINDOWS_CSV)

print("=" * 80)
print("VISTA-DF TEMPORAL WINDOW VALIDATION")
print("=" * 80)

print(
    f"\nTotal windows in CSV: "
    f"{len(windows_df)}"
)


# ============================================================
# SELECT WINDOWS
# ============================================================

# Take windows from different videos when possible
test_df = (
    windows_df
    .drop_duplicates(subset=["video_id"])
    .head(NUM_WINDOWS_TO_TEST)
)


print(
    f"Testing {len(test_df)} temporal windows."
)


# ============================================================
# FFMPEG/FFPROBE HELPER
# ============================================================

def run_command(command):
    """Run a media-tool command and return its captured result."""

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    return result


# ============================================================
# GET MEDIA INFORMATION
# ============================================================

def get_media_info(video_path):
    """Read container and stream metadata, or return ``None`` on bad output."""

    command = [
        "ffprobe",
        "-v",
        "error",

        "-show_entries",
        "format=duration",

        "-show_streams",

        "-of",
        "json",

        str(video_path)
    ]

    result = run_command(command)

    if result.returncode != 0:

        return None

    try:
        return json.loads(result.stdout)

    except json.JSONDecodeError:

        return None


# ============================================================
# TEST ONE WINDOW
# ============================================================

def test_window(row, test_number):
    """Check one CSV window's source and extracted audio/video content."""

    video_id = row["video_id"]

    relative_path = row["path"]

    start_sec = float(row["start_sec"])

    end_sec = float(row["end_sec"])

    duration = end_sec - start_sec


    print("\n" + "-" * 80)

    print(
        f"TEST {test_number}"
    )

    print(
        f"Video ID      : {video_id}"
    )

    print(
        f"Class         : {row['class']}"
    )

    print(
        f"Window        : "
        f"{start_sec:.3f} - {end_sec:.3f} sec"
    )

    print(
        f"Window length : "
        f"{duration:.3f} sec"
    )


    # --------------------------------------------------------
    # Locate video
    # --------------------------------------------------------

    video_path = VIDEOS_DIR / relative_path


    print(
        f"Video path    : {video_path}"
    )


    if not video_path.exists():

        print("RESULT: FAIL")
        print("Reason: video file does not exist.")

        return False


    # --------------------------------------------------------
    # Check media
    # --------------------------------------------------------

    media_info = get_media_info(video_path)

    if media_info is None:

        print("RESULT: FAIL")
        print("Reason: ffprobe could not read video.")

        return False


    streams = media_info.get(
        "streams",
        []
    )


    video_stream = None
    audio_stream = None


    for stream in streams:

        if stream.get("codec_type") == "video":

            if video_stream is None:
                video_stream = stream


        elif stream.get("codec_type") == "audio":

            if audio_stream is None:
                audio_stream = stream


    if video_stream is None:

        print("RESULT: FAIL")
        print("Reason: no video stream.")

        return False


    if audio_stream is None:

        print("RESULT: FAIL")
        print("Reason: no audio stream.")

        return False


    print("Video stream : FOUND")
    print("Audio stream : FOUND")


    # --------------------------------------------------------
    # Temporary directory
    # --------------------------------------------------------

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="vista_df_test_"
        )
    )


    try:

        # ====================================================
        # EXTRACT VIDEO FRAMES
        # ====================================================

        frames_dir = (
            temp_dir
            / "frames"
        )

        frames_dir.mkdir()


        frame_interval = (
            duration / NUM_TEST_FRAMES
        )


        frame_fps = (
            NUM_TEST_FRAMES / duration
        )


        video_output = (
            frames_dir
            / "frame_%02d.jpg"
        )


        video_command = [
            "ffmpeg",

            "-hide_banner",
            "-loglevel",
            "error",

            "-ss",
            str(start_sec),

            "-i",
            str(video_path),

            "-t",
            str(duration),

            "-vf",
            f"fps={frame_fps}",

            "-frames:v",
            str(NUM_TEST_FRAMES),

            "-q:v",
            "2",

            str(video_output)
        ]


        result = run_command(
            video_command
        )


        if result.returncode != 0:

            print("RESULT: FAIL")
            print(
                "Reason: video frame extraction failed."
            )

            print(result.stderr)

            return False


        extracted_frames = list(
            frames_dir.glob("*.jpg")
        )


        print(
            f"Video frames extracted: "
            f"{len(extracted_frames)}"
        )


        if len(extracted_frames) == 0:

            print("RESULT: FAIL")
            print(
                "Reason: zero frames extracted."
            )

            return False


        # ====================================================
        # EXTRACT AUDIO
        # ====================================================

        audio_output = (
            temp_dir
            / "audio.wav"
        )


        audio_command = [
            "ffmpeg",

            "-hide_banner",
            "-loglevel",
            "error",

            "-ss",
            str(start_sec),

            "-i",
            str(video_path),

            "-t",
            str(duration),

            "-vn",

            "-ac",
            "1",

            "-ar",
            "16000",

            "-c:a",
            "pcm_s16le",

            str(audio_output)
        ]


        result = run_command(
            audio_command
        )


        if result.returncode != 0:

            print("RESULT: FAIL")
            print(
                "Reason: audio extraction failed."
            )

            print(result.stderr)

            return False


        if not audio_output.exists():

            print("RESULT: FAIL")
            print(
                "Reason: audio file was not created."
            )

            return False


        # ====================================================
        # CHECK AUDIO DURATION
        # ====================================================

        audio_probe_command = [
            "ffprobe",

            "-v",
            "error",

            "-show_entries",
            "format=duration",

            "-of",
            "default=noprint_wrappers=1:"
            "nokey=1",

            str(audio_output)
        ]


        result = run_command(
            audio_probe_command
        )


        if result.returncode != 0:

            print("RESULT: FAIL")
            print(
                "Reason: could not inspect audio."
            )

            return False


        audio_duration = float(
            result.stdout.strip()
        )


        print(
            f"Audio duration       : "
            f"{audio_duration:.3f} sec"
        )


        # ====================================================
        # DURATION CHECK
        # ====================================================

        duration_difference = abs(
            audio_duration
            - EXPECTED_DURATION
        )


        if duration_difference > 0.15:

            print("WARNING:")
            print(
                "Audio duration differs "
                "from expected duration."
            )


        # ====================================================
        # SUCCESS
        # ====================================================

        print(
            "RESULT: PASS"
        )

        return True


    finally:

        # Remove temporary files
        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )


# ============================================================
# RUN TESTS
# ============================================================

passed = 0
failed = 0


for index, (_, row) in enumerate(
    test_df.iterrows(),
    start=1
):

    success = test_window(
        row,
        index
    )


    if success:

        passed += 1

    else:

        failed += 1


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 80)
print("TEMPORAL WINDOW VALIDATION REPORT")
print("=" * 80)

print(
    f"\nWindows tested : "
    f"{len(test_df)}"
)

print(
    f"Passed         : "
    f"{passed}"
)

print(
    f"Failed         : "
    f"{failed}"
)


if failed == 0:

    print("\nSTATUS: ALL TESTS PASSED")

    print(
        "\nVideo and audio temporal windows "
        "can be extracted successfully."
    )

else:

    print(
        "\nSTATUS: SOME TESTS FAILED"
    )

    print(
        "\nFix the data-loading issue "
        "before moving to model development."
    )


print("\n")
print("=" * 80)

