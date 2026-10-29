"""Read RUN/PAUSE/STOP commands from a file between training batches.

The trainer uses this helper to pause or stop only at safe boundaries and save
a checkpoint before waiting or exiting.
"""

import time
from pathlib import Path


class TrainingController:
    """Provide cooperative file-based control for a running trainer."""

    def __init__(self, command_file: str | Path, poll_seconds: float = 3.0):
        """Set the command-file location and pause polling interval."""
        self.command_file = Path(command_file)
        self.poll_seconds = poll_seconds

    def command(self) -> str:
        """Return a valid command, defaulting to RUN if the file is unavailable."""
        if not self.command_file.exists():
            return "RUN"
        try:
            value = self.command_file.read_text(encoding="utf-8").strip().upper()
        except OSError:
            return "RUN"
        return value if value in {"RUN", "PAUSE", "STOP"} else "RUN"

    def batch_boundary(self, save_checkpoint) -> bool:
        """Save and wait on PAUSE; save and return False on STOP."""
        command = self.command()
        if command == "STOP":
            save_checkpoint()
            print("STOP requested: checkpoint saved; exiting cleanly.", flush=True)
            return False
        if command == "PAUSE":
            save_checkpoint()
            print("PAUSE requested: checkpoint saved. Write RUN to resume.", flush=True)
            while self.command() == "PAUSE":
                time.sleep(self.poll_seconds)
            if self.command() == "STOP":
                save_checkpoint()
                print("STOP requested while paused: exiting cleanly.", flush=True)
                return False
            print("RUN received: continuing with the next batch.", flush=True)
        return True

