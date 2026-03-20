"""Checkpoint management: saving, loading, and cleanup of training checkpoints."""
from __future__ import annotations

import shutil
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


class CheckpointManager:
    """Manage training checkpoints with rotation and cleanup."""

    def __init__(self, checkpoint_dir: Path, max_checkpoints: int = 3) -> None:
        """Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Base directory for checkpoints.
            max_checkpoints: Maximum number of checkpoints to retain.
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def list_checkpoints(self) -> list[Path]:
        """List existing checkpoints sorted by step number.

        Returns:
            List of checkpoint directory paths sorted ascending by step.
        """
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        return checkpoints

    def cleanup_old(self) -> list[Path]:
        """Remove checkpoints exceeding the maximum retention count.

        Returns:
            List of removed checkpoint paths.
        """
        checkpoints = self.list_checkpoints()
        removed: list[Path] = []
        while len(checkpoints) > self.max_checkpoints:
            oldest = checkpoints.pop(0)
            logger.info("Removing old checkpoint", path=str(oldest))
            shutil.rmtree(oldest)
            removed.append(oldest)
        return removed

    def get_latest(self) -> Path | None:
        """Get the path to the most recent checkpoint.

        Returns:
            Path to the latest checkpoint, or None if none exist.
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        return checkpoints[-1]

    def get_best(self, metric_file: str = "trainer_state.json") -> Path | None:
        """Get the checkpoint with the best evaluation metric.

        Args:
            metric_file: Name of the trainer state file to inspect.

        Returns:
            Path to the best checkpoint, or None if none exist.
        """
        import json

        best_path: Path | None = None
        best_loss = float("inf")

        for checkpoint in self.list_checkpoints():
            state_file = checkpoint / metric_file
            if state_file.exists():
                with open(state_file) as f:
                    state = json.load(f)
                loss = state.get("best_metric", float("inf"))
                if loss < best_loss:
                    best_loss = loss
                    best_path = checkpoint

        return best_path


__all__: list[str] = ["CheckpointManager"]
