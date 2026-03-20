"""Unit tests for training.checkpointing module."""

import json
from pathlib import Path

from src.training.checkpointing import CheckpointManager


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_list_empty(self, tmp_path: Path) -> None:
        """Test listing when no checkpoints exist."""
        mgr = CheckpointManager(tmp_path)
        assert mgr.list_checkpoints() == []

    def test_list_sorted(self, tmp_path: Path) -> None:
        """Test that checkpoints are listed in order."""
        (tmp_path / "checkpoint-200").mkdir()
        (tmp_path / "checkpoint-100").mkdir()
        (tmp_path / "checkpoint-300").mkdir()
        mgr = CheckpointManager(tmp_path)
        checkpoints = mgr.list_checkpoints()
        assert [p.name for p in checkpoints] == [
            "checkpoint-100",
            "checkpoint-200",
            "checkpoint-300",
        ]

    def test_cleanup_old(self, tmp_path: Path) -> None:
        """Test that old checkpoints are removed."""
        for i in range(5):
            (tmp_path / f"checkpoint-{i * 100}").mkdir()
        mgr = CheckpointManager(tmp_path, max_checkpoints=2)
        removed = mgr.cleanup_old()
        assert len(removed) == 3
        assert len(mgr.list_checkpoints()) == 2

    def test_get_latest(self, tmp_path: Path) -> None:
        """Test getting the latest checkpoint."""
        (tmp_path / "checkpoint-100").mkdir()
        (tmp_path / "checkpoint-500").mkdir()
        mgr = CheckpointManager(tmp_path)
        assert mgr.get_latest() is not None
        assert mgr.get_latest().name == "checkpoint-500"  # type: ignore[union-attr]

    def test_get_latest_empty(self, tmp_path: Path) -> None:
        """Test getting latest when no checkpoints exist."""
        mgr = CheckpointManager(tmp_path)
        assert mgr.get_latest() is None
