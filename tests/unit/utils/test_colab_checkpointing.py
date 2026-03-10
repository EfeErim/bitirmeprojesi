from pathlib import Path

from scripts.colab_checkpointing import TrainingCheckpointManager


class _FakeAdapter:
    def save_training_checkpoint(self, checkpoint_dir: str, *, session_state, run_id: str) -> Path:
        root = Path(checkpoint_dir) / "continual_sd_lora_adapter"
        root.mkdir(parents=True, exist_ok=True)
        (root / "checkpoint.pt").write_text("weights", encoding="utf-8")
        return root


class _FakeSession:
    def snapshot_state(self):
        return {
            "progress_state": {
                "epoch": 2,
                "global_step": 5,
            }
        }


def test_checkpoint_manager_marks_best_without_duplicate_best_tree(tmp_path: Path):
    manager = TrainingCheckpointManager(tmp_path / "telemetry" / "run_1", retention=2)

    record = manager.save_checkpoint(
        adapter=_FakeAdapter(),
        session=_FakeSession(),
        reason="epoch_end",
        run_id="run_1",
        mark_best=True,
        val_loss=0.2,
    )

    best = manager.get_best()

    assert isinstance(best, dict)
    assert best["path"] == record["path"]
    assert Path(record["path"]).exists()
    assert not (manager.checkpoints_dir / "best").exists()
