import json
from pathlib import Path

from src.adapter import independent_crop_adapter as adapter_module
from src.adapter.independent_crop_adapter import IndependentCropAdapter


class FakeTrainer:
    def __init__(self, config):
        self.config = config
        self.class_to_idx = {"healthy": 0}
        self.target_modules_resolved = []
        self.ood_detector = type("OOD", (), {"calibration_version": 0})()

    def initialize_engine(self, class_to_idx=None):
        self.class_to_idx = dict(class_to_idx or {})

    def save_training_checkpoint(self, output_dir, progress_state=None, history=None, run_id=""):
        root = Path(output_dir) / "training_checkpoint"
        root.mkdir(parents=True, exist_ok=True)
        payload = {
            "class_to_idx": dict(self.class_to_idx),
            "progress_state": dict(progress_state or {}),
            "history_snapshot": dict(history or {}),
            "run_id": run_id,
        }
        (root / "checkpoint_meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (root / "training_checkpoint.pt").write_bytes(b"checkpoint")
        return root

    def load_training_checkpoint(self, checkpoint_dir):
        root = Path(checkpoint_dir)
        if (root / "training_checkpoint").exists():
            root = root / "training_checkpoint"
        payload = json.loads((root / "checkpoint_meta.json").read_text(encoding="utf-8"))
        self.class_to_idx = dict(payload.get("class_to_idx", {}))
        return payload


def test_adapter_checkpoint_save_load(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter_module, "ContinualSDLoRATrainer", FakeTrainer)

    adapter = IndependentCropAdapter(crop_name="tomato", device="cpu")
    adapter.initialize_engine(class_names=["healthy"])

    saved = adapter.save_training_checkpoint(
        str(tmp_path / "run"),
        progress_state={"epoch": 2, "global_step": 42},
        history={"train_loss": [0.3, 0.2]},
        run_id="run_123",
    )
    assert (saved / "checkpoint_meta.json").exists()

    loaded = adapter.load_training_checkpoint(str(saved))
    assert loaded["run_id"] == "run_123"
    assert loaded["progress_state"]["global_step"] == 42

