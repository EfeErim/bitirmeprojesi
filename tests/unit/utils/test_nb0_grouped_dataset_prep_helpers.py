from pathlib import Path

from scripts.notebook_helpers import nb0_grouped_dataset_prep_helpers as nb0


class _FakeTelemetry:
    def __init__(self) -> None:
        self.latest_payloads = []
        self.summary_payloads = []
        self.closed_payloads = []

    def update_latest(self, payload):
        self.latest_payloads.append(dict(payload))

    def merge_summary_metadata(self, payload):
        self.summary_payloads.append(dict(payload))

    def close(self, payload):
        self.closed_payloads.append(dict(payload))


def test_runtime_materialization_uses_copy_for_portable_prepared_dataset(tmp_path: Path, monkeypatch):
    calls = []

    def _fake_materialize_grouped_runtime_dataset(**kwargs):
        calls.append(dict(kwargs))
        return Path(kwargs["runtime_root"])

    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset.materialize_grouped_runtime_dataset",
        _fake_materialize_grouped_runtime_dataset,
    )
    monkeypatch.setattr(
        nb0,
        "export_current_colab_notebook",
        lambda target: str(target),
    )

    state = {
        "validated": True,
        "audit_summary": {"runtime_ready": True},
        "dataset_root": tmp_path / "prepared_class_root" / "tomato__leaf",
        "artifact_root": tmp_path / "artifacts",
    }
    telemetry = _FakeTelemetry()

    nb0.run_materialize_runtime_dataset(
        ROOT=tmp_path,
        STATE=state,
        TELEMETRY=telemetry,
        CROP_NAME="tomato",
        PART_NAME="leaf",
        OOD_ROOT="",
        OOD_DATASET_NAME="",
        OOD_DATASET_ROOT="data/ood_dataset",
        ASK_FOR_OOD_ROOT=False,
        PREPARED_RUNTIME_ROOT="data/prepared_runtime_datasets",
        MATERIALIZE_AFTER_REVIEW=True,
        SAVE_RUNTIME_DATASET_TO_GITHUB=False,
        RUNTIME_DATASET_PUSH_REMOTE_NAME="origin",
        RUNTIME_DATASET_PUSH_BRANCH="master",
        REPO_NOTEBOOK_OUTPUT_PATH=tmp_path / "runs" / "run_1" / "notebooks" / "executed.ipynb",
        REPO_RUN_DIR=tmp_path / "runs" / "run_1",
        REPO_RUN_EXPORTS={},
    )

    assert calls
    assert calls[0]["materialization_strategy"] == "copy"
    assert state["runtime_dataset_root"] == tmp_path / "data" / "prepared_runtime_datasets"
    assert telemetry.closed_payloads[-1]["materialized"] is True


def test_fix_gitignore_handles_legacy_turkish_encoding(tmp_path: Path):
    gitignore = tmp_path / ".gitignore"
    gitignore.write_bytes(
        b"# legacy cp1254 comment: \xc7ilek\n"
        b"data/prepared_runtime_datasets/*\n"
        b"!data/prepared_runtime_datasets/.gitkeep\n"
    )

    nb0.fix_gitignore(tmp_path)

    text = gitignore.read_text(encoding="cp1254")
    assert "# legacy cp1254 comment: Cilek" not in text
    assert "# legacy cp1254 comment: \u00c7ilek" in text
    assert "!data/prepared_runtime_datasets/*/" in text
    assert "!data/prepared_runtime_datasets/**/*" in text


def test_nb0_cell09_gitignore_fix_handles_legacy_turkish_encoding(tmp_path: Path):
    gitignore = tmp_path / ".gitignore"
    gitignore.write_bytes(
        b"# legacy cp1254 comment: \xc7ilek\n"
        b"data/prepared_runtime_datasets/*\n"
        b"!data/prepared_runtime_datasets/.gitkeep\n"
    )
    script_path = Path("scripts/notebook_cells/nb0_cell09_gitignore_fix.py")
    namespace = {"ROOT": tmp_path}

    exec(compile(script_path.read_text(encoding="utf-8"), str(script_path), "exec"), namespace)

    text = gitignore.read_text(encoding="cp1254")
    assert "# legacy cp1254 comment: \u00c7ilek" in text
    assert "!data/prepared_runtime_datasets/*/" in text
    assert "!data/prepared_runtime_datasets/**/*" in text
