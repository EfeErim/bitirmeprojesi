import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scripts.prepare_grouped_runtime_dataset import (
    build_grouped_dataset_plan,
    materialize_grouped_runtime_dataset,
    normalize_prepared_class_name,
)


def _write_pattern(path: Path, *, offset: int) -> None:
    image = Image.new("RGB", (32, 32), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((offset, offset, offset + 7, offset + 7), fill=(255, 0, 0))
    draw.line((0, 31 - offset, 31, offset), fill=(0, 128, 0), width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _fake_embeddings(paths, *, model_id: str, batch_size: int, device: str):  # noqa: ARG001
    size = len(paths)
    if size == 0:
        return np.empty((0, 0), dtype=np.float32)
    return np.eye(size, dtype=np.float32)


class _FakeModel:
    def eval(self):
        return self

    def to(self, *args, **kwargs):  # noqa: ARG002
        return self


def test_normalize_prepared_class_name_uses_taxonomy_aliases():
    normalized = normalize_prepared_class_name(
        "Tomato Healthy Leaf",
        crop_name="tomato",
        expected_classes={"healthy", "early_blight"},
    )

    assert normalized == "healthy"


def test_build_grouped_dataset_plan_blocks_cross_class_exact_duplicate(tmp_path: Path, monkeypatch):
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifacts"
    image_path = source_root / "Healthy" / "shared.jpg"
    _write_pattern(image_path, offset=2)
    duplicate_target = source_root / "Late Blight" / "shared.jpg"
    duplicate_target.parent.mkdir(parents=True, exist_ok=True)
    duplicate_target.write_bytes(image_path.read_bytes())

    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_dinov3_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_bioclip_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_dinov3_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_bioclip_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )

    summary = build_grouped_dataset_plan(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        taxonomy_path=None,
    )

    assert summary["runtime_ready"] is False
    assert summary["summary"]["cross_class_conflicts"] == 1
    assert (artifact_root / "cross_class_conflicts.csv").exists()


def test_materialize_grouped_runtime_dataset_writes_runtime_layout(tmp_path: Path, monkeypatch):
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifacts"
    runtime_root = tmp_path / "runtime"

    for index, offset in enumerate((2, 10, 18)):
        _write_pattern(source_root / "Healthy" / f"healthy_{index}.jpg", offset=offset)
        _write_pattern(source_root / "Early Blight" / f"disease_{index}.jpg", offset=offset + 1)

    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_dinov3_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_bioclip_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_dinov3_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_bioclip_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )

    summary = build_grouped_dataset_plan(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        taxonomy_path=None,
    )

    assert summary["runtime_ready"] is True

    result_root = materialize_grouped_runtime_dataset(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        runtime_root=runtime_root,
    )

    crop_root = result_root / "tomato"
    assert (crop_root / "continual").exists()
    assert (crop_root / "val").exists()
    assert (crop_root / "test").exists()
    assert (crop_root / "split_manifest.json").exists()


def test_review_candidates_include_adjacency_ranking_fields(tmp_path: Path, monkeypatch):
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifacts"
    _write_pattern(source_root / "Healthy" / "source_a" / "healthy_0.jpg", offset=2)
    _write_pattern(source_root / "Healthy" / "source_b" / "healthy_1.jpg", offset=4)
    _write_pattern(source_root / "Healthy" / "source_c" / "healthy_2.jpg", offset=20)

    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_dinov3_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_bioclip_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_dinov3_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_bioclip_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._compute_neighbor_pairs",
        lambda embeddings, *, paths, neighbors: {
            tuple(sorted((paths[0], paths[1]))): 0.97,
            tuple(sorted((paths[0], paths[2]))): 0.966,
        },
    )

    summary = build_grouped_dataset_plan(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        taxonomy_path=None,
    )

    assert summary["summary"]["adjacency_used_for_review_ranking_only"] is True
    review_csv = artifact_root / "same_class_review_candidates.csv"
    with review_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert "adjacency_distance" in rows[0]
    assert "review_rank" in rows[0]
    assert int(rows[0]["adjacency_distance"]) <= int(rows[-1]["adjacency_distance"])
    assert rows[0]["triage_resolution"] == "manual_review"


def test_low_risk_same_class_review_clusters_are_auto_resolved(tmp_path: Path, monkeypatch):
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifacts"
    _write_pattern(source_root / "Healthy" / "healthy_aug_0.jpg", offset=2)
    _write_pattern(source_root / "Healthy" / "healthy_aug_1.jpg", offset=3)
    _write_pattern(source_root / "Healthy" / "healthy_aug_2.jpg", offset=4)

    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_dinov3_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._load_bioclip_components",
        lambda model_id, device="cpu": (object(), _FakeModel()),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_dinov3_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._encode_bioclip_with_components",
        lambda paths, **kwargs: _fake_embeddings(paths, model_id="fake", batch_size=0, device="cpu"),
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._compute_neighbor_pairs",
        lambda embeddings, *, paths, neighbors: {
            tuple(sorted((paths[0], paths[1]))): 0.97,
            tuple(sorted((paths[1], paths[2]))): 0.969,
        },
    )
    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset._phash_distance",
        lambda a, b: 6,
    )

    summary = build_grouped_dataset_plan(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        taxonomy_path=None,
    )

    assert summary["summary"]["same_class_review_pairs_total"] == 2
    assert summary["summary"]["same_class_auto_resolved_clusters"] == 1
    assert summary["summary"]["same_class_review_pairs"] == 0
    auto_clusters_csv = artifact_root / "same_class_auto_resolved_clusters.csv"
    with auto_clusters_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["resolution"] == "auto_resolve"
