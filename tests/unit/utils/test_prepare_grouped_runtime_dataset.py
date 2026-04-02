import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scripts.prepare_grouped_runtime_dataset import (
    build_prepared_dataset_key,
    build_grouped_dataset_plan,
    materialize_grouped_runtime_dataset,
    normalize_prepared_class_name,
    scan_class_root_dataset,
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


def test_build_prepared_dataset_key_includes_part_only_when_specified():
    assert build_prepared_dataset_key("tomato", "unspecified") == "tomato"
    assert build_prepared_dataset_key("Tomato", "Fruit") == "tomato__fruit"


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


def test_grouped_dataset_plan_roundtrips_provenance_manifest_fields(tmp_path: Path, monkeypatch):
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifacts"
    runtime_root = tmp_path / "runtime"

    for index, offset in enumerate((2, 10, 18)):
        _write_pattern(source_root / "Healthy" / f"healthy_{index}.jpg", offset=offset)
        _write_pattern(source_root / "Early Blight" / f"disease_{index}.jpg", offset=offset + 1)

    provenance_path = source_root / "provenance_manifest.csv"
    image_paths = sorted(path for path in source_root.rglob("*.jpg"))
    with provenance_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "relative_path",
                "source_dataset",
                "source_subset",
                "capture_group_id",
                "domain_tag",
            ],
        )
        writer.writeheader()
        for index, image_path in enumerate(image_paths):
            writer.writerow(
                {
                    "relative_path": image_path.relative_to(source_root).as_posix(),
                    "source_dataset": "dataset_a" if index < 3 else "dataset_b",
                    "source_subset": "subset_train" if index % 2 == 0 else "subset_eval",
                    "capture_group_id": f"group_{index // 2}",
                    "domain_tag": "field" if index < 3 else "lab",
                }
            )
        writer.writerow(
            {
                "relative_path": "Healthy/missing_row.jpg",
                "source_dataset": "dataset_extra",
                "source_subset": "subset_extra",
                "capture_group_id": "group_extra",
                "domain_tag": "synthetic",
            }
        )

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
    assert summary["provenance_manifest"]["available"] is True
    assert summary["provenance_manifest"]["matched_rows"] == len(image_paths)
    assert summary["provenance_manifest"]["unmatched_manifest_rows"] == 1
    assert summary["provenance_manifest"]["warnings"]

    result_root = materialize_grouped_runtime_dataset(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        runtime_root=runtime_root,
    )

    manifest = json.loads((result_root / "tomato" / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["provenance_manifest"]["available"] is True
    rows = manifest["rows"]
    assert rows
    assert all("runtime_relative_path" in row and row["runtime_relative_path"] for row in rows)
    assert all("source_dataset" in row for row in rows)
    assert any(row["source_dataset"] == "dataset_a" for row in rows)
    assert any(row["domain_tag"] == "lab" for row in rows)


def test_materialize_grouped_runtime_dataset_uses_part_aware_dataset_key(tmp_path: Path, monkeypatch):
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
        part_name="fruit",
        artifact_root=artifact_root,
        runtime_root=runtime_root,
    )

    crop_root = result_root / "tomato__fruit"
    manifest = json.loads((crop_root / "split_manifest.json").read_text(encoding="utf-8"))

    assert (crop_root / "continual").exists()
    assert manifest["crop_name"] == "tomato"
    assert manifest["part_name"] == "fruit"
    assert manifest["dataset_key"] == "tomato__fruit"


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


def test_materialized_eval_splits_only_contain_canonical_clean_family_members(tmp_path: Path, monkeypatch):
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifacts"
    runtime_root = tmp_path / "runtime"

    for index, offset in enumerate((2, 10, 18)):
        _write_pattern(source_root / "Healthy" / f"healthy_clean_{index}.jpg", offset=offset)
        _write_pattern(source_root / "Early Blight" / f"disease_clean_{index}.jpg", offset=offset + 1)

    _write_pattern(source_root / "Healthy" / "healthy_real.jpg", offset=22)
    _write_pattern(source_root / "Healthy" / "healthy_aug_flip.jpg", offset=23)
    _write_pattern(source_root / "Early Blight" / "disease_real.jpg", offset=24)
    _write_pattern(source_root / "Early Blight" / "disease_aug_flip.jpg", offset=25)

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
            tuple(sorted((real_path, aug_path))): 0.99
            for real_path in paths
            for aug_path in paths
            if "real" in real_path and "aug" in aug_path
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

    assert summary["runtime_ready"] is True

    result_root = materialize_grouped_runtime_dataset(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        runtime_root=runtime_root,
    )

    crop_root = result_root / "tomato"
    eval_files = {
        path.relative_to(crop_root).as_posix()
        for split_name in ("val", "test")
        for path in (crop_root / split_name).rglob("*.jpg")
    }
    assert all("aug" not in path for path in eval_files)

    manifest_rows = json.loads((artifact_root / "proposed_split_manifest.json").read_text(encoding="utf-8"))["rows"]
    synthetic_rows = [row for row in manifest_rows if row["family_role"] == "canonical_with_synthetic_derivatives"]
    assert synthetic_rows
    assert all(row["split"] == "continual" for row in synthetic_rows if not row["is_family_canonical"])


def test_build_grouped_dataset_plan_excludes_images_missing_after_scan(tmp_path: Path, monkeypatch):
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifacts"

    existing_paths = [
        source_root / "Healthy" / "healthy_0.jpg",
        source_root / "Healthy" / "healthy_1.jpg",
        source_root / "Healthy" / "healthy_2.jpg",
    ]
    for index, path in enumerate(existing_paths):
        _write_pattern(path, offset=index + 2)

    missing_path = source_root / "Healthy" / "vanished.jpg"

    real_records, normalization_report = scan_class_root_dataset(
        class_root=source_root,
        crop_name="tomato",
        taxonomy_path=None,
    )
    missing_record = next(record for record in real_records if record.relative_path.endswith("healthy_0.jpg"))
    missing_record.relative_path = "Healthy/vanished.jpg"
    missing_record.absolute_path = str(missing_path.resolve())

    monkeypatch.setattr(
        "scripts.prepare_grouped_runtime_dataset.scan_class_root_dataset",
        lambda **kwargs: (real_records, normalization_report),
    )
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

    assert summary["summary"]["excluded_images"] == 1
    assert summary["summary"]["readable_images"] == 2
    assert summary["summary"]["excluded_reason_breakdown"]["missing_after_scan"] == 1

def test_grouped_dataset_plan_writes_guided_catalog(tmp_path: Path, monkeypatch):
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

    build_grouped_dataset_plan(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        taxonomy_path=None,
    )
    materialize_grouped_runtime_dataset(
        class_root=source_root,
        crop_name="tomato",
        artifact_root=artifact_root,
        runtime_root=runtime_root,
    )

    guided_dir = artifact_root / "guided"
    catalog = json.loads((guided_dir / "02_file_catalog.json").read_text(encoding="utf-8"))

    assert (guided_dir / "00_start_here.md").exists()
    assert (guided_dir / "01_prep_overview.json").exists()
    assert any(entry["relative_path"] == "prep_summary.json" for entry in catalog["entries"])
    assert any(entry["title_tr"] == "Materyalize edilmis runtime split manifesti" for entry in catalog["entries"])
