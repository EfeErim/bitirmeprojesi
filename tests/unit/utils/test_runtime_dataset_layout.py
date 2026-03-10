from pathlib import Path

from PIL import Image

from scripts.colab_dataset_layout import (
    build_runtime_split_manifest,
    prepare_runtime_dataset_layout,
    resolve_notebook_training_classes,
)


def _write_images(root: Path, class_name: str, count: int) -> None:
    class_dir = root / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        Image.new("RGB", (8, 8), color=(255, 0, 0)).save(class_dir / f"image_{idx}.jpg")


def test_build_runtime_split_manifest_contains_counts(tmp_path: Path):
    _write_images(tmp_path, "Tomato Healthy", 4)
    _write_images(tmp_path, "Tomato Blight", 2)

    manifest = build_runtime_split_manifest(class_root=tmp_path, crop_name="tomato", seed=123)

    assert manifest["crop"] == "tomato"
    assert manifest["seed"] == 123
    assert manifest["split_policy"] == "80/10/10"
    assert manifest["summary"]["num_classes"] == 2
    assert any(item["class_name"] == "tomato_healthy" for item in manifest["classes"])


def test_prepare_runtime_dataset_layout_writes_split_manifest(tmp_path: Path):
    source_root = tmp_path / "source"
    runtime_root = tmp_path / "runtime"
    _write_images(source_root, "Healthy", 5)
    _write_images(source_root, "Disease A", 4)

    result_root = prepare_runtime_dataset_layout(
        source_root,
        "tomato",
        seed=42,
        runtime_root=runtime_root,
    )

    crop_root = result_root / "tomato"
    assert (crop_root / "split_manifest.json").exists()
    assert (crop_root / "_split_metadata.json").exists()
    assert (crop_root / "continual").exists()
    assert (crop_root / "val").exists()
    assert (crop_root / "test").exists()


def test_prepare_runtime_dataset_layout_preserves_nested_relative_paths(tmp_path: Path):
    source_root = tmp_path / "source"
    runtime_root = tmp_path / "runtime"
    nested_dir = source_root / "Healthy" / "camera_a"
    nested_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(nested_dir / "image_nested.jpg")

    result_root = prepare_runtime_dataset_layout(
        source_root,
        "tomato",
        seed=42,
        runtime_root=runtime_root,
    )

    crop_root = result_root / "tomato"
    nested_targets = list(crop_root.rglob("camera_a/image_nested.jpg"))
    assert len(nested_targets) == 1


def test_resolve_notebook_training_classes_uses_taxonomy_when_aliases_cover_dataset():
    resolution = resolve_notebook_training_classes(
        available_classes=[
            "Tomato Early Blight",
            "Tomato Healthy Leaf",
            "Tomato Late Blight",
        ],
        crop_name="tomato",
        taxonomy={
            "crop_specific_diseases": {
                "tomato": [
                    "early blight",
                    "late blight",
                ]
            }
        },
    )

    assert resolution["used_taxonomy_filter"] is True
    assert resolution["reason"] == "full_taxonomy_alignment"
    assert resolution["unmatched_classes"] == []
    assert set(resolution["selected_classes"]) == {
        "tomato_early_blight",
        "tomato_healthy_leaf",
        "tomato_late_blight",
    }


def test_resolve_notebook_training_classes_falls_back_to_all_available_when_taxonomy_is_incomplete():
    resolution = resolve_notebook_training_classes(
        available_classes=[
            "Tomato Early Blight",
            "Tomato Healthy Leaf",
            "Tomato Spider Mites",
        ],
        crop_name="tomato",
        taxonomy={
            "crop_specific_diseases": {
                "tomato": [
                    "early blight",
                ]
            }
        },
    )

    assert resolution["used_taxonomy_filter"] is False
    assert resolution["reason"] == "partial_taxonomy_alignment_fallback"
    assert "tomato_spider_mites" in resolution["unmatched_classes"]
    assert set(resolution["selected_classes"]) == {
        "tomato_early_blight",
        "tomato_healthy_leaf",
        "tomato_spider_mites",
    }
