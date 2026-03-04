from pathlib import Path

from src.utils.data_loader import CropDataset, infer_crop_classes_from_layout


def test_infer_crop_classes_from_layout_reads_split_dirs(tmp_path: Path):
    root = tmp_path / "runtime"
    crop = "tomato leaf"

    for split in ("continual", "val", "test"):
        for cls in ("bacterial_spot", "healthy", "late_blight"):
            target = root / crop / split / cls
            target.mkdir(parents=True, exist_ok=True)

    classes = infer_crop_classes_from_layout(str(root), crop)

    assert classes == ["bacterial_spot", "healthy", "late_blight"]


def test_crop_dataset_uses_inferred_classes_for_custom_crop_name(tmp_path: Path):
    root = tmp_path / "runtime"
    crop = "tomato leaf"

    class_to_count = {
        "bacterial_spot": 2,
        "healthy": 2,
        "late_blight": 2,
    }

    for cls, count in class_to_count.items():
        cls_dir = root / crop / "val" / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            (cls_dir / f"img_{i}.jpg").write_bytes(b"x")

    dataset = CropDataset(
        data_dir=str(root),
        crop=crop,
        split="val",
        transform=False,
        use_cache=False,
    )

    assert dataset.classes == ["bacterial_spot", "healthy", "late_blight"]
    assert set(dataset.labels) == {0, 1, 2}
    assert len(dataset) == sum(class_to_count.values())
