from pathlib import Path

from PIL import Image

from src.router.prototype_bank import build_prototype_bank, centroid, euclidean_distance


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 10), color=color).save(path)


def test_build_prototype_bank_from_synthetic_fixture(tmp_path: Path):
    dataset_root = tmp_path / "datasets"
    _write_image(dataset_root / "tomato__leaf" / "train" / "healthy" / "a.png", (20, 80, 40))
    _write_image(dataset_root / "tomato__leaf" / "test" / "late_blight" / "b.png", (180, 30, 20))
    _write_image(dataset_root / "tomato__leaf" / "ood" / "nonplant" / "c.png", (0, 0, 0))

    payload = build_prototype_bank(
        dataset_root=dataset_root,
        max_images_per_class=2,
        created_at="20260617T000000Z",
    )

    assert payload["schema_version"] == "router_prototype_bank.v1"
    assert payload["embedding_backend"] == "image_stats_v1"
    assert payload["summary"]["targets"] == ["tomato__leaf"]
    assert payload["summary"]["sample_count"] == 2
    assert payload["summary"]["class_prototype_count"] == 2
    assert payload["target_prototypes"]["tomato__leaf"]["split_counts"] == {"test": 1, "train": 1}
    assert "tomato__leaf::late_blight" in payload["class_prototypes"]
    assert payload["skipped"] == []


def test_build_prototype_bank_honors_max_images_per_class(tmp_path: Path):
    dataset_root = tmp_path / "datasets"
    _write_image(dataset_root / "grape__fruit" / "train" / "healthy" / "a.png", (10, 10, 10))
    _write_image(dataset_root / "grape__fruit" / "train" / "healthy" / "b.png", (20, 20, 20))

    payload = build_prototype_bank(
        dataset_root=dataset_root,
        max_images_per_class=1,
        created_at="20260617T000000Z",
    )

    assert payload["summary"]["sample_count"] == 1
    assert payload["target_prototypes"]["grape__fruit"]["sample_count"] == 1


def test_vector_math_helpers_are_deterministic():
    center = centroid([(0.0, 2.0), (2.0, 4.0)])

    assert center == (1.0, 3.0)
    assert round(euclidean_distance((0.0, 0.0), (3.0, 4.0)), 4) == 5.0
