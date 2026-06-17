from pathlib import Path

from PIL import Image

from scripts.calibrate_router_prototype_reconciler import calibrate, score_manifest
from src.router.prototype_bank import build_prototype_bank, write_prototype_bank


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


def test_score_manifest_and_calibrate_selects_policy(tmp_path: Path):
    repo_root = tmp_path
    dataset_root = repo_root / "data" / "prepared_runtime_datasets"
    tomato_image = dataset_root / "tomato__fruit" / "train" / "healthy" / "a.png"
    grape_image = dataset_root / "grape__fruit" / "train" / "healthy" / "b.png"
    _write_image(tomato_image, (190, 30, 30))
    _write_image(grape_image, (40, 20, 120))
    prototype_payload = build_prototype_bank(dataset_root=dataset_root, created_at="20260617T000000Z")
    prototype_path = write_prototype_bank(prototype_payload, tmp_path / "prototype_bank.json")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "\n".join(
            [
                "image_id,source,expected_target,expected_behavior,notes",
                "demo_001,local_test_pool:data/prepared_runtime_datasets/tomato__fruit/train/healthy/a.png,tomato__fruit,answer,",
                "demo_002,local_test_pool:data/prepared_runtime_datasets/grape__fruit/train/healthy/b.png,grape__fruit,answer,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = score_manifest(manifest_path=manifest, prototype_bank_path=prototype_path, repo_root=repo_root)
    result = calibrate(
        rows,
        similarity_grid=(0.1,),
        margin_grid=(0.0,),
        min_precision=1.0,
        min_coverage=1.0,
    )

    assert len(rows) == 2
    assert result["selected_policy"]["precision"] == 1.0
    assert result["selected_policy"]["coverage"] == 1.0
