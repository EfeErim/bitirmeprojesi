from pathlib import Path

from PIL import Image

from scripts.calibrate_router_prototype_reconciler import ScoredRow, calibrate, score_manifest
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
                "demo_003,local_test_pool:data/prepared_runtime_datasets/tomato__fruit/train/healthy/a.png,non_plant,abstain,",
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

    assert len(rows) == 3
    assert result["selected_policy"] is None
    assert result["best_candidate"]["supported_precision"] == 1.0
    assert result["best_candidate"]["supported_coverage"] == 1.0
    assert result["best_candidate"]["non_plant_false_accept_count"] == 1
    assert result["target_policies"]["grape__fruit"]["status"] == "no_eligible_policy"


def test_calibrate_emits_target_policy_for_safe_supported_rows():
    rows = [
        ScoredRow(
            image_id="demo_001",
            expected_target="tomato__fruit",
            expected_behavior="answer",
            predicted_target="tomato__fruit",
            similarity=0.7,
            margin=0.04,
            resolved_image="a.png",
            status="ok",
        ),
        ScoredRow(
            image_id="demo_002",
            expected_target="grape__fruit",
            expected_behavior="answer",
            predicted_target="grape__fruit",
            similarity=0.7,
            margin=0.04,
            resolved_image="b.png",
            status="ok",
        ),
    ]

    result = calibrate(
        rows,
        similarity_grid=(0.1,),
        margin_grid=(0.0,),
        min_precision=1.0,
        min_coverage=1.0,
    )

    assert result["selected_policy"]["supported_precision"] == 1.0
    assert result["target_policies"]["tomato__fruit"]["status"] == "target_specific"


def test_calibrate_blocks_unknown_crop_false_accept_by_default():
    rows = [
        ScoredRow(
            image_id="demo_001",
            expected_target="tomato__fruit",
            expected_behavior="answer",
            predicted_target="tomato__fruit",
            similarity=0.7,
            margin=0.04,
            resolved_image="a.png",
            status="ok",
        ),
        ScoredRow(
            image_id="demo_002",
            expected_target="unknown_crop",
            expected_behavior="abstain",
            predicted_target="tomato__fruit",
            similarity=0.7,
            margin=0.04,
            resolved_image="b.png",
            status="ok",
        ),
    ]

    result = calibrate(
        rows,
        similarity_grid=(0.1,),
        margin_grid=(0.0,),
        min_precision=1.0,
        min_coverage=1.0,
    )

    assert result["selected_policy"] is None
    assert result["best_candidate"]["supported_precision"] == 1.0
    assert result["best_candidate"]["negative_false_accept_count"] == 1
    assert result["best_candidate"]["non_plant_false_accept_count"] == 0


def test_calibrate_can_select_target_policies_without_global_negatives():
    rows = [
        ScoredRow(
            image_id="demo_001",
            expected_target="tomato__fruit",
            expected_behavior="answer",
            predicted_target="tomato__fruit",
            similarity=0.7,
            margin=0.04,
            resolved_image="a.png",
            status="ok",
        ),
        ScoredRow(
            image_id="demo_002",
            expected_target="unknown_crop",
            expected_behavior="abstain",
            predicted_target="tomato__fruit",
            similarity=0.7,
            margin=0.04,
            resolved_image="b.png",
            status="ok",
        ),
    ]

    result = calibrate(
        rows,
        similarity_grid=(0.1,),
        margin_grid=(0.0,),
        min_precision=1.0,
        min_coverage=1.0,
        target_policy_negative_mode="none",
    )

    assert result["selected_policy"] is None
    assert result["target_policies"]["tomato__fruit"]["status"] == "target_specific"
    assert result["target_policies"]["tomato__fruit"]["negative_mode"] == "none"


def test_calibrate_can_allow_negative_false_accepts_explicitly():
    rows = [
        ScoredRow(
            image_id="demo_001",
            expected_target="tomato__fruit",
            expected_behavior="answer",
            predicted_target="tomato__fruit",
            similarity=0.7,
            margin=0.04,
            resolved_image="a.png",
            status="ok",
        ),
        ScoredRow(
            image_id="demo_002",
            expected_target="unknown_crop",
            expected_behavior="abstain",
            predicted_target="tomato__fruit",
            similarity=0.7,
            margin=0.04,
            resolved_image="b.png",
            status="ok",
        ),
    ]

    result = calibrate(
        rows,
        similarity_grid=(0.1,),
        margin_grid=(0.0,),
        min_precision=1.0,
        min_coverage=1.0,
        max_negative_false_accepts=1,
        max_negative_false_accept_rate=1.0,
    )

    assert result["selected_policy"]["negative_false_accept_count"] == 1
