import json
from pathlib import Path

from scripts.build_m2_supported_disease_manifest import build_rows, is_healthy_class
from scripts.run_demo_checklist import (
    build_analysis_summary,
    build_parser,
    classify_failure,
    parse_checklist_rows,
    parse_manifest_rows,
    resolve_image_path,
    resolve_prototype_thresholds_from_calibration,
    summarize_results,
)


def test_parse_checklist_rows_reads_demo_table(tmp_path: Path):
    checklist = tmp_path / "demo.md"
    checklist.write_text(
        "\n".join(
            [
                "| image_id | source | expected_target | expected_behavior | actual_status | predicted_crop | "
                "predicted_part | predicted_disease | confidence_or_ood | pass_fail | failure_bucket | notes |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|",
                "| demo_xxx | internet | tomato__leaf | template row |  |  |  |  |  |  |  | ignore |",
                "",
                "## M1 Candidate Checklist",
                "",
                "| image_id | source | expected_target | expected_behavior | actual_status | predicted_crop | "
                "predicted_part | predicted_disease | confidence_or_ood | pass_fail | failure_bucket | notes |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|",
                "| demo_001 | local_test_pool:data/x | tomato__leaf | known disease |  |  |  |  |  |  |  | note |",
                "",
                "## Pass Criteria",
            ]
        ),
        encoding="utf-8",
    )

    rows = parse_checklist_rows(checklist)

    assert len(rows) == 1
    assert rows[0].image_id == "demo_001"
    assert rows[0].expected_target == "tomato__leaf"
    assert rows[0].notes == "note"


def test_resolve_image_path_uses_first_sorted_local_image(tmp_path: Path):
    image_dir = tmp_path / "data" / "x"
    image_dir.mkdir(parents=True)
    (image_dir / "b.png").write_bytes(b"b")
    (image_dir / "a.jpg").write_bytes(b"a")

    image, status = resolve_image_path("local_test_pool:data/x", tmp_path)

    assert status == "ok"
    assert image == image_dir / "a.jpg"


def test_resolve_image_path_matches_ascii_source_to_unicode_folder(tmp_path: Path):
    image_dir = tmp_path / "data" / "test" / "üzüm_sağlıklı_meyve"
    image_dir.mkdir(parents=True)
    (image_dir / "image.png").write_bytes(b"x")

    image, status = resolve_image_path("local_test_pool:data/test/uzum_saglikli_meyve", tmp_path)

    assert status == "ok"
    assert image == image_dir / "image.png"


def test_parse_manifest_rows_reads_extra_csv(tmp_path: Path):
    manifest = tmp_path / "extra.csv"
    manifest.write_text(
        "\n".join(
            [
                "image_id,source,expected_target,expected_crop,expected_part,expected_class,expected_behavior,notes,origin_url",
                (
                    "demo_049,staged_external:.runtime_tmp/x.jpg,tomato__leaf,tomato,leaf,"
                    "domates_late_blight_yaprak,\"answer or review, no crash\",internet,http://example.test"
                ),
            ]
        ),
        encoding="utf-8",
    )

    rows = parse_manifest_rows(manifest)

    assert len(rows) == 1
    assert rows[0].image_id == "demo_049"
    assert rows[0].expected_target == "tomato__leaf"
    assert rows[0].expected_crop == "tomato"
    assert rows[0].expected_part == "leaf"
    assert rows[0].expected_class == "domates_late_blight_yaprak"
    assert rows[0].notes == "internet"


def test_parse_manifest_rows_infers_expected_fields_from_existing_manifest_shape(tmp_path: Path):
    manifest = tmp_path / "extra.csv"
    manifest.write_text(
        "\n".join(
            [
                "image_id,source,expected_target,expected_behavior,notes,original_source",
                (
                    "demo_049,staged_external:.runtime_tmp/x.jpg,tomato__fruit,"
                    "\"answer or review, no crash\",internet,"
                    "local_test_pool:data/prepared_runtime_datasets/tomato__fruit/test/domates_late_blight_meyve/x.jpg"
                ),
            ]
        ),
        encoding="utf-8",
    )

    rows = parse_manifest_rows(manifest)

    assert rows[0].expected_crop == "tomato"
    assert rows[0].expected_part == "fruit"
    assert rows[0].expected_class == "domates_late_blight_meyve"


def test_parser_supports_manifest_only_runs():
    args = build_parser().parse_args(["--no-checklist", "--extra-manifest", "manifest.csv"])

    assert args.no_checklist is True
    assert [str(path) for path in args.extra_manifest] == ["manifest.csv"]


def test_resolve_prototype_thresholds_from_calibration_uses_selected_policy(tmp_path: Path):
    report_path = tmp_path / "calibration.json"
    report_path.write_text(
        json.dumps(
            {
                "selected_policy": {
                    "min_similarity": 0.4,
                    "min_margin": 0.08,
                    "min_negative_gap": 0.02,
                    "precision": 0.95,
                    "coverage": 0.7,
                },
                "target_policies": {"tomato__leaf": {"status": "target_specific"}},
            }
        ),
        encoding="utf-8",
    )

    min_similarity, min_margin, min_negative_gap, report, target_policies = (
        resolve_prototype_thresholds_from_calibration(
            report_path,
            min_similarity=None,
            min_margin=None,
        )
    )

    assert min_similarity == 0.4
    assert min_margin == 0.08
    assert min_negative_gap == 0.02
    assert report["policy_selected"] is True
    assert report["selected_policy"]["precision"] == 0.95
    assert target_policies["tomato__leaf"]["status"] == "target_specific"


def test_resolve_prototype_thresholds_preserves_explicit_values(tmp_path: Path):
    report_path = tmp_path / "calibration.json"
    report_path.write_text(json.dumps({"selected_policy": {"min_similarity": 0.4, "min_margin": 0.08}}), encoding="utf-8")

    min_similarity, min_margin, min_negative_gap, _report, target_policies = resolve_prototype_thresholds_from_calibration(
        report_path,
        min_similarity=0.6,
        min_margin=0.1,
        min_negative_gap=0.03,
    )

    assert min_similarity == 0.6
    assert min_margin == 0.1
    assert min_negative_gap == 0.03
    assert target_policies == {}


def test_resolve_prototype_thresholds_preserves_explicit_negative_gap_without_report():
    min_similarity, min_margin, min_negative_gap, report, target_policies = resolve_prototype_thresholds_from_calibration(
        None,
        min_similarity=0.6,
        min_margin=0.1,
        min_negative_gap=0.04,
    )

    assert min_similarity == 0.6
    assert min_margin == 0.1
    assert min_negative_gap == 0.04
    assert report == {"enabled": False}
    assert target_policies == {}


def test_supported_disease_manifest_excludes_healthy_classes(tmp_path: Path):
    dataset_root = tmp_path / "datasets"
    disease_dir = dataset_root / "tomato__leaf" / "test" / "domates_late_blight_yaprak"
    healthy_dir = dataset_root / "tomato__leaf" / "test" / "domates_sağlıklı_yaprak"
    disease_dir.mkdir(parents=True)
    healthy_dir.mkdir(parents=True)
    (disease_dir / "image.png").write_bytes(b"x")
    (healthy_dir / "image.png").write_bytes(b"x")

    for idx in range(1, 12):
        (disease_dir / f"image_{idx:02d}.png").write_bytes(b"x")

    rows = build_rows(dataset_root, start_id=145, images_per_class=10)

    assert is_healthy_class("domates_sağlıklı_yaprak")
    assert len(rows) == 10
    assert rows[0]["image_id"] == "demo_145"
    assert rows[0]["expected_target"] == "tomato__leaf"
    assert rows[0]["expected_crop"] == "tomato"
    assert rows[0]["expected_part"] == "leaf"
    assert rows[0]["expected_class"] == "domates_late_blight_yaprak"
    assert rows[0]["disease_class"] == "domates_late_blight_yaprak"
    assert rows[0]["source"].endswith("image.png")
    assert rows[-1]["source"].endswith("image_09.png")


def test_classify_failure_marks_gated_model_as_dependency_access():
    bucket = classify_failure(
        {
            "status": "router_unavailable",
            "message": "Strict VLM model loading failed: gated repo 401 Client Error",
        },
        asset_status="ok",
    )

    assert bucket == "dependency_access"


def test_summarize_results_counts_buckets_and_targets():
    summary = summarize_results(
        [
            {
                "actual_status": "success",
                "pass_fail": "pass",
                "expected_target": "tomato__leaf",
                "failure_bucket": "",
            },
            {
                "actual_status": "router_unavailable",
                "pass_fail": "fail",
                "expected_target": "tomato__leaf",
                "failure_bucket": "dependency_access",
            },
        ]
    )

    assert json.loads(json.dumps(summary)) == {
        "total": 2,
        "passed": 1,
        "failed": 1,
        "answered": 1,
        "abstained_or_reviewed": 1,
        "asset_ready": 0,
        "failure_buckets": {"dependency_access": 1},
        "per_target": {"tomato__leaf": {"total": 2, "pass": 1, "fail": 1}},
    }


def test_build_analysis_summary_separates_router_and_adapter_failures():
    analysis = build_analysis_summary(
        [
            {
                "image_id": "demo_001",
                "actual_status": "success",
                "pass_fail": "pass",
                "expected_target": "tomato__fruit",
                "expected_crop": "tomato",
                "expected_part": "fruit",
                "expected_class": "domates_late_blight_meyve",
                "predicted_crop": "tomato",
                "predicted_part": "fruit",
                "predicted_disease": "domates_late_blight_meyve",
                "failure_bucket": "",
            },
            {
                "image_id": "demo_002",
                "actual_status": "success",
                "pass_fail": "fail",
                "expected_target": "tomato__fruit",
                "expected_crop": "tomato",
                "expected_part": "fruit",
                "expected_class": "domates_late_blight_meyve",
                "predicted_crop": "tomato",
                "predicted_part": "fruit",
                "predicted_disease": "domates_late_blight_yaprak",
                "failure_bucket": "",
            },
            {
                "image_id": "demo_003",
                "actual_status": "adapter_unavailable",
                "pass_fail": "fail",
                "expected_target": "tomato__fruit",
                "expected_crop": "tomato",
                "expected_part": "fruit",
                "expected_class": "",
                "predicted_crop": "potato",
                "predicted_part": "leaf",
                "predicted_disease": None,
                "failure_bucket": "adapter_loading",
            },
        ]
    )

    assert analysis["router_crop_correctness"] == {"correct": 2, "incorrect": 1, "not_applicable": 0}
    assert analysis["router_part_correctness"] == {"correct": 2, "incorrect": 1, "not_applicable": 0}
    assert analysis["normalized_disease_class_correctness"] == {
        "correct": 1,
        "incorrect": 1,
        "not_applicable": 1,
    }
    assert analysis["adapter_unavailable"] == {"wrong_router": 1, "missing_adapter": 0, "unknown": 0}
    assert analysis["opposite_part_disease_labels"]["image_ids"] == ["demo_002"]
