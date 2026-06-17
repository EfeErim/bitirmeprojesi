import json
from pathlib import Path

from scripts.build_m2_supported_disease_manifest import build_rows, is_healthy_class
from scripts.run_demo_checklist import (
    build_parser,
    classify_failure,
    parse_checklist_rows,
    parse_manifest_rows,
    resolve_image_path,
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
                "image_id,source,expected_target,expected_behavior,notes,origin_url",
                "demo_049,staged_external:.runtime_tmp/x.jpg,tomato__leaf,\"answer or review, no crash\",internet,http://example.test",
            ]
        ),
        encoding="utf-8",
    )

    rows = parse_manifest_rows(manifest)

    assert len(rows) == 1
    assert rows[0].image_id == "demo_049"
    assert rows[0].expected_target == "tomato__leaf"
    assert rows[0].notes == "internet"


def test_parser_supports_manifest_only_runs():
    args = build_parser().parse_args(["--no-checklist", "--extra-manifest", "manifest.csv"])

    assert args.no_checklist is True
    assert [str(path) for path in args.extra_manifest] == ["manifest.csv"]


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
