import json
from pathlib import Path

from scripts.index_training_runs import build_run_registry, collect_trial_records


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _canonical_record(*, run_id: str, cohort_key: str, artifact_root: Path) -> tuple[dict, dict]:
    manifest = {
        "schema_version": "v1_training_experiment_manifest",
        "record_quality": "canonical",
        "run_id": run_id,
        "run_label": run_id,
        "created_at": "2026-04-14T12:00:00+00:00",
        "surface": "notebook_2",
        "crop_name": "tomato",
        "part_name": "leaf",
        "dataset_key": "tomato__leaf",
        "dataset_lineage_key": "tomato__leaf::sha_a",
        "model_family": {"engine": "continual_sd_lora", "backbone_model_name": "fake/backbone"},
        "artifacts": {"artifact_root": str(artifact_root)},
    }
    optimization = {
        "schema_version": "v1_training_optimization_record",
        "record_quality": "canonical",
        "run_id": run_id,
        "run_label": run_id,
        "created_at": "2026-04-14T12:00:00+00:00",
        "surface": "notebook_2",
        "crop_name": "tomato",
        "part_name": "leaf",
        "dataset_key": "tomato__leaf",
        "dataset_lineage_key": "tomato__leaf::sha_a",
        "comparability": {
            "dataset_lineage_key": "tomato__leaf::sha_a",
            "crop_name": "tomato",
            "part_name": "leaf",
            "engine": "continual_sd_lora",
            "backbone_model_name": "fake/backbone",
            "cohort_key": cohort_key,
        },
        "status": {
            "readiness_status": "ready",
            "readiness_passed": True,
            "authoritative_split": "test",
            "ood_evidence_source": "real_ood_split",
        },
        "parameters": {"training.learning_rate": 0.00015},
        "objectives": {"classification.macro_f1": 0.93},
        "objective_directions": {"classification.macro_f1": "maximize"},
        "artifacts": {"artifact_root": str(artifact_root)},
    }
    return manifest, optimization


def test_collect_trial_records_dedupes_mirrored_notebook_artifacts(tmp_path: Path):
    runs_root = tmp_path / "runs"
    outputs_root = runs_root / "run_1" / "outputs" / "colab_notebook_training" / "artifacts"
    telemetry_root = runs_root / "run_1" / "telemetry" / "artifacts"
    for root in (outputs_root, telemetry_root):
        manifest, optimization = _canonical_record(
            run_id="tomato_leaf_2026-04-14_12-00-00",
            cohort_key="cohort_a",
            artifact_root=root,
        )
        _write_json(root / "training" / "experiment_manifest.json", manifest)
        _write_json(root / "training" / "optimization_record.json", optimization)

    trials = collect_trial_records(runs_root)

    assert len(trials) == 1
    assert trials[0]["record_quality"] == "canonical"
    assert str(trials[0]["registry_source"]["artifact_root"]).replace("\\", "/").endswith(
        "outputs/colab_notebook_training/artifacts"
    )


def test_build_run_registry_backfills_old_runs_and_separates_cohorts(tmp_path: Path):
    runs_root = tmp_path / "runs"
    first_root = runs_root / "run_a" / "training_metrics"
    second_root = runs_root / "run_b" / "training_metrics"

    for artifact_root, dataset_sha, part_name in (
        (first_root, "sha_a", "leaf"),
        (second_root, "sha_b", "fruit"),
    ):
        _write_json(
            artifact_root / "training" / "summary.json",
            {
                "run_id": f"{artifact_root.parent.name}_run",
                "run_label": f"{artifact_root.parent.name}_run",
                "created_at": "2026-04-14T12:00:00+00:00",
                "surface": "workflow",
                "crop_name": "tomato",
                "part_name": part_name,
                "dataset_key": f"tomato__{part_name}",
                "class_count": 2,
                "loader_sizes": {"train": 120, "val": 30, "test": 30, "ood": 10},
                "optimizer_steps": 64,
                "global_step": 128,
                "checkpoint_count": 2,
            },
        )
        _write_json(
            artifact_root / "training" / "run_context.json",
            {
                "run_id": f"{artifact_root.parent.name}_run",
                "created_at": "2026-04-14T12:00:00+00:00",
                "surface": "workflow",
                "crop_name": "tomato",
                "part_name": part_name,
                "resolved_config": {
                    "training": {
                        "continual": {
                            "backbone": {"model_name": "fake/backbone"},
                            "adapter": {"lora_r": 24, "lora_alpha": 24, "lora_dropout": 0.1},
                            "fusion": {"layers": [2, 5, 8, 11], "output_dim": 768, "dropout": 0.1, "gating": "softmax"},
                            "ood": {
                                "threshold_factor": 3.0,
                                "primary_score_method": "ensemble",
                                "radial_l2_enabled": True,
                                "radial_beta_range": [0.5, 2.0],
                                "radial_beta_steps": 16,
                                "sure_enabled": True,
                                "sure_semantic_percentile": 90.0,
                                "sure_confidence_percentile": 97.0,
                                "conformal_enabled": True,
                                "conformal_alpha": 0.05,
                                "conformal_method": "raps",
                                "conformal_raps_lambda": 0.2,
                                "conformal_raps_k_reg": 1,
                            },
                            "learning_rate": 0.00015,
                            "weight_decay": 0.01,
                            "num_epochs": 16,
                            "batch_size": 8,
                            "seed": 42,
                            "optimization": {
                                "loss_name": "logitnorm",
                                "logitnorm_tau": 1.0,
                                "grad_accumulation_steps": 4,
                                "mixed_precision": "auto",
                                "max_grad_norm": 1.0,
                                "scheduler": {
                                    "name": "cosine",
                                    "warmup_ratio": 0.1,
                                    "min_lr": 1e-6,
                                    "step_on": "batch",
                                },
                            },
                            "data": {
                                "sampler": "auto",
                                "augmentation_policy": "randaugment",
                                "randaugment_num_ops": 2,
                                "randaugment_magnitude": 7,
                            },
                        }
                    }
                },
                "dataset": {
                    "crop_root": str(artifact_root / "runtime"),
                    "dataset_key": f"tomato__{part_name}",
                    "resolution_source": "manifest:split_manifest.json",
                    "manifests": {
                        "split_manifest.json": {
                            "path": str(artifact_root / "runtime" / "split_manifest.json"),
                            "exists": True,
                            "sha256": dataset_sha,
                            "schema_version": "v1_grouped_runtime_layout",
                            "source_root": f"/tmp/source/{part_name}",
                            "crop_name": "tomato",
                            "part_name": part_name,
                            "dataset_key": f"tomato__{part_name}",
                            "split_policy": "grouped_family_canonical_eval_60_20_20",
                            "ood": {"source_root": f"/tmp/ood/{part_name}", "image_count": 10, "image_fingerprint": f"fp_{part_name}"},
                        }
                    },
                },
                "training_runtime": {"train_sampler": {"resolved_sampler": "weighted"}},
            },
        )
        _write_json(
            artifact_root / "production_readiness.json",
            {
                "status": "ready",
                "passed": True,
                "ood_evidence_source": "real_ood_split",
                "classification_evidence": {
                    "split_name": "test",
                    "metrics": {"accuracy": 0.95, "balanced_accuracy": 0.94, "macro_f1": 0.93},
                },
                "ood_evidence": {
                    "metrics": {
                        "ood_auroc": 0.91,
                        "ood_false_positive_rate": 0.04,
                        "sure_ds_f1": 0.90,
                        "conformal_empirical_coverage": 0.96,
                        "conformal_avg_set_size": 1.2,
                        "ood_samples": 10,
                        "in_distribution_samples": 30,
                    }
                },
            },
        )
        _write_json(
            artifact_root / "test" / "classification_report.json",
            {
                "accuracy": 0.95,
                "macro avg": {"f1-score": 0.93},
                "weighted avg": {"f1-score": 0.92},
            },
        )
        _write_json(
            artifact_root / "test" / "metric_gate.json",
            {
                "metrics": {
                    "accuracy": 0.95,
                    "balanced_accuracy": 0.94,
                    "macro_f1": 0.93,
                    "classification_samples": 30,
                }
            },
        )

    result = build_run_registry(runs_root=runs_root)
    latest_registry = result["latest_registry"]
    trials = [json.loads(line) for line in (runs_root / "_index" / "trials.jsonl").read_text(encoding="utf-8").splitlines()]
    pareto_frontiers = json.loads((runs_root / "_index" / "pareto_frontiers.json").read_text(encoding="utf-8"))
    bayesian_recommendations = json.loads((runs_root / "_index" / "bayesian_recommendations.json").read_text(encoding="utf-8"))

    assert latest_registry["trial_count"] == 2
    assert latest_registry["backfilled_count"] == 2
    assert latest_registry["cohort_count"] == 2
    assert latest_registry["paths"]["pareto_frontiers_json"].endswith("pareto_frontiers.json")
    assert latest_registry["paths"]["bayesian_recommendations_json"].endswith("bayesian_recommendations.json")
    assert {trial["record_quality"] for trial in trials} == {"backfilled"}
    assert {trial["comparability"]["cohort_key"] for trial in trials} == {
        "tomato__leaf::sha_a::tomato::leaf::continual_sd_lora::fake/backbone",
        "tomato__fruit::sha_b::tomato::fruit::continual_sd_lora::fake/backbone",
    }
    assert pareto_frontiers["cohort_count"] == 2
    assert bayesian_recommendations["cohort_count"] == 2
