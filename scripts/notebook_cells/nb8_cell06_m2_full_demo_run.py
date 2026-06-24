# Auto-extracted from colab_notebooks/8_auto_router_adapter_prediction.ipynb cell 6.
# Keep notebook execute-only cells thin; edit behavior here.

import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.colab_notebook_helpers import maybe_auto_disconnect_colab_runtime
from scripts.colab_repo_bootstrap import push_repo_paths_to_github
from scripts.compare_m2_demo_results import (
    compare_results,
    comparison_markdown,
    enrich_summary_manifest_sha256,
)

M2_RUN_FULL_DEMO = bool(globals().get("M2_RUN_FULL_DEMO", True))
M2_DEMO_MANIFEST = str(
    globals().get(
        "M2_DEMO_MANIFEST",
        "docs/demo_assets/m2_full_image_set/manifests/m2_full_image_set_run_manifest.csv",
    )
)
M2_DEMO_OUTPUT = str(globals().get("M2_DEMO_OUTPUT", ".runtime_tmp/m2_demo_checklist_run.json"))
M2_DEMO_MARKDOWN_OUTPUT = str(
    globals().get("M2_DEMO_MARKDOWN_OUTPUT", ".runtime_tmp/m2_demo_checklist_run.md")
)
M2_ANALYSIS_OUTPUT = str(globals().get("M2_ANALYSIS_OUTPUT", ".runtime_tmp/analysis_summary.json"))
M2_ANALYSIS_MARKDOWN_OUTPUT = str(globals().get("M2_ANALYSIS_MARKDOWN_OUTPUT", ".runtime_tmp/analysis_summary.md"))
M2_DEMO_LIMIT = globals().get("M2_DEMO_LIMIT", None)
M2_BATCH_SIZE = int(globals().get("M2_BATCH_SIZE", 12))
M2_ADAPTER_BATCH_SIZE = int(globals().get("M2_ADAPTER_BATCH_SIZE", 24))
M2_HANDOFF_CACHE = str(globals().get("M2_HANDOFF_CACHE", ".runtime_tmp/m2_router_prototype_handoff_cache.json"))
M2_REFRESH_HANDOFF_CACHE = bool(globals().get("M2_REFRESH_HANDOFF_CACHE", False))
M2_STOP_ON_DEPENDENCY_BLOCKER = bool(globals().get("M2_STOP_ON_DEPENDENCY_BLOCKER", True))
M2_AUTO_PUSH_RESULTS = bool(globals().get("M2_AUTO_PUSH_RESULTS", True))
M2_AUTO_PUSH_REMOTE_NAME = str(globals().get("M2_AUTO_PUSH_REMOTE_NAME", "origin"))
M2_AUTO_PUSH_BRANCH = str(globals().get("M2_AUTO_PUSH_BRANCH", "master") or "").strip() or None
M2_REPO_RESULTS_ROOT = str(globals().get("M2_REPO_RESULTS_ROOT", "docs/demo_results/m2"))
M2_COMPARISON_BASELINE = str(
    globals().get("M2_COMPARISON_BASELINE", "docs/demo_results/m2/20260622T161859Z/summary.json") or ""
)
M2_AUTO_DISCONNECT_RUNTIME = bool(globals().get("M2_AUTO_DISCONNECT_RUNTIME", True))
M2_AUTO_DISCONNECT_GRACE_SECONDS = float(globals().get("M2_AUTO_DISCONNECT_GRACE_SECONDS", 20))


def format_elapsed_seconds(seconds):
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"
DEVICE = str(globals().get("DEVICE", "cuda"))
CONFIG_ENV = str(globals().get("CONFIG_ENV", "colab"))
M2_ENABLE_PROTOTYPE_RECONCILER = bool(globals().get("M2_ENABLE_PROTOTYPE_RECONCILER", True))
M2_AUTO_BUILD_PROTOTYPES = bool(globals().get("M2_AUTO_BUILD_PROTOTYPES", True))
M2_PROTOTYPE_RUN_ID = str(globals().get("M2_PROTOTYPE_RUN_ID", "") or "")
M2_PROTOTYPE_EMBEDDING_BACKEND = str(globals().get("M2_PROTOTYPE_EMBEDDING_BACKEND", "bioclip_open_clip"))
M2_PROTOTYPE_EMBEDDING_MODEL_ID = str(
    globals().get("M2_PROTOTYPE_EMBEDDING_MODEL_ID", "imageomics/bioclip-2.5-vith14")
)
M2_PROTOTYPE_EMBEDDING_DEVICE = str(globals().get("M2_PROTOTYPE_EMBEDDING_DEVICE", DEVICE if "DEVICE" in globals() else "cuda"))
M2_REUSE_EXISTING_PROTOTYPES = bool(globals().get("M2_REUSE_EXISTING_PROTOTYPES", True))
M2_REUSE_EXISTING_PROTOTYPE_CALIBRATION = bool(globals().get("M2_REUSE_EXISTING_PROTOTYPE_CALIBRATION", True))
M2_PROTOTYPE_MAX_IMAGES_PER_CLASS = globals().get("M2_PROTOTYPE_MAX_IMAGES_PER_CLASS", 50)
M2_PROTOTYPE_BANK = str(globals().get("M2_PROTOTYPE_BANK", "") or "")
M2_TAXONOMY_REGISTRY = str(globals().get("M2_TAXONOMY_REGISTRY", "") or "")
M2_PROTOTYPE_MIN_SIMILARITY = globals().get("M2_PROTOTYPE_MIN_SIMILARITY", None)
M2_PROTOTYPE_MIN_MARGIN = globals().get("M2_PROTOTYPE_MIN_MARGIN", None)
M2_PROTOTYPE_MIN_NEGATIVE_GAP = globals().get("M2_PROTOTYPE_MIN_NEGATIVE_GAP", None)
M2_AUTO_CALIBRATE_PROTOTYPE_RECONCILER = bool(
    globals().get(
        "M2_AUTO_CALIBRATE_PROTOTYPE_RECONCILER",
        globals().get("M2_AUTO_CALIBRATE_PROTOTYPES", True),
    )
)
M2_REQUIRE_CALIBRATED_PROTOTYPE_POLICY = bool(globals().get("M2_REQUIRE_CALIBRATED_PROTOTYPE_POLICY", True))
M2_PROTOTYPE_CALIBRATION_OUTPUT = str(
    globals().get("M2_PROTOTYPE_CALIBRATION_OUTPUT", ".runtime_tmp/router_prototype_calibration.json")
)
M2_PROTOTYPE_CALIBRATION_LIMIT = globals().get("M2_PROTOTYPE_CALIBRATION_LIMIT", None)
M2_PROTOTYPE_CALIBRATION_MIN_PRECISION = float(globals().get("M2_PROTOTYPE_CALIBRATION_MIN_PRECISION", 0.985))
M2_PROTOTYPE_CALIBRATION_MIN_COVERAGE = float(globals().get("M2_PROTOTYPE_CALIBRATION_MIN_COVERAGE", 0.80))
M2_PROTOTYPE_CALIBRATION_MAX_NEGATIVE_FALSE_ACCEPTS = int(
    globals().get("M2_PROTOTYPE_CALIBRATION_MAX_NEGATIVE_FALSE_ACCEPTS", 0)
)
M2_PROTOTYPE_CALIBRATION_MAX_NEGATIVE_FALSE_ACCEPT_RATE = float(
    globals().get("M2_PROTOTYPE_CALIBRATION_MAX_NEGATIVE_FALSE_ACCEPT_RATE", 0.05)
)
M2_PROTOTYPE_TARGET_MIN_PRECISION = float(globals().get("M2_PROTOTYPE_TARGET_MIN_PRECISION", 0.98))
M2_PROTOTYPE_TARGET_MAX_SUPPORTED_WRONG = globals().get("M2_PROTOTYPE_TARGET_MAX_SUPPORTED_WRONG", 1)
M2_PROTOTYPE_TARGET_CLASS_MIN_ACCEPTED = int(globals().get("M2_PROTOTYPE_TARGET_CLASS_MIN_ACCEPTED", 5))
M2_PROTOTYPE_TARGET_MAX_CROSS_PART_SUPPORTED_WRONG = int(
    globals().get("M2_PROTOTYPE_TARGET_MAX_CROSS_PART_SUPPORTED_WRONG", 0)
)
M2_PROTOTYPE_SIMILARITY_GRID = str(
    globals().get("M2_PROTOTYPE_SIMILARITY_GRID", "0.20,0.30,0.40,0.50,0.60,0.70")
)
M2_PROTOTYPE_MARGIN_GRID = str(globals().get("M2_PROTOTYPE_MARGIN_GRID", "0.00,0.02,0.04,0.06,0.08,0.10"))
M2_PROTOTYPE_NEGATIVE_GAP_GRID = str(
    globals().get("M2_PROTOTYPE_NEGATIVE_GAP_GRID", "0.00,0.02,0.04,0.06,0.08,0.10")
)
M2_ALLOW_NON_PLANT_FALSE_ACCEPTS = bool(globals().get("M2_ALLOW_NON_PLANT_FALSE_ACCEPTS", False))
M2_PROTOTYPE_TARGET_POLICY_NEGATIVE_MODE = str(
    globals().get("M2_PROTOTYPE_TARGET_POLICY_NEGATIVE_MODE", "none")
).strip().lower()
if M2_PROTOTYPE_TARGET_POLICY_NEGATIVE_MODE not in {"all", "none"}:
    M2_PROTOTYPE_TARGET_POLICY_NEGATIVE_MODE = "none"


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_calibration_constraints():
    return {
        "min_precision": M2_PROTOTYPE_CALIBRATION_MIN_PRECISION,
        "min_coverage": M2_PROTOTYPE_CALIBRATION_MIN_COVERAGE,
        "require_zero_non_plant_false_accepts": not M2_ALLOW_NON_PLANT_FALSE_ACCEPTS,
        "max_negative_false_accepts": M2_PROTOTYPE_CALIBRATION_MAX_NEGATIVE_FALSE_ACCEPTS,
        "max_negative_false_accept_rate": M2_PROTOTYPE_CALIBRATION_MAX_NEGATIVE_FALSE_ACCEPT_RATE,
        "target_min_precision": M2_PROTOTYPE_TARGET_MIN_PRECISION,
        "target_max_supported_wrong": int(M2_PROTOTYPE_TARGET_MAX_SUPPORTED_WRONG),
        "target_max_cross_part_supported_wrong": M2_PROTOTYPE_TARGET_MAX_CROSS_PART_SUPPORTED_WRONG,
        "target_policy_negative_mode": M2_PROTOTYPE_TARGET_POLICY_NEGATIVE_MODE,
        "target_class_min_accepted": M2_PROTOTYPE_TARGET_CLASS_MIN_ACCEPTED,
        "promotion_mode": "prototype_override",
    }


def _same_scalar(left, right):
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) is bool(right)
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        try:
            return abs(float(left) - float(right)) < 1e-9
        except (TypeError, ValueError):
            return False
    return str(left) == str(right)


def _constraints_match(payload):
    constraints = payload.get("constraints") if isinstance(payload, dict) else {}
    if not isinstance(constraints, dict):
        return False
    expected = _expected_calibration_constraints()
    return all(_same_scalar(constraints.get(key), value) for key, value in expected.items())


def _summary_manifest_sha(candidate_dir):
    summary_path = candidate_dir / "summary.json"
    if not summary_path.is_file():
        return ""
    try:
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return str(summary_payload.get("manifest_sha256") or "")


def _path_matches_repo_suffix(saved_path, current_path, repo_root):
    try:
        current_rel = current_path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return False
    normalized_saved = str(saved_path or "").replace("\\", "/")
    return normalized_saved == current_rel or normalized_saved.endswith(f"/{current_rel}")


def _copy_reusable_calibration_if_available(repo_root, manifest_path, prototype_bank_path, output_path):
    if not M2_REUSE_EXISTING_PROTOTYPE_CALIBRATION:
        return None
    if not manifest_path.is_file() or not prototype_bank_path.is_file():
        return None
    manifest_sha256 = _sha256_file(manifest_path)
    prototype_sha256 = _sha256_file(prototype_bank_path)
    for candidate_dir in sorted((repo_root / M2_REPO_RESULTS_ROOT).glob("*"), reverse=True):
        calibration_path = candidate_dir / "router_prototype_calibration.json"
        if not calibration_path.is_file():
            continue
        try:
            payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not _constraints_match(payload):
            continue
        candidate_manifest_sha = str(payload.get("manifest_sha256") or _summary_manifest_sha(candidate_dir))
        if candidate_manifest_sha and candidate_manifest_sha != manifest_sha256:
            continue
        if not candidate_manifest_sha and not _path_matches_repo_suffix(payload.get("manifest"), manifest_path, repo_root):
            continue
        candidate_prototype_sha = str(payload.get("prototype_bank_sha256") or "")
        co_located_prototype = candidate_dir / "prototype_bank.json"
        if not candidate_prototype_sha and co_located_prototype.is_file():
            candidate_prototype_sha = _sha256_file(co_located_prototype)
        if candidate_prototype_sha and candidate_prototype_sha != prototype_sha256:
            continue
        if not candidate_prototype_sha and not _path_matches_repo_suffix(
            payload.get("prototype_bank"),
            prototype_bank_path,
            repo_root,
        ):
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(calibration_path, output_path)
        return calibration_path
    return None


def _discard_stale_calibration(output_path):
    try:
        output_path.unlink()
    except FileNotFoundError:
        pass


if not M2_RUN_FULL_DEMO:
    m2_demo_result = None
    m2_demo_publish_report = {"enabled": False, "pushed": False, "reason": "M2_RUN_FULL_DEMO=False"}
    m2_demo_disconnect_report = {"ready": False, "missing": ["m2_full_demo_skipped"]}
    print("[M2] Full demo manifest run skipped because M2_RUN_FULL_DEMO=False.")
else:
    m2_run_started_at = datetime.now(timezone.utc)
    m2_run_start_perf = time.perf_counter()
    cell_script_root = Path(str(globals().get("__notebook_cell_script_root__", ""))).resolve()
    repo_root = cell_script_root.parents[1] if cell_script_root.name == "notebook_cells" else Path.cwd().resolve()

    manifest_path = (repo_root / M2_DEMO_MANIFEST).resolve()
    output_path = (repo_root / M2_DEMO_OUTPUT).resolve()
    markdown_output_path = (repo_root / M2_DEMO_MARKDOWN_OUTPUT).resolve()
    analysis_output_path = (repo_root / M2_ANALYSIS_OUTPUT).resolve()
    analysis_markdown_output_path = (repo_root / M2_ANALYSIS_MARKDOWN_OUTPUT).resolve()
    prototype_calibration_output_path = (repo_root / M2_PROTOTYPE_CALIBRATION_OUTPUT).resolve()
    adapter_root_path = Path(str(globals().get("ADAPTER_ROOT") or "runs"))
    if (
        M2_ENABLE_PROTOTYPE_RECONCILER
        and M2_REUSE_EXISTING_PROTOTYPES
        and (not M2_PROTOTYPE_BANK or not M2_TAXONOMY_REGISTRY)
    ):
        for candidate_dir in sorted((repo_root / M2_REPO_RESULTS_ROOT).glob("*"), reverse=True):
            prototype_bank_candidate = candidate_dir / "prototype_bank.json"
            taxonomy_registry_candidate = candidate_dir / "taxonomy_registry.json"
            if prototype_bank_candidate.is_file() and taxonomy_registry_candidate.is_file():
                if not M2_PROTOTYPE_BANK:
                    M2_PROTOTYPE_BANK = str(prototype_bank_candidate.relative_to(repo_root))
                if not M2_TAXONOMY_REGISTRY:
                    M2_TAXONOMY_REGISTRY = str(taxonomy_registry_candidate.relative_to(repo_root))
                print(f"[M2] Reusing existing prototype artifacts from {candidate_dir.relative_to(repo_root)}.")
                break

    if M2_ENABLE_PROTOTYPE_RECONCILER and M2_AUTO_BUILD_PROTOTYPES and (
        not M2_PROTOTYPE_BANK or not M2_TAXONOMY_REGISTRY
    ):
        prototype_run_id = M2_PROTOTYPE_RUN_ID or f"m2_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        prototype_command = [
            sys.executable,
            str(repo_root / "scripts" / "build_router_prototype_bank.py"),
            "--run-id",
            prototype_run_id,
            "--embedding-backend",
            M2_PROTOTYPE_EMBEDDING_BACKEND,
            "--embedding-model-id",
            M2_PROTOTYPE_EMBEDDING_MODEL_ID,
            "--embedding-device",
            M2_PROTOTYPE_EMBEDDING_DEVICE,
        ]
        if M2_PROTOTYPE_MAX_IMAGES_PER_CLASS is not None:
            prototype_command.extend(["--max-images-per-class", str(int(M2_PROTOTYPE_MAX_IMAGES_PER_CLASS))])
        print("[M2] Building router prototype artifacts before manifest run.")
        prototype_completed = subprocess.run(prototype_command, cwd=repo_root, check=False)
        print(f"[M2] prototype_builder_exit_code={prototype_completed.returncode}")
        if prototype_completed.returncode != 0:
            if M2_REQUIRE_CALIBRATED_PROTOTYPE_POLICY:
                M2_ENABLE_PROTOTYPE_RECONCILER = False
            _discard_stale_calibration(prototype_calibration_output_path)
            print("[M2] Prototype builder failed; prototype reconciler disabled for this run.")
        prototype_dir = repo_root / "runs" / "_index" / "router_prototypes" / prototype_run_id
        if not M2_PROTOTYPE_BANK:
            M2_PROTOTYPE_BANK = str((prototype_dir / "prototype_bank.json").relative_to(repo_root))
        if not M2_TAXONOMY_REGISTRY:
            M2_TAXONOMY_REGISTRY = str((prototype_dir / "taxonomy_registry.json").relative_to(repo_root))

    prototype_calibration_selected = False
    prototype_target_policy_selected = False
    if M2_ENABLE_PROTOTYPE_RECONCILER and M2_AUTO_CALIBRATE_PROTOTYPE_RECONCILER and M2_PROTOTYPE_BANK:
        prototype_bank_path = (repo_root / M2_PROTOTYPE_BANK).resolve()
        reused_calibration_path = _copy_reusable_calibration_if_available(
            repo_root,
            manifest_path,
            prototype_bank_path,
            prototype_calibration_output_path,
        )
        if reused_calibration_path is not None:
            print(
                "[M2] Reusing existing prototype calibration from "
                f"{reused_calibration_path.relative_to(repo_root)}."
            )
        else:
            calibration_command = [
                sys.executable,
                str(repo_root / "scripts" / "calibrate_router_prototype_reconciler.py"),
                "--manifest",
                str(manifest_path),
                "--prototype-bank",
                str(prototype_bank_path),
                "--output",
                str(prototype_calibration_output_path),
                "--min-precision",
                str(M2_PROTOTYPE_CALIBRATION_MIN_PRECISION),
                "--min-coverage",
                str(M2_PROTOTYPE_CALIBRATION_MIN_COVERAGE),
                "--max-negative-false-accepts",
                str(M2_PROTOTYPE_CALIBRATION_MAX_NEGATIVE_FALSE_ACCEPTS),
                "--max-negative-false-accept-rate",
                str(M2_PROTOTYPE_CALIBRATION_MAX_NEGATIVE_FALSE_ACCEPT_RATE),
                "--target-min-precision",
                str(M2_PROTOTYPE_TARGET_MIN_PRECISION),
                "--target-max-supported-wrong",
                str(int(M2_PROTOTYPE_TARGET_MAX_SUPPORTED_WRONG)),
                "--target-max-cross-part-supported-wrong",
                str(M2_PROTOTYPE_TARGET_MAX_CROSS_PART_SUPPORTED_WRONG),
                "--target-class-min-accepted",
                str(int(M2_PROTOTYPE_TARGET_CLASS_MIN_ACCEPTED)),
                "--similarity-grid",
                M2_PROTOTYPE_SIMILARITY_GRID,
                "--margin-grid",
                M2_PROTOTYPE_MARGIN_GRID,
                "--negative-gap-grid",
                M2_PROTOTYPE_NEGATIVE_GAP_GRID,
                "--target-policy-negative-mode",
                M2_PROTOTYPE_TARGET_POLICY_NEGATIVE_MODE,
            ]
            if M2_ALLOW_NON_PLANT_FALSE_ACCEPTS:
                calibration_command.append("--allow-non-plant-false-accepts")
            if M2_PROTOTYPE_CALIBRATION_LIMIT is not None:
                calibration_command.extend(["--limit", str(int(M2_PROTOTYPE_CALIBRATION_LIMIT))])
            print("[M2] Calibrating prototype reconciler thresholds.")
            calibration_completed = subprocess.run(calibration_command, cwd=repo_root, check=False)
            print(f"[M2] prototype_calibration_exit_code={calibration_completed.returncode}")
            if calibration_completed.returncode != 0:
                if M2_REQUIRE_CALIBRATED_PROTOTYPE_POLICY:
                    M2_ENABLE_PROTOTYPE_RECONCILER = False
                print("[M2] Prototype calibration failed; ignoring any stale calibration output for this run.")
                _discard_stale_calibration(prototype_calibration_output_path)
        if prototype_calibration_output_path.is_file():
            calibration_payload = json.loads(prototype_calibration_output_path.read_text(encoding="utf-8"))
            selected_policy = calibration_payload.get("selected_policy")
            target_policies = calibration_payload.get("target_policies")
            prototype_calibration_selected = isinstance(selected_policy, dict)
            prototype_target_policy_selected = (
                any(
                    isinstance(entry, dict) and isinstance(entry.get("selected_policy"), dict)
                    or (
                        isinstance(entry, dict)
                        and any(
                            isinstance(class_entry, dict) and isinstance(class_entry.get("selected_policy"), dict)
                            for class_entry in (
                                entry.get("class_policies")
                                if isinstance(entry.get("class_policies"), dict)
                                else {}
                            ).values()
                        )
                    )
                    for entry in target_policies.values()
                )
                if isinstance(target_policies, dict)
                else False
            )
            if prototype_calibration_selected:
                if M2_PROTOTYPE_MIN_SIMILARITY is None:
                    M2_PROTOTYPE_MIN_SIMILARITY = selected_policy.get("min_similarity")
                if M2_PROTOTYPE_MIN_MARGIN is None:
                    M2_PROTOTYPE_MIN_MARGIN = selected_policy.get("min_margin")
                if M2_PROTOTYPE_MIN_NEGATIVE_GAP is None:
                    M2_PROTOTYPE_MIN_NEGATIVE_GAP = selected_policy.get("min_negative_gap")
                print("[M2] Prototype calibration selected policy:")
                print(json.dumps(selected_policy, indent=2, ensure_ascii=False))
            else:
                print("[M2] Prototype calibration did not select a runtime policy.")
            if prototype_target_policy_selected:
                print("[M2] Prototype calibration selected at least one target-specific policy.")
        if M2_REQUIRE_CALIBRATED_PROTOTYPE_POLICY and not (
            prototype_calibration_selected or prototype_target_policy_selected
        ):
            M2_ENABLE_PROTOTYPE_RECONCILER = False
            print("[M2] Prototype reconciler disabled because no calibrated policy was selected.")

    command = [
        sys.executable,
        str(repo_root / "scripts" / "run_demo_checklist.py"),
        "--no-checklist",
        "--extra-manifest",
        str(manifest_path),
        "--device",
        str(DEVICE),
        "--config-env",
        str(CONFIG_ENV),
        "--adapter-root",
        str(adapter_root_path),
        "--output",
        str(output_path),
        "--markdown-output",
        str(markdown_output_path),
        "--analysis-output",
        str(analysis_output_path),
        "--analysis-markdown-output",
        str(analysis_markdown_output_path),
        "--batch-size",
        str(max(1, int(M2_BATCH_SIZE))),
        "--adapter-batch-size",
        str(max(1, int(M2_ADAPTER_BATCH_SIZE))),
        "--handoff-cache",
        str((repo_root / M2_HANDOFF_CACHE).resolve()),
    ]
    if M2_DEMO_LIMIT is not None:
        command.extend(["--limit", str(int(M2_DEMO_LIMIT))])
    if M2_STOP_ON_DEPENDENCY_BLOCKER:
        command.append("--stop-on-dependency-blocker")
    if M2_REFRESH_HANDOFF_CACHE:
        command.append("--refresh-handoff-cache")
    if M2_ENABLE_PROTOTYPE_RECONCILER:
        command.append("--enable-prototype-reconciler")
        if M2_PROTOTYPE_BANK:
            command.extend(["--prototype-bank", str((repo_root / M2_PROTOTYPE_BANK).resolve())])
        if M2_TAXONOMY_REGISTRY:
            command.extend(["--taxonomy-registry", str((repo_root / M2_TAXONOMY_REGISTRY).resolve())])
        if M2_PROTOTYPE_MIN_SIMILARITY is not None:
            command.extend(["--prototype-min-similarity", str(float(M2_PROTOTYPE_MIN_SIMILARITY))])
        if M2_PROTOTYPE_MIN_MARGIN is not None:
            command.extend(["--prototype-min-margin", str(float(M2_PROTOTYPE_MIN_MARGIN))])
        if M2_PROTOTYPE_MIN_NEGATIVE_GAP is not None:
            command.extend(["--prototype-min-negative-gap", str(float(M2_PROTOTYPE_MIN_NEGATIVE_GAP))])
        if prototype_calibration_output_path.is_file():
            command.extend(["--prototype-calibration-report", str(prototype_calibration_output_path)])

    print(f"[M2] repo_root={repo_root}")
    print(f"[M2] manifest={manifest_path}")
    print(f"[M2] output={output_path}")
    print(f"[M2] markdown_output={markdown_output_path}")
    print(f"[M2] analysis_output={analysis_output_path}")
    print(f"[M2] analysis_markdown_output={analysis_markdown_output_path}")
    print(f"[M2] handoff_cache={(repo_root / M2_HANDOFF_CACHE).resolve()}")
    print(f"[M2] prototype_reconciler={M2_ENABLE_PROTOTYPE_RECONCILER}")
    if M2_ENABLE_PROTOTYPE_RECONCILER:
        print(f"[M2] prototype_bank={M2_PROTOTYPE_BANK or 'missing'}")
        print(f"[M2] taxonomy_registry={M2_TAXONOMY_REGISTRY or 'missing'}")
        print(f"[M2] prototype_calibration={prototype_calibration_output_path}")
    print("[M2] Starting full manifest run. This can take a while on 512 images.")

    runner_started_at = datetime.now(timezone.utc)
    runner_start_perf = time.perf_counter()
    completed = subprocess.run(command, cwd=repo_root, check=False)
    runner_finished_at = datetime.now(timezone.utc)
    runner_elapsed_seconds = time.perf_counter() - runner_start_perf
    m2_run_finished_at = runner_finished_at
    m2_run_elapsed_seconds = time.perf_counter() - m2_run_start_perf
    print(f"[M2] runner_exit_code={completed.returncode}")
    print(f"[M2] runner_elapsed={format_elapsed_seconds(runner_elapsed_seconds)}")

    m2_demo_result = None
    report_ready = False
    if output_path.is_file():
        m2_demo_result = json.loads(output_path.read_text(encoding="utf-8"))
        report_ready = True
        print("[M2] Summary:")
        print(json.dumps(m2_demo_result.get("summary", {}), indent=2, ensure_ascii=False))
    else:
        print("[M2] Output report was not written. Check the cell log above.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    repo_results_rel = Path(M2_REPO_RESULTS_ROOT) / stamp
    repo_results_dir = repo_root / repo_results_rel
    m2_demo_publish_report = {"enabled": bool(M2_AUTO_PUSH_RESULTS), "pushed": False}
    m2_comparison_report = {
        "baseline": M2_COMPARISON_BASELINE,
        "enabled": bool(M2_COMPARISON_BASELINE),
        "written": False,
        "status": "not_run",
        "checks": {},
    }

    if report_ready:
        repo_results_dir.mkdir(parents=True, exist_ok=True)
        copied_paths = []
        provenance_paths = []
        if prototype_calibration_output_path.is_file():
            provenance_paths.append(prototype_calibration_output_path)
        if M2_PROTOTYPE_BANK:
            prototype_bank_path = (repo_root / M2_PROTOTYPE_BANK).resolve()
            if prototype_bank_path.is_file():
                provenance_paths.append(prototype_bank_path)
                prototype_summary_path = prototype_bank_path.parent / "summary.md"
                if prototype_summary_path.is_file():
                    provenance_paths.append(prototype_summary_path)
        if M2_TAXONOMY_REGISTRY:
            taxonomy_registry_path = (repo_root / M2_TAXONOMY_REGISTRY).resolve()
            if taxonomy_registry_path.is_file():
                provenance_paths.append(taxonomy_registry_path)
        for source_path in (
            output_path,
            markdown_output_path,
            analysis_output_path,
            analysis_markdown_output_path,
            *provenance_paths,
        ):
            if source_path.is_file():
                destination = repo_results_dir / source_path.name
                shutil.copy2(source_path, destination)
                copied_paths.append(destination.relative_to(repo_root).as_posix())

        summary_path = repo_results_dir / "summary.json"
        summary_payload = {
            "created_at": stamp,
            "started_at": m2_run_started_at.isoformat(),
            "finished_at": m2_run_finished_at.isoformat(),
            "elapsed_seconds": m2_run_elapsed_seconds,
            "elapsed_human": format_elapsed_seconds(m2_run_elapsed_seconds),
            "runner_started_at": runner_started_at.isoformat(),
            "runner_finished_at": runner_finished_at.isoformat(),
            "runner_elapsed_seconds": runner_elapsed_seconds,
            "runner_elapsed_human": format_elapsed_seconds(runner_elapsed_seconds),
            "runner_exit_code": int(completed.returncode),
            "manifest": str(manifest_path.relative_to(repo_root)),
            "batch_size": int(max(1, M2_BATCH_SIZE)),
            "adapter_batch_size": int(max(1, M2_ADAPTER_BATCH_SIZE)),
            "handoff_cache": {
                "path": str(M2_HANDOFF_CACHE),
                "refresh": bool(M2_REFRESH_HANDOFF_CACHE),
            },
            "output": str(output_path.relative_to(repo_root)),
            "markdown_output": str(markdown_output_path.relative_to(repo_root)),
            "analysis_output": str(analysis_output_path.relative_to(repo_root)),
            "analysis_markdown_output": str(analysis_markdown_output_path.relative_to(repo_root)),
            "prototype_reconciler": {
                "enabled": bool(M2_ENABLE_PROTOTYPE_RECONCILER),
                "reuse_existing_prototypes": bool(M2_REUSE_EXISTING_PROTOTYPES),
                "reuse_existing_prototype_calibration": bool(M2_REUSE_EXISTING_PROTOTYPE_CALIBRATION),
                "auto_build_prototypes": bool(M2_AUTO_BUILD_PROTOTYPES),
                "prototype_max_images_per_class": M2_PROTOTYPE_MAX_IMAGES_PER_CLASS,
                "auto_calibrate": bool(M2_AUTO_CALIBRATE_PROTOTYPE_RECONCILER),
                "require_calibrated_policy": bool(M2_REQUIRE_CALIBRATED_PROTOTYPE_POLICY),
                "prototype_bank": M2_PROTOTYPE_BANK,
                "taxonomy_registry": M2_TAXONOMY_REGISTRY,
                "prototype_calibration_output": str(prototype_calibration_output_path.relative_to(repo_root))
                if prototype_calibration_output_path.is_file()
                else "",
                "prototype_min_similarity": M2_PROTOTYPE_MIN_SIMILARITY,
                "prototype_min_margin": M2_PROTOTYPE_MIN_MARGIN,
                "prototype_min_negative_gap": M2_PROTOTYPE_MIN_NEGATIVE_GAP,
                "calibration_selected_policy": bool(prototype_calibration_selected),
                "calibration_selected_target_policy": bool(prototype_target_policy_selected),
                "target_min_precision": M2_PROTOTYPE_TARGET_MIN_PRECISION,
                "target_max_supported_wrong": M2_PROTOTYPE_TARGET_MAX_SUPPORTED_WRONG,
                "target_max_cross_part_supported_wrong": M2_PROTOTYPE_TARGET_MAX_CROSS_PART_SUPPORTED_WRONG,
                "target_class_min_accepted": M2_PROTOTYPE_TARGET_CLASS_MIN_ACCEPTED,
                "target_policy_negative_mode": M2_PROTOTYPE_TARGET_POLICY_NEGATIVE_MODE,
            },
            "copied_artifacts": copied_paths,
            "summary": m2_demo_result.get("summary", {}) if isinstance(m2_demo_result, dict) else {},
            "analysis_summary": m2_demo_result.get("analysis_summary", {}) if isinstance(m2_demo_result, dict) else {},
        }
        enrich_summary_manifest_sha256(summary_payload, repo_root=repo_root)
        summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        copied_paths.append(summary_path.relative_to(repo_root).as_posix())
        summary_payload["copied_artifacts"] = copied_paths
        comparison_baseline_path = (repo_root / M2_COMPARISON_BASELINE).resolve() if M2_COMPARISON_BASELINE else None
        if comparison_baseline_path and comparison_baseline_path.is_file():
            comparison_path = repo_results_dir / "m2_result_comparison.json"
            comparison_markdown_path = repo_results_dir / "m2_result_comparison.md"
            baseline_payload = enrich_summary_manifest_sha256(
                json.loads(comparison_baseline_path.read_text(encoding="utf-8")),
                repo_root=repo_root,
            )
            comparison_payload = compare_results(
                baseline=baseline_payload,
                candidate=summary_payload,
            )
            comparison_path.write_text(json.dumps(comparison_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            comparison_markdown_path.write_text(comparison_markdown(comparison_payload), encoding="utf-8")
            copied_paths.append(comparison_path.relative_to(repo_root).as_posix())
            copied_paths.append(comparison_markdown_path.relative_to(repo_root).as_posix())
            summary_payload["comparison"] = {
                "baseline": str(comparison_baseline_path.relative_to(repo_root)),
                "output": str(comparison_path.relative_to(repo_root)),
                "markdown_output": str(comparison_markdown_path.relative_to(repo_root)),
                "status": comparison_payload.get("status"),
                "checks": comparison_payload.get("checks", {}),
            }
            m2_comparison_report = {
                "baseline": str(comparison_baseline_path.relative_to(repo_root)),
                "enabled": True,
                "written": True,
                "output": str(comparison_path.relative_to(repo_root)),
                "markdown_output": str(comparison_markdown_path.relative_to(repo_root)),
                "status": comparison_payload.get("status"),
                "checks": comparison_payload.get("checks", {}),
            }
            summary_payload["copied_artifacts"] = copied_paths
        elif comparison_baseline_path:
            m2_comparison_report = {
                "baseline": M2_COMPARISON_BASELINE,
                "enabled": True,
                "written": False,
                "status": "baseline_missing",
                "checks": {},
            }
        summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print("[M2] Repo result copy:")
        print(json.dumps(copied_paths, indent=2, ensure_ascii=False))

        if M2_AUTO_PUSH_RESULTS:
            try:
                m2_demo_publish_report = push_repo_paths_to_github(
                    repo_root=repo_root,
                    relative_paths=[repo_results_rel.as_posix()],
                    remote_name=M2_AUTO_PUSH_REMOTE_NAME,
                    branch=M2_AUTO_PUSH_BRANCH,
                    commit_message=f"Add M2 demo results {stamp}",
                    print_fn=print,
                )
            except Exception as exc:
                m2_demo_publish_report = {
                    "enabled": True,
                    "pushed": False,
                    "paths": [repo_results_rel.as_posix()],
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
                print(f"[GIT] M2 result auto-push failed: {m2_demo_publish_report['error']}")
        else:
            m2_demo_publish_report = {
                "enabled": False,
                "pushed": False,
                "paths": [repo_results_rel.as_posix()],
            }
    else:
        m2_demo_publish_report = {
            "enabled": bool(M2_AUTO_PUSH_RESULTS),
            "pushed": False,
            "error": "M2 output report was not written.",
        }

    push_done = bool(
        m2_demo_publish_report.get("pushed")
        or (
            m2_demo_publish_report.get("enabled")
            and not m2_demo_publish_report.get("error")
            and m2_demo_publish_report.get("staged_files") == []
        )
    )
    comparison_required = bool(m2_comparison_report.get("enabled"))
    comparison_written = bool(m2_comparison_report.get("written")) if comparison_required else True
    comparison_passed = m2_comparison_report.get("status") == "pass" if comparison_required else True
    completion_checks = {
        "m2_report_written": bool(report_ready),
        "git_push": bool(push_done),
        "m2_comparison_written": bool(comparison_written),
        "m2_comparison_passed": bool(comparison_passed),
    }
    m2_completion_report = {
        "ready": bool(report_ready and push_done and comparison_written),
        "checks": completion_checks,
        "missing": [
            name
            for name in ("m2_report_written", "git_push", "m2_comparison_written")
            if not completion_checks[name]
        ],
        "soft_missing": [] if completion_checks["m2_comparison_passed"] else ["m2_comparison_passed"],
        "comparison": m2_comparison_report,
    }
    print(f"[COLAB] M2 completion checks -> {m2_completion_report['checks']}")
    m2_demo_disconnect_report = maybe_auto_disconnect_colab_runtime(
        enabled=bool(M2_AUTO_DISCONNECT_RUNTIME),
        grace_period_sec=M2_AUTO_DISCONNECT_GRACE_SECONDS,
        completion_report=m2_completion_report,
        print_fn=print,
    )

m2_demo_result
