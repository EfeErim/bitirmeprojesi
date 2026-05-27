#!/usr/bin/env python3
"""Validate the maintained notebook support surfaces."""

from __future__ import annotations

import builtins
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _safe_print(*args, **kwargs):
    try:
        builtins.print(*args, **kwargs)
    except UnicodeEncodeError:
        converted = [str(a).encode("ascii", errors="replace").decode("ascii") for a in args]
        builtins.print(*converted, **kwargs)


print = _safe_print

PARAMETER_CAPTURE = 'with TELEMETRY.capture_cell_output("Cell 3: Parameters"):'
ACCESS_CHECK_CAPTURE = (
    'with TELEMETRY.capture_cell_output("Cell 3b: Guncelleme ve Erisim Kontrolu"):'
)
REPO_BOOTSTRAP_REQUIRED = (
    "from pathlib import Path",
    "CLONE_TARGET = Path('/content/bitirmeprojesi')",
    "REPO_URL = os.environ.get('AADS_REPO_URL'",
    "['git', 'clone', '--depth', '1'",
)
UPDATE_CHECK_REQUIRED = (
    "repo_root_for_update_check = _ensure_repo_root_for_update_check()",
    "def _build_repo_access_url(",
    "from scripts.colab_repo_bootstrap import probe_repo_update_status",
    "[KONTROL] Ilk hucre:",
)
DRIVE_REPO_BOOTSTRAP_FORBIDDEN = (
    "Path('/content/drive/MyDrive/bitirme projesi')",
    "Path('/content/drive/MyDrive/bitirmeprojesi')",
    "def _mount_drive_inline()",
    "mount_drive_if_available",
)


def _assert_contains(source: str, snippet: str, message: str) -> None:
    assert snippet in source, message.format(snippet=snippet)


def _assert_not_contains(source: str, snippet: str, message: str) -> None:
    assert snippet not in source, message.format(snippet=snippet)


def _assert_contains_all(source: str, snippets: tuple[str, ...], message: str) -> None:
    for snippet in snippets:
        _assert_contains(source, snippet, message)


def _assert_not_contains_all(source: str, snippets: tuple[str, ...], message: str) -> None:
    for snippet in snippets:
        _assert_not_contains(source, snippet, message)


@dataclass(frozen=True)
class NotebookSources:
    notebook_path: Path
    code_cells: tuple[str, ...]
    full_source: str
    first_code_source: str


def _load_notebook_sources_from_path(notebook_path: Path) -> NotebookSources:
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    cell_runner_pattern = re.compile(r"run_cell_script\((['\"])(?P<name>[^'\"]+)\1,\s*globals\(\)\)")

    def _expand_cell_source(source: str) -> str:
        matches = tuple(cell_runner_pattern.finditer(source))
        if not matches:
            return source
        expanded_parts = [source]
        for match in matches:
            script_path = ROOT / "scripts" / "notebook_cells" / match.group("name")
            assert script_path.is_file(), f"Notebook cell script was not found: {script_path}"
            expanded_parts.append(script_path.read_text(encoding="utf-8"))
        return "\n".join(expanded_parts)

    code_cells = tuple(
        _expand_cell_source("".join(cell.get("source", [])))
        for cell in payload.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    assert code_cells, f"{notebook_path.name} code cells were not found"
    return NotebookSources(
        notebook_path=notebook_path,
        code_cells=code_cells,
        full_source="\n\n".join(code_cells),
        first_code_source=code_cells[0],
    )


def _load_notebook_sources(notebook_name: str) -> NotebookSources:
    return _load_notebook_sources_from_path(ROOT / "colab_notebooks" / notebook_name)


def _find_code_cell_source(sources: NotebookSources, marker: str, missing_message: str) -> str:
    for source in sources.code_cells:
        if marker in source:
            return source
    raise AssertionError(missing_message)


def _assert_code_cells_compile(sources: NotebookSources, notebook_label: str) -> None:
    for index, source in enumerate(sources.code_cells, start=1):
        try:
            compile(source, f"{sources.notebook_path}:code_cell_{index}", "exec")
        except SyntaxError as exc:
            raise AssertionError(
                f"{notebook_label} code cell {index} has invalid Python syntax: "
                f"line {exc.lineno}, offset {exc.offset}: {exc.msg}"
            ) from exc


def _assert_repo_bootstrap_contract(first_code_source: str, notebook_label: str) -> None:
    _assert_contains(
        first_code_source,
        "def _ensure_aads_repo_on_path():",
        f"{notebook_label} first code cell should make repo scripts importable before runner import: {{snippet}}",
    )
    assert first_code_source.index("def _ensure_aads_repo_on_path():") < first_code_source.index(
        "from scripts.notebook_helpers.cell_script_runner import run_cell_script"
    ), f"{notebook_label} first code cell imports the cell runner before repo path bootstrap"
    _assert_contains_all(
        first_code_source,
        REPO_BOOTSTRAP_REQUIRED,
        f"{notebook_label} first code cell is missing required GitHub bootstrap: {{snippet}}",
    )
    assert first_code_source.index("from pathlib import Path") < first_code_source.index(
        "CLONE_TARGET = Path('/content/bitirmeprojesi')"
    ), f"{notebook_label} first code cell uses Path before importing it"
    _assert_not_contains_all(
        first_code_source,
        DRIVE_REPO_BOOTSTRAP_FORBIDDEN,
        f"{notebook_label} first code cell should not use Drive for repo bootstrap: {{snippet}}",
    )


def _assert_clone_bootstrap_contract(first_code_source: str, notebook_label: str) -> None:
    _assert_contains(
        first_code_source,
        "def _ensure_aads_repo_on_path():",
        f"{notebook_label} first code cell should define the clone bootstrap: {{snippet}}",
    )
    _assert_contains_all(
        first_code_source,
        (
            "from pathlib import Path",
            "DEFAULT_REPO_URL = 'https://github.com/EfeErim/bitirmeprojesi.git'",
            "REPO_URL = os.environ.get('AADS_REPO_URL', DEFAULT_REPO_URL)",
            "CLONE_TARGET = Path('/content/bitirmeprojesi')",
            "subprocess.run(",
            "'clone', '--depth', '1', '--branch', REPO_REF",
            "Notebook 4 repo ready:",
        ),
        f"{notebook_label} first code cell is missing required clone bootstrap: {{snippet}}",
    )
    assert first_code_source.index("from pathlib import Path") < first_code_source.index(
        "CLONE_TARGET = Path('/content/bitirmeprojesi')"
    ), f"{notebook_label} first code cell uses Path before importing it"
    _assert_not_contains_all(
        first_code_source,
        (
            "https://api.github.com/repos/",
            "DOWNLOAD_MANIFEST",
            "raw.githubusercontent.com",
            "manifest_text = response.read().decode('utf-8')",
            "urllib.request",
        ),
        f"{notebook_label} first code cell should clone instead of downloading raw source files: {{snippet}}",
    )


def _assert_update_check_contract(
    first_code_source: str,
    notebook_label: str,
    *,
    forbid_drive_bootstrap: bool,
) -> None:
    _assert_contains(
        first_code_source,
        "def _ensure_aads_repo_on_path():",
        f"{notebook_label} first code cell should make repo scripts importable before runner import: {{snippet}}",
    )
    assert first_code_source.index("def _ensure_aads_repo_on_path():") < first_code_source.index(
        "from scripts.notebook_helpers.cell_script_runner import run_cell_script"
    ), f"{notebook_label} first code cell imports the cell runner before repo path bootstrap"
    _assert_contains_all(
        first_code_source,
        UPDATE_CHECK_REQUIRED,
        f"{notebook_label} first code cell is missing required freshness check: {{snippet}}",
    )
    if forbid_drive_bootstrap:
        _assert_not_contains_all(
            first_code_source,
            DRIVE_REPO_BOOTSTRAP_FORBIDDEN,
            f"{notebook_label} first code cell should stay repo-first without Drive bootstrap: {{snippet}}",
        )


def gate_label(step_id: str, name: str) -> str:
    return f"[{step_id}] {name}"


@dataclass(frozen=True)
class ValidationCheck:
    result_name: str
    step_id: str
    description: str
    success_message: str
    failure_prefix: str
    callback: Callable[[], None]
    requires_runtime_dependencies: bool = True


def _run_check(check: ValidationCheck, *, leading_newline: bool = False) -> bool:
    prefix = "\n" if leading_newline else ""
    print(f"{prefix}Testing {gate_label(check.step_id, check.description)}...")
    try:
        check.callback()
    except Exception as exc:
        detail = str(exc).strip()
        failure_message = check.failure_prefix if not detail else f"{check.failure_prefix}: {detail}"
        print(f"FAIL {gate_label(check.step_id, failure_message)}")
        return False

    print(f"PASS {gate_label(check.step_id, check.success_message)}")
    return True


def _check_runtime_dependencies() -> None:
    required = (
        "torch",
        "torchvision",
        "transformers",
        "peft",
        "accelerate",
        "huggingface_hub",
        "PIL",
    )
    missing = []
    for module_name in required:
        try:
            __import__(module_name)
        except Exception as exc:
            import logging
            logging.exception('Unhandled exception')
            raise
            missing.append(module_name)

    if missing:
        missing_display = ", ".join(sorted(missing))
        raise RuntimeError(
            f"Missing dependencies: {missing_display}. Install requirements.txt before running this validation."
        )


def test_config_surface() -> None:
    from src.core.config_manager import ConfigurationManager

    cfg = ConfigurationManager(config_dir=str(ROOT / "config"), environment="colab").load_all_configs()
    assert {"training", "router", "colab", "inference"} <= set(cfg.keys())


def test_continual_trainer_imports() -> None:
    from src.training.continual_sd_lora import ContinualSDLoRAConfig, ContinualSDLoRATrainer
    from src.training.session import ContinualTrainingSession
    from src.training.validation import evaluate_model
    from src.workflows.training import TrainingWorkflow

    config = ContinualSDLoRAConfig.from_training_config(
        {
            "backbone": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
            "adapter": {
                "target_modules_strategy": "all_linear_transformer",
                "lora_r": 4,
                "lora_alpha": 8,
            },
            "fusion": {"layers": [2, 5, 8, 11]},
            "ood": {"threshold_factor": 2.0},
            "device": "cpu",
        }
    )
    trainer = ContinualSDLoRATrainer(config)
    assert hasattr(trainer, "initialize_engine")
    assert hasattr(trainer, "add_classes")
    assert hasattr(trainer, "train_batch")
    assert hasattr(trainer, "snapshot_training_state")
    assert hasattr(trainer, "restore_training_state")
    assert hasattr(trainer, "save_adapter")
    assert hasattr(trainer, "load_adapter")
    assert ContinualTrainingSession is not None
    assert TrainingWorkflow is not None
    assert callable(evaluate_model)


def test_quantization_guard() -> None:
    from src.training.quantization import assert_no_prohibited_4bit_flags

    valid_payload = {
        "training": {
            "continual": {
                "adapter": {"target_modules_strategy": "all_linear_transformer"}
            }
        }
    }
    assert_no_prohibited_4bit_flags(valid_payload)

    rejected = False
    try:
        forbidden_key = "load_in_" + "4bit"
        assert_no_prohibited_4bit_flags({forbidden_key: True})
    except ValueError:
        rejected = True

    assert rejected, "4-bit payload was expected to be rejected"


def test_adapter_surface() -> None:
    from src.adapter.independent_crop_adapter import IndependentCropAdapter

    adapter = IndependentCropAdapter(crop_name="tomato", device="cpu")
    assert hasattr(adapter, "initialize_engine")
    assert hasattr(adapter, "add_classes")
    assert hasattr(adapter, "build_training_session")
    assert hasattr(adapter, "save_adapter")
    assert hasattr(adapter, "load_adapter")


def test_runtime_surface() -> None:
    from src.pipeline.router_adapter_runtime import RouterAdapterRuntime
    from src.workflows.inference import InferenceWorkflow
    from scripts.colab_auto_router_adapter_prediction import run_auto_router_adapter_prediction

    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {
                "continual": {
                    "backbone": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
                    "adapter": {"target_modules_strategy": "all_linear_transformer"},
                    "fusion": {"layers": [2, 5, 8, 11]},
                    "ood": {"threshold_factor": 2.0},
                }
            },
            "inference": {"adapter_root": "models/adapters", "target_size": 224},
        },
        device="cpu",
    )
    assert hasattr(runtime, "load_router")
    assert hasattr(runtime, "load_adapter")
    assert hasattr(runtime, "predict")
    assert InferenceWorkflow is not None
    assert callable(run_auto_router_adapter_prediction)


def test_auto_router_adapter_notebook_contract() -> None:
    import inspect

    from scripts.colab_auto_router_adapter_prediction import run_auto_router_adapter_prediction

    sources = _load_notebook_sources("8_auto_router_adapter_prediction.ipynb")
    helper_source = inspect.getsource(run_auto_router_adapter_prediction)

    _assert_code_cells_compile(sources, "Notebook 8")
    _assert_contains(
        sources.first_code_source,
        "def _ensure_aads_repo_on_path():",
        "Notebook 8 first code cell should bootstrap the repo before cell runner import: {snippet}",
    )
    _assert_contains(
        sources.first_code_source,
        "run_cell_script('nb1_cell01_bootstrap.py', globals())",
        "Notebook 8 should reuse Notebook 1 bootstrap cell script: {snippet}",
    )
    for script_name in (
        "nb1_cell02_access_check.py",
        "nb1_cell03_runtime_setup.py",
        "nb1_cell04_analysis.py",
        "nb8_cell05_adapter_prediction.py",
    ):
        _assert_contains(
            sources.full_source,
            f"run_cell_script('{script_name}', globals())",
            "Notebook 8 should stay a thin wrapper over maintained cell scripts: {snippet}",
        )
    _assert_contains(
        sources.full_source,
        "from scripts.colab_auto_router_adapter_prediction import run_auto_router_adapter_prediction",
        "Notebook 8 should use the maintained auto router-adapter helper: {snippet}",
    )
    full_prediction_cell = _find_code_cell_source(
        sources,
        "router_result = result",
        "Notebook 8 should have a full prediction cell that keeps the router result.",
    )
    _assert_contains(
        full_prediction_cell,
        "run_cell_script('nb1_cell04_analysis.py', globals())",
        "Notebook 8 full prediction cell should run Notebook 1 router analysis first: {snippet}",
    )
    _assert_contains(
        full_prediction_cell,
        "run_cell_script('nb8_cell05_adapter_prediction.py', globals())",
        "Notebook 8 full prediction cell should immediately load the adapter and predict: {snippet}",
    )
    assert full_prediction_cell.index("run_cell_script('nb1_cell04_analysis.py', globals())") < full_prediction_cell.index(
        "run_cell_script('nb8_cell05_adapter_prediction.py', globals())"
    ), "Notebook 8 should run router analysis before adapter prediction in the same cell"
    assert full_prediction_cell.index("run_cell_script('nb8_cell05_adapter_prediction.py', globals())") < full_prediction_cell.index(
        "auto_result"
    ), "Notebook 8 should return the adapter prediction result, not only the router target"
    _assert_contains(
        helper_source,
        "workflow_factory: WorkflowFactory = InferenceWorkflow",
        "Notebook 8 helper should call the canonical InferenceWorkflow: {snippet}",
    )
    _assert_contains(
        helper_source,
        "trust_crop_hint=True",
        "Notebook 8 helper should avoid duplicating Notebook 1 routing by using a trusted router handoff: {snippet}",
    )
    assert sources.full_source.count("run_inference(") == 1, (
        "Notebook 8 should inherit the single Notebook 1 router call, not add a second router-only implementation"
    )


def test_adapter_smoke_notebook_surface() -> None:
    from scripts.colab_adapter_smoke_test import (
        discover_adapter_candidates,
        load_adapter_summary,
        predict_image_folder,
        predict_single_image,
    )

    assert callable(discover_adapter_candidates)
    assert callable(load_adapter_summary)
    assert callable(predict_single_image)
    assert callable(predict_image_folder)


def test_adapter_smoke_notebook_bootstrap_contract() -> None:
    sources = _load_notebook_sources("3_validate_exported_adapter_directly.ipynb")

    _assert_repo_bootstrap_contract(sources.first_code_source, "Notebook 3")

    assert "Path('/content/drive/MyDrive/aads_ulora')" not in sources.full_source
    assert "SEARCH_ROOTS = [" in sources.full_source
    assert "INCLUDE_RUN_ADAPTERS = False" in sources.full_source
    assert "ROOT / 'outputs' / 'colab_notebook_training'" in sources.full_source
    assert "SEARCH_ROOTS.append(ROOT / 'runs')" in sources.full_source


def test_simple_adapter_smoke_notebook_bootstrap_contract() -> None:
    sources = _load_notebook_sources("4_simple_direct_adapter_test_ui.ipynb")

    _assert_code_cells_compile(sources, "Notebook 4")
    _assert_clone_bootstrap_contract(sources.first_code_source, "Notebook 4")
    assert "collect_notebook_access_report" in sources.full_source
    assert "install_colab_requirements(ROOT / 'colab_notebooks' / 'requirements_colab.txt', running_in_colab())" in sources.full_source
    assert "print_notebook_access_report" in sources.full_source
    assert "from scripts import colab_simple_adapter_smoke_ui" in sources.full_source
    assert "importlib.reload(colab_simple_adapter_smoke_ui)" in sources.full_source
    assert "['reset', '--hard', f'origin/{REPO_REF}']" in sources.full_source
    assert "['pull', '--ff-only', 'origin', REPO_REF]" not in sources.full_source
    assert "ROOT / 'outputs' / 'colab_notebook_training'" in sources.full_source
    assert "ROOT / 'models' / 'adapters'" in sources.full_source
    assert "ROOT / 'runs'" not in sources.full_source
    assert "show_all_adapters=True" not in sources.full_source
    assert "show_mirror_adapters=True" not in sources.full_source
    assert "launch_simple_adapter_smoke_ui(ROOT, search_roots=SEARCH_ROOTS)" in sources.full_source


def test_colab_helpers() -> None:
    from scripts.colab_checkpointing import TrainingCheckpointManager
    from scripts.colab_live_telemetry import ColabLiveTelemetry
    from scripts.colab_repo_bootstrap import (
        export_current_colab_notebook,
        mirror_checkpoint_state_to_repo,
        mirror_path_to_repo,
        push_repo_paths_to_github,
        push_repo_run_to_github,
    )
    from scripts.colab_simple_adapter_smoke_ui import launch_simple_adapter_smoke_ui
    from scripts.evaluate_dataset_layout import evaluate_layout
    from scripts.prepare_grouped_runtime_dataset import (
        build_grouped_dataset_plan,
        materialize_grouped_runtime_dataset,
        scan_class_root_dataset,
    )
    from scripts.prepare_materialization_dataset import prepare_class_root_for_materialization

    assert hasattr(ColabLiveTelemetry, "configure_repo_output_export")
    assert callable(export_current_colab_notebook)
    assert callable(mirror_checkpoint_state_to_repo)
    assert callable(mirror_path_to_repo)
    assert callable(push_repo_paths_to_github)
    assert callable(push_repo_run_to_github)
    assert callable(launch_simple_adapter_smoke_ui)
    assert callable(prepare_class_root_for_materialization)
    _ = (
        TrainingCheckpointManager,
        ColabLiveTelemetry,
        evaluate_layout,
        build_grouped_dataset_plan,
        materialize_grouped_runtime_dataset,
        scan_class_root_dataset,
    )


def test_data_prep_notebook_contract() -> None:
    sources = _load_notebook_sources("0_prepare_grouped_dataset_for_training.ipynb")
    bootstrap_source = _find_code_cell_source(
        sources,
        "from scripts.colab_live_telemetry import ColabLiveTelemetry",
        "Notebook 0 bootstrap cell was not found",
    )
    parameter_source = _find_code_cell_source(
        sources,
        PARAMETER_CAPTURE,
        "Notebook 0 parameter cell was not found",
    )
    access_check_source = _find_code_cell_source(
        sources,
        ACCESS_CHECK_CAPTURE,
        "Notebook 0 access-check cell was not found",
    )

    _assert_update_check_contract(
        sources.first_code_source,
        "Notebook 0",
        forbid_drive_bootstrap=True,
    )
    for snippet in (
        "RUN_ID =",
        "TELEMETRY = ColabLiveTelemetry(",
        "REPO_RUN_DIR =",
        "REPO_NOTEBOOK_OUTPUT_PATH =",
    ):
        assert snippet in bootstrap_source, f"Notebook 0 bootstrap is missing: {snippet}"
    for snippet in (
        "mount_drive_if_available",
        "def _mount_drive_inline()",
        "Path('/content/drive/MyDrive/bitirme projesi')",
        "Path('/content/drive/MyDrive/bitirmeprojesi')",
        "def _copy_path_to_drive_exports",
    ):
        _assert_not_contains(
            bootstrap_source,
            snippet,
            "Notebook 0 bootstrap should not mirror repo prep outputs through Drive: {snippet}",
        )
    for snippet in (
        "REPO_DATASET_ROOT =",
        'REPO_DATASET_NAME = ""',
        "DATASET_ROOT =",
        "IMPORT_FROM_DRIVE = False",
        "DRIVE_DATASET_PATH =",
        "DRIVE_DATASET_NAME =",
        "CROP_NAME =",
        "PART_NAME =",
    ):
        assert snippet in parameter_source, f"Notebook 0 parameter cell is missing: {snippet}"
    assert "IMPORT_FROM_DRIVE = FALSE" not in parameter_source
    for snippet in (
        "PREP_ARTIFACT_ROOT =",
        "PREPARED_RUNTIME_ROOT =",
        "OOD_DATASET_ROOT =",
        "OOD_DATASET_NAME =",
        "OOD_ROOT =",
        "ASK_FOR_OOD_ROOT =",
        "PREPARED_CLASS_ROOT =",
        "PREPARE_DATASET_FROM_REPORTS =",
        "MATERIALIZE_AFTER_REVIEW =",
        "INTERACTIVE_AUDIT_REVIEW =",
        "MAX_INTERACTIVE_REVIEW_ITEMS =",
        "SAVE_RUNTIME_DATASET_TO_GITHUB =",
        "RUNTIME_DATASET_PUSH_REMOTE_NAME =",
        "RUNTIME_DATASET_PUSH_BRANCH =",
        "CLEANUP_SEED =",
        "PREP_DINOV3_MODEL_ID =",
        "PREP_BIOCLIP_MODEL_ID =",
    ):
        assert snippet in bootstrap_source, f"Notebook 0 bootstrap is missing: {snippet}"
    assert "collect_notebook_access_report" in access_check_source
    assert "print_notebook_access_report" in access_check_source
    assert "build_grouped_dataset_plan" in sources.full_source
    assert "build_human_review_packet" in sources.full_source
    assert "format_human_review_packet" in sources.full_source
    assert "evaluate_layout(root=dataset_root)" in sources.full_source
    assert "ASK_FOR_OOD_ROOT" in sources.full_source
    assert "resolve_dataset_directory_from_parent" in sources.full_source
    assert "build_prepared_dataset_key" in sources.full_source
    assert "prepare_class_root_for_materialization" in sources.full_source
    assert "def _resolve_repo_dataset_root" in sources.full_source
    assert "resolve_repo_dataset_directory" in sources.full_source
    assert 'dataset_source = "drive" if IMPORT_FROM_DRIVE' in sources.full_source
    assert 'STATE["dataset_name"] = dataset_name' in sources.full_source
    assert 'STATE["dataset_source"] = dataset_source' in sources.full_source
    assert "MATERIALIZE_AFTER_REVIEW = True" in sources.full_source
    assert "materialize_grouped_runtime_dataset" in sources.full_source
    assert "push_repo_paths_to_github" in sources.full_source
    assert "runtime_dataset_push_report" in sources.full_source


def test_repo_dataset_scaffold() -> None:
    required_paths = (
        ROOT / "data" / "README.md",
        ROOT / "data" / "class_root_dataset" / ".gitkeep",
        ROOT / "data" / "ood_dataset" / ".gitkeep",
        ROOT / "data" / "prepared_class_root_datasets" / ".gitkeep",
        ROOT / "data" / "prepared_runtime_datasets" / ".gitkeep",
    )
    missing = [str(path.relative_to(ROOT)) for path in required_paths if not path.exists()]
    assert not missing, f"Missing dataset scaffold path(s): {', '.join(missing)}"


def test_training_notebook_dataset_contract_detection() -> None:
    sources = _load_notebook_sources("2_train_continual_sd_lora_adapter.ipynb")
    assert 'RUNTIME_DATASET_ROOT = "data/prepared_runtime_datasets"' in sources.full_source
    assert "Notebook 0'un yazdigi <dataset_key>/continual|val|test|ood yapisini tutan repo-ici root." in sources.full_source
    assert "from scripts.colab_dataset_layout import list_repo_dataset_directories, resolve_direct_repo_dataset_root, resolve_repo_relative_root" in sources.full_source
    assert "from scripts.colab_training_recommendations import inspect_runtime_dataset" in sources.full_source
    assert "inspect_runtime_dataset" in sources.full_source
    assert "resolve_notebook_params" in sources.full_source
    assert 'direct_runtime_dataset = resolve_direct_repo_dataset_root(' in sources.full_source
    assert 'STATE["runtime_dataset_key"] = selected_dataset_name' in sources.full_source
    assert "from src.data.loaders import create_training_loaders" in sources.full_source
    assert "src.utils.data_loader" not in sources.full_source
    assert "No prepared runtime datasets were found under RUNTIME_DATASET_ROOT. Notebook 0'u once calistirin." in sources.full_source
    assert "Prepared runtime dataset is missing split folder(s)" in sources.full_source
    assert 'STATE["resolved_ood_root"] = resolved_ood_root_value' in sources.full_source
    assert 'STATE["resolved_oe_root"] = resolved_oe_root_value' in sources.full_source
    assert 'STATE["dataset_inspection"] = dataset_inspection' in sources.full_source
    assert 'STATE["hardware_inspection"] = {}' in sources.full_source
    assert 'STATE["recommendation_report"] = {}' in sources.full_source
    assert 'STATE["recommendation_decision"] = "disabled"' in sources.full_source
    assert 'STATE["effective_params"] = effective_params' in sources.full_source
    assert "ASK_FOR_OOD_ROOT = True" in sources.full_source
    assert "ASK_FOR_OE_ROOT = True" in sources.full_source
    assert "OOD klasoru yolunu girin" in sources.full_source
    assert "OE klasoru yolunu girin" in sources.full_source
    assert "ood_root=resolved_ood_root or None" in sources.full_source
    assert "oe_root=resolved_oe_root or None" in sources.full_source
    assert "build_grouped_dataset_plan" not in sources.full_source
    assert "materialize_grouped_runtime_dataset" not in sources.full_source


def test_training_notebook_bootstrap_contract() -> None:
    sources = _load_notebook_sources("2_train_continual_sd_lora_adapter.ipynb")
    bootstrap_source = _find_code_cell_source(
        sources,
        "from scripts.colab_live_telemetry import ColabLiveTelemetry",
        "Notebook 2 bootstrap cell was not found",
    )
    parameter_source = _find_code_cell_source(
        sources,
        PARAMETER_CAPTURE,
        "Notebook 2 parameter cell was not found",
    )
    run_identity_source = _find_code_cell_source(
        sources,
        "# Notebook 2 calisma kimligi",
        "Notebook 2 run identity cell was not found",
    )
    access_check_source = _find_code_cell_source(
        sources,
        ACCESS_CHECK_CAPTURE,
        "Notebook 2 access-check cell was not found",
    )

    _assert_update_check_contract(
        sources.first_code_source,
        "Notebook 2",
        forbid_drive_bootstrap=True,
    )

    required_bootstrap_snippets = (
        "RUN_ID =",
        "TELEMETRY = ColabLiveTelemetry(",
        "LOCAL_TELEMETRY_ROOT = ROOT / 'outputs' / 'colab_notebook_training' / 'telemetry_runtime'",
        'exclude_dir_names=("checkpoints", "telemetry_runtime")',
        "CHECKPOINT_MANAGER =",
        "DEVICE =",
        "def rt(",
        "REPO_RUN_DIR =",
        "REPO_NOTEBOOK_OUTPUT_PATH =",
        "def save_run_outputs_to_repo()",
        "build_notebook_run_dir",
        "build_notebook_run_id",
    )
    missing = [snippet for snippet in required_bootstrap_snippets if snippet not in bootstrap_source]
    if missing:
        raise AssertionError(f"Notebook 2 bootstrap cell is missing required setup: {', '.join(missing)}")

    assert PARAMETER_CAPTURE in parameter_source
    assert sources.full_source.index("TELEMETRY = ColabLiveTelemetry(") < sources.full_source.index(
        PARAMETER_CAPTURE
    )
    assert sources.full_source.index("RUN_ID =") < sources.full_source.index("run_id = RUN_ID")
    assert sources.full_source.index("CHECKPOINT_MANAGER =") < sources.full_source.index('"checkpoint_manager": CHECKPOINT_MANAGER')
    assert 'PART_NAME = "unspecified"' in run_identity_source
    assert "collect_notebook_access_report" in access_check_source
    assert "print_notebook_access_report" in access_check_source
    assert "REPO_RUN_DIR = build_notebook_run_dir(ROOT, CROP_NAME, PART_NAME, RUN_ID)" in bootstrap_source

    required_parameter_snippets = (
        'PART_NAME = globals().get("PART_NAME", "unspecified")',
        'RUNTIME_DATASET_ROOT = "data/prepared_runtime_datasets"',
        'DATASET_NAME = ""',
        'OOD_ROOT = ""',
        'ASK_FOR_OOD_ROOT = True',
        'OE_ROOT = ""',
        'ASK_FOR_OE_ROOT = True',
        'OE_ENABLED = False',
        'OE_LOSS_WEIGHT = 0.5',
        'from scripts.notebook_helpers.adapter_recommendations import get_adapter_recs',
        'ADAPTER_RECS = get_adapter_recs()',
        'MANUAL_PARAM_OVERRIDES = {}',
        'EPOCHS = ',
        'BATCH_SIZE = ',
        'LEARNING_RATE = ',
        'LORA_R = ',
        'AUGMENTATION_POLICY = str(CONTINUAL_DATA_CFG.get("augmentation_policy", "randaugment")).strip().lower()',
        'RANDAUGMENT_NUM_OPS = int(CONTINUAL_DATA_CFG.get("randaugment_num_ops", 2))',
        'RANDAUGMENT_MAGNITUDE = int(CONTINUAL_DATA_CFG.get("randaugment_magnitude", 7))',
        'ALLOW_UNDER_MIN_TRAINING = False',
        'ALLOW_UNDER_MIN_TRAINING = bool(ALLOW_UNDER_MIN_TRAINING)',
        'BER_ENABLED = False',
        'LOSS_NAME = "logitnorm"',
        'LOGITNORM_TAU = 1.0',
        'OOD_FACTOR = ',
        'CHECKPOINT_EVERY_N_STEPS = ',
        'source=notebook_cell',
        'defaults=notebook_cell',
        'parameter_source": "notebook_cell"',
    )
    for snippet in required_parameter_snippets:
        _assert_contains(
            parameter_source,
            snippet,
            "Notebook 2 parameter cell is missing required direct parameter surface: {snippet}",
        )

    required_training_surface_snippets = (
        'optimization_cfg["loss_name"] = str(effective_params["LOSS_NAME"]).strip().lower()',
        'optimization_cfg["logitnorm_tau"] = float(effective_params["LOGITNORM_TAU"])',
        'data_cfg["augmentation_policy"] = str(effective_params.get("AUGMENTATION_POLICY", AUGMENTATION_POLICY))',
        'data_cfg["allow_under_min_training"] = bool(effective_params["ALLOW_UNDER_MIN_TRAINING"])',
        'augmentation_policy=str(effective_params.get("AUGMENTATION_POLICY", AUGMENTATION_POLICY))',
        'STATE["resolved_ood_root"] = resolved_ood_root_value',
        'STATE["resolved_oe_root"] = resolved_oe_root_value',
        'continual_cfg["ood"]["oe_enabled"] = bool(OE_ENABLED)',
        'continual_cfg["ood"]["oe_root"] = resolved_oe_root',
        'list_repo_dataset_directories',
        'resolve_direct_repo_dataset_root',
        'resolve_repo_relative_root',
        'inspect_runtime_dataset',
        'resolve_notebook_params',
        'STATE["recommendation_decision"] = "disabled"',
        'STATE["effective_params"] = effective_params',
        'effective_params = dict(STATE.get("effective_params") or {})',
        'STATE["runtime_dataset_key"] = selected_dataset_name',
        'STATE["selected_dataset_name"] = selected_dataset_name',
    )
    for snippet in required_training_surface_snippets:
        _assert_contains(
            sources.full_source,
            snippet,
            "Notebook 2 training surface is missing required explicit config wiring: {snippet}",
        )

    forbidden_parameter_snippets = (
        'MAX_STABLE_PROFILE = {',
        'profile_payload = dict(MAX_STABLE_PROFILE)',
        'profile=max_stable',
        'NOTEBOOK_OVERRIDE_CASTERS = {',
        'NOTEBOOK_SETTINGS = {',
        'NOTEBOOK_OVERRIDES =',
        'EPOCHS = int(CONTINUAL_CFG.get("num_epochs"',
        'BATCH_SIZE = int(CONTINUAL_CFG.get("batch_size"',
        'LEARNING_RATE = float(CONTINUAL_CFG.get("learning_rate"',
        'source=merged_config(colab)',
        'defaults=config(colab)',
        'DATASET_ROOT = "data/class_root_dataset"',
        'OOD_DATASET_ROOT = "data/ood_dataset"',
        'OOD_DATASET_NAME = ""',
        'inspect_runtime_hardware',
        'recommend_notebook_training_params',
        'resolve_effective_notebook_params',
        'Apply recommended parameters? [y/N]:',
        'accepted_recommendations',
        'recommendation_report = recommend_notebook_training_params',
    )
    for snippet in forbidden_parameter_snippets:
        _assert_not_contains(
            sources.full_source,
            snippet,
            (
                "Notebook 2 parameter cell should not contain hidden overrides "
                "or config-derived parameter remapping: {snippet}"
            ),
        )


def test_batch_training_notebook_contract() -> None:
    sources = _load_notebook_sources("6_train_all_continual_sd_lora_adapters.ipynb")

    _assert_update_check_contract(
        sources.first_code_source,
        "Notebook 6",
        forbid_drive_bootstrap=True,
    )
    for snippet in (
        'NOTEBOOK_NAME = "6_train_all_continual_sd_lora_adapters.ipynb"',
        'NOTEBOOK_FILENAME = "6_train_all_continual_sd_lora_adapters.executed.ipynb"',
        "NB6_AUTO_DISCONNECT_RUNTIME = True",
        "NB6_AUTO_DISCONNECT_GRACE_SECONDS = 20",
        '"AUTO_DISCONNECT_RUNTIME": False',
        '"AUTO_PUSH_TO_GITHUB": True',
        "NB6_MANUAL_PARAM_OVERRIDES = {}",
        "from scripts.notebook_helpers.adapter_recommendations import get_adapter_recs",
        "ADAPTER_RECS = get_adapter_recs()",
        "NB6_ADAPTER_SEQUENCE = [",
        "for index, adapter_key in enumerate(NB6_ADAPTER_SEQUENCE, start=1):",
        "MANUAL_PARAM_OVERRIDES = dict(NB6_MANUAL_PARAM_OVERRIDES.get(adapter_key, {}))",
        "from scripts.colab_notebook_helpers import maybe_auto_disconnect_colab_runtime",
        '"batch_loop_completed": True',
        '"all_adapters_attempted": len(NB6_RESULTS) == len(NB6_ADAPTER_SEQUENCE)',
        "enabled=bool(NB6_AUTO_DISCONNECT_RUNTIME)",
    ):
        _assert_contains(
            sources.full_source,
            snippet,
            "Notebook 6 batch surface is missing required batch-training contract: {snippet}",
        )
    for stale_snippet in (
        '"grape__fruit": {"crop": "grape"',
        '"strawberry__leaf": {"crop": "strawberry"',
        '"tomato__leaf": {"crop": "tomato"',
    ):
        _assert_not_contains(
            sources.full_source,
            stale_snippet,
            "Notebook 6 should not embed adapter recommendation copies; use get_adapter_recs(): {snippet}",
        )
    for script_name in (
        "nb2_cell03_runtime_setup.py",
        "nb2_cell04_parameter_resolution.py",
        "nb2_cell05_access_check.py",
        "nb2_cell06_dataset_validation.py",
        "nb2_cell07_engine_init.py",
        "nb2_cell08_ood_config_verify.py",
        "nb2_cell09_training.py",
        "nb2_cell10_ood_calibration.py",
        "nb2_cell11_adapter_save.py",
        "nb2_cell12_final_evaluation.py",
    ):
        _assert_contains(
            sources.full_source,
            f"run_cell_script('{script_name}', globals())",
            "Notebook 6 should execute the maintained Notebook 2 cell script sequence: {snippet}",
        )


def test_router_calibration_notebook_contract() -> None:
    sources = _load_notebook_sources("5_calibrate_router_handoff_thresholds.ipynb")

    _assert_repo_bootstrap_contract(sources.first_code_source, "Notebook 5")

    assert "from scripts.evaluate_router_surface import discover_eval_samples, evaluate_router_surface" in sources.full_source
    assert "from scripts.calibrate_router_surface import calibrate_router_surface" in sources.full_source
    assert "ROUTER_EVAL_ROOT = 'data/router_eval'" in sources.full_source
    assert "HOLDOUT_EVAL_ROOT = 'data/router_eval_holdout'" in sources.full_source
    assert "RUN_BASELINE_EVAL = False" in sources.full_source
    assert "RUN_CALIBRATION = True" in sources.full_source
    assert "RUN_HOLDOUT_VALIDATION = True" in sources.full_source
    assert "CALIBRATION_STRATEGY = 'replay-thresholds'" in sources.full_source
    assert "CALIBRATION_PRESET = 'handoff'" in sources.full_source
    assert "Notebook 5 first cell started." in sources.full_source
    assert "['git', 'clone', '--depth', '1', '--progress'" in sources.full_source
    assert "validate_router_candidate_overrides" in sources.full_source
    assert "run_cell_script('nb5_cell06_holdout_validation.py', globals())" in sources.full_source
    assert "target_negative_false_accept_rate=TARGET_NEGATIVE_FALSE_ACCEPT_RATE" in sources.full_source
    assert "max_crop_accuracy_drop=MAX_CROP_ACCURACY_DROP" in sources.full_source
    assert "max_part_precision_drop=MAX_PART_PRECISION_DROP" in sources.full_source
    assert "max_wrong_part_rejection_drop=MAX_WRONG_PART_REJECTION_DROP" in sources.full_source
    assert "strategy=CALIBRATION_STRATEGY" in sources.full_source


def test_ood_oe_quality_notebook_contract() -> None:
    sources = _load_notebook_sources("7_ood_oe_quality.ipynb")

    _assert_code_cells_compile(sources, "Notebook 7")
    _assert_contains(
        sources.full_source,
        "RUN_ALL_DATASETS = True",
        "Notebook 7 should default to the batch prepared-runtime audit flow: {snippet}",
    )
    _assert_contains(
        sources.full_source,
        "review_decisions.csv",
        "Notebook 7 should expose the maintained review CSV contract: {snippet}",
    )
    _assert_contains(
        sources.full_source,
        "APPLY_REVIEW_DECISIONS = False",
        "Notebook 7 should keep quarantine application opt-in: {snippet}",
    )
    _assert_contains(
        sources.full_source,
        "--apply-decisions",
        "Notebook 7 should apply decisions through the maintained audit script: {snippet}",
    )
    for stale_snippet in (
        "Fully Automated",
        "AUTO_QUARANTINE",
        "APPLY_NOW",
        "Auto-generate",
        "auto-quarantine",
    ):
        _assert_not_contains(
            sources.full_source,
            stale_snippet,
            "Notebook 7 should stay human-in-loop and avoid automatic quarantine wording: {snippet}",
        )


CHECKS = (
    ValidationCheck(
        result_name="Runtime Dependencies",
        step_id="ENV",
        description="runtime dependencies",
        success_message="Runtime dependencies available",
        failure_prefix="Missing dependencies",
        callback=_check_runtime_dependencies,
        requires_runtime_dependencies=False,
    ),
    ValidationCheck(
        result_name="Minimal Config",
        step_id="CONFIG",
        description="minimal config load",
        success_message="Configuration loaded successfully",
        failure_prefix="Configuration load failed",
        callback=test_config_surface,
    ),
    ValidationCheck(
        result_name="Continual Trainer",
        step_id="TRAINING",
        description="continual trainer imports",
        success_message="Continual trainer surface imported and validated",
        failure_prefix="Continual trainer test failed",
        callback=test_continual_trainer_imports,
    ),
    ValidationCheck(
        result_name="Quantization Guard",
        step_id="LOW_BIT_GUARD",
        description="4-bit rejection guard",
        success_message="Quantization guard behaves correctly",
        failure_prefix="Quantization guard failed",
        callback=test_quantization_guard,
    ),
    ValidationCheck(
        result_name="Adapter Lifecycle",
        step_id="ADAPTER_API",
        description="adapter lifecycle surface",
        success_message="Adapter lifecycle surface available",
        failure_prefix="Adapter API test failed",
        callback=test_adapter_surface,
    ),
    ValidationCheck(
        result_name="Router Runtime",
        step_id="INFERENCE",
        description="router runtime surface",
        success_message="Router runtime surface available",
        failure_prefix="Router runtime test failed",
        callback=test_runtime_surface,
    ),
    ValidationCheck(
        result_name="Adapter Smoke Notebook",
        step_id="ADAPTER_SMOKE",
        description="adapter smoke-test helper surface",
        success_message="Adapter smoke-test helper surface available",
        failure_prefix="Adapter smoke-test surface failed",
        callback=test_adapter_smoke_notebook_surface,
    ),
    ValidationCheck(
        result_name="Notebook 3 Bootstrap",
        step_id="NB3_BOOTSTRAP",
        description="Notebook 3 bootstrap contract",
        success_message="Notebook 3 bootstrap uses GitHub/local repo discovery without Drive repo mounts",
        failure_prefix="Notebook 3 bootstrap contract failed",
        callback=test_adapter_smoke_notebook_bootstrap_contract,
    ),
    ValidationCheck(
        result_name="Notebook 4 Bootstrap",
        step_id="NB4_BOOTSTRAP",
        description="Notebook 4 bootstrap contract",
        success_message="Notebook 4 bootstrap clones the repo and launches the minimal smoke UI",
        failure_prefix="Notebook 4 bootstrap contract failed",
        callback=test_simple_adapter_smoke_notebook_bootstrap_contract,
    ),
    ValidationCheck(
        result_name="Notebook 5 Router Calibration",
        step_id="NB5_ROUTER_CAL",
        description="Notebook 5 router calibration contract",
        success_message="Notebook 5 wraps maintained router evaluation and calibration scripts",
        failure_prefix="Notebook 5 router calibration contract failed",
        callback=test_router_calibration_notebook_contract,
        requires_runtime_dependencies=False,
    ),
    ValidationCheck(
        result_name="Notebook 6 Batch Training",
        step_id="NB6_BATCH",
        description="Notebook 6 batch training contract",
        success_message="Notebook 6 bootstraps Colab and wraps maintained Notebook 2 training cells",
        failure_prefix="Notebook 6 batch training contract failed",
        callback=test_batch_training_notebook_contract,
        requires_runtime_dependencies=False,
    ),
    ValidationCheck(
        result_name="Notebook 7 OOD/OE Review",
        step_id="NB7_OOD_OE_REVIEW",
        description="Notebook 7 OOD/OE human-review contract",
        success_message="Notebook 7 stays batchable and human-in-loop for quarantine decisions",
        failure_prefix="Notebook 7 OOD/OE review contract failed",
        callback=test_ood_oe_quality_notebook_contract,
        requires_runtime_dependencies=False,
    ),
    ValidationCheck(
        result_name="Notebook 8 Auto Inference",
        step_id="NB8_AUTO_INFER",
        description="Notebook 8 auto router-adapter contract",
        success_message="Notebook 8 wraps Notebook 1 routing and canonical adapter inference",
        failure_prefix="Notebook 8 auto inference contract failed",
        callback=test_auto_router_adapter_notebook_contract,
        requires_runtime_dependencies=False,
    ),
    ValidationCheck(
        result_name="Colab Helpers",
        step_id="COLAB",
        description="colab support helpers",
        success_message="Colab helper surfaces imported successfully",
        failure_prefix="Colab helper import failed",
        callback=test_colab_helpers,
    ),
    ValidationCheck(
        result_name="Notebook 2 Bootstrap",
        step_id="NB2_BOOTSTRAP",
        description="Notebook 2 bootstrap contract",
        success_message="Notebook 2 bootstrap globals are defined before use",
        failure_prefix="Notebook 2 bootstrap contract failed",
        callback=test_training_notebook_bootstrap_contract,
        requires_runtime_dependencies=False,
    ),
    ValidationCheck(
        result_name="Notebook 0 Bootstrap",
        step_id="NB0_BOOTSTRAP",
        description="Notebook 0 bootstrap contract",
        success_message="Notebook 0 bootstrap globals are defined before use",
        failure_prefix="Notebook 0 bootstrap contract failed",
        callback=test_data_prep_notebook_contract,
        requires_runtime_dependencies=False,
    ),
    ValidationCheck(
        result_name="Data Scaffold",
        step_id="DATA_LAYOUT",
        description="repo dataset scaffold",
        success_message="Repo-local dataset scaffold is present",
        failure_prefix="Repo dataset scaffold check failed",
        callback=test_repo_dataset_scaffold,
        requires_runtime_dependencies=False,
    ),
    ValidationCheck(
        result_name="Notebook 2 Dataset Contract",
        step_id="NB2_RUNTIME",
        description="Notebook 2 runtime dataset contract",
        success_message="Notebook 2 requires a prepared runtime dataset from Notebook 0",
        failure_prefix="Notebook 2 dataset contract check failed",
        callback=test_training_notebook_dataset_contract_detection,
        requires_runtime_dependencies=False,
    ),
)


def main() -> int:
    print("=" * 60)
    print("AADS v6 Minimal Surface Validation")
    print("=" * 60)

    results = []
    runtime_dependencies_ready = True
    for index, check in enumerate(CHECKS):
        if check.requires_runtime_dependencies and not runtime_dependencies_ready:
            continue
        ok = _run_check(check, leading_newline=index > 1)
        results.append((check.result_name, ok))
        if check.step_id == "ENV":
            runtime_dependencies_ready = ok

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {name}")

    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
