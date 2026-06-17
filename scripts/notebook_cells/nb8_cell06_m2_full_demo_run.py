# Auto-extracted from colab_notebooks/8_auto_router_adapter_prediction.ipynb cell 6.
# Keep notebook execute-only cells thin; edit behavior here.

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts.colab_notebook_helpers import maybe_auto_disconnect_colab_runtime
from scripts.colab_repo_bootstrap import push_repo_paths_to_github

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
M2_DEMO_LIMIT = globals().get("M2_DEMO_LIMIT", None)
M2_STOP_ON_DEPENDENCY_BLOCKER = bool(globals().get("M2_STOP_ON_DEPENDENCY_BLOCKER", True))
M2_AUTO_PUSH_RESULTS = bool(globals().get("M2_AUTO_PUSH_RESULTS", True))
M2_AUTO_PUSH_REMOTE_NAME = str(globals().get("M2_AUTO_PUSH_REMOTE_NAME", "origin"))
M2_AUTO_PUSH_BRANCH = str(globals().get("M2_AUTO_PUSH_BRANCH", "master") or "").strip() or None
M2_REPO_RESULTS_ROOT = str(globals().get("M2_REPO_RESULTS_ROOT", "docs/demo_results/m2"))
M2_AUTO_DISCONNECT_RUNTIME = bool(globals().get("M2_AUTO_DISCONNECT_RUNTIME", True))
M2_AUTO_DISCONNECT_GRACE_SECONDS = float(globals().get("M2_AUTO_DISCONNECT_GRACE_SECONDS", 20))
DEVICE = str(globals().get("DEVICE", "cuda"))
CONFIG_ENV = str(globals().get("CONFIG_ENV", "colab"))

if not M2_RUN_FULL_DEMO:
    m2_demo_result = None
    m2_demo_publish_report = {"enabled": False, "pushed": False, "reason": "M2_RUN_FULL_DEMO=False"}
    m2_demo_disconnect_report = {"ready": False, "missing": ["m2_full_demo_skipped"]}
    print("[M2] Full demo manifest run skipped because M2_RUN_FULL_DEMO=False.")
else:
    cell_script_root = Path(str(globals().get("__notebook_cell_script_root__", ""))).resolve()
    repo_root = cell_script_root.parents[1] if cell_script_root.name == "notebook_cells" else Path.cwd().resolve()

    manifest_path = (repo_root / M2_DEMO_MANIFEST).resolve()
    output_path = (repo_root / M2_DEMO_OUTPUT).resolve()
    markdown_output_path = (repo_root / M2_DEMO_MARKDOWN_OUTPUT).resolve()
    adapter_root_path = Path(str(globals().get("ADAPTER_ROOT") or "runs"))

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
    ]
    if M2_DEMO_LIMIT is not None:
        command.extend(["--limit", str(int(M2_DEMO_LIMIT))])
    if M2_STOP_ON_DEPENDENCY_BLOCKER:
        command.append("--stop-on-dependency-blocker")

    print(f"[M2] repo_root={repo_root}")
    print(f"[M2] manifest={manifest_path}")
    print(f"[M2] output={output_path}")
    print(f"[M2] markdown_output={markdown_output_path}")
    print("[M2] Starting full manifest run. This can take a while on 512 images.")

    completed = subprocess.run(command, cwd=repo_root, check=False)
    print(f"[M2] runner_exit_code={completed.returncode}")

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

    if report_ready:
        repo_results_dir.mkdir(parents=True, exist_ok=True)
        copied_paths = []
        for source_path in (output_path, markdown_output_path):
            if source_path.is_file():
                destination = repo_results_dir / source_path.name
                shutil.copy2(source_path, destination)
                copied_paths.append(destination.relative_to(repo_root).as_posix())

        summary_path = repo_results_dir / "summary.json"
        summary_payload = {
            "created_at": stamp,
            "runner_exit_code": int(completed.returncode),
            "manifest": str(manifest_path.relative_to(repo_root)),
            "output": str(output_path.relative_to(repo_root)),
            "markdown_output": str(markdown_output_path.relative_to(repo_root)),
            "summary": m2_demo_result.get("summary", {}) if isinstance(m2_demo_result, dict) else {},
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        copied_paths.append(summary_path.relative_to(repo_root).as_posix())
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
    m2_completion_report = {
        "ready": bool(report_ready and push_done),
        "checks": {
            "m2_report_written": bool(report_ready),
            "git_push": bool(push_done),
        },
        "missing": [
            name
            for name, ok in {
                "m2_report_written": bool(report_ready),
                "git_push": bool(push_done),
            }.items()
            if not ok
        ],
        "soft_missing": [],
    }
    print(f"[COLAB] M2 completion checks -> {m2_completion_report['checks']}")
    m2_demo_disconnect_report = maybe_auto_disconnect_colab_runtime(
        enabled=bool(M2_AUTO_DISCONNECT_RUNTIME),
        grace_period_sec=M2_AUTO_DISCONNECT_GRACE_SECONDS,
        completion_report=m2_completion_report,
        print_fn=print,
    )

m2_demo_result
