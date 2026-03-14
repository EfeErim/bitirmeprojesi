import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from scripts import colab_repo_bootstrap as bootstrap


def test_resolve_hf_token_prefers_existing_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "env-token")
    monkeypatch.setattr(bootstrap, "running_in_colab", lambda: False)

    assert bootstrap.resolve_hf_token() == "env-token"


def test_resolve_hf_token_reads_colab_secret(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(bootstrap, "running_in_colab", lambda: True)

    fake_colab = ModuleType("google.colab")
    fake_colab.userdata = SimpleNamespace(get=lambda name: "secret-token" if name == "HF_TOKEN" else None)
    fake_google = ModuleType("google")
    fake_google.colab = fake_colab

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.colab", fake_colab)

    assert bootstrap.resolve_hf_token() == "secret-token"
    assert os.environ["HF_TOKEN"] == "secret-token"


def test_resolve_github_token_prefers_existing_env(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "gh-env-token")
    monkeypatch.setattr(bootstrap, "running_in_colab", lambda: False)

    assert bootstrap.resolve_github_token() == "gh-env-token"


def test_resolve_github_token_reads_colab_secret(monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(bootstrap, "running_in_colab", lambda: True)

    fake_colab = ModuleType("google.colab")
    fake_colab.userdata = SimpleNamespace(get=lambda name: "gh-secret" if name == "GH_TOKEN" else None)
    fake_google = ModuleType("google")
    fake_google.colab = fake_colab

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.colab", fake_colab)

    assert bootstrap.resolve_github_token() == "gh-secret"
    assert os.environ["GH_TOKEN"] == "gh-secret"


def test_maybe_clone_repo_uses_github_token_for_https_clone(tmp_path: Path, monkeypatch):
    clone_target = tmp_path / "bitirmeprojesi"
    monkeypatch.delenv("AADS_DISABLE_AUTO_CLONE", raising=False)
    monkeypatch.setenv("AADS_REPO_URL", "https://github.com/EfeErim/bitirmeprojesi.git")
    monkeypatch.setenv("AADS_REPO_CLONE_TARGET", str(clone_target))
    monkeypatch.setattr(bootstrap, "resolve_github_token", lambda: "gh-secret")

    calls: list[list[str]] = []

    def fake_run(command, check=False, stdout=None, stderr=None, text=None):
        calls.append(list(command))
        clone_target.mkdir(parents=True, exist_ok=True)
        (clone_target / "src").mkdir(exist_ok=True)
        (clone_target / "config").mkdir(exist_ok=True)
        (clone_target / "scripts").mkdir(exist_ok=True)
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)

    resolved = bootstrap.maybe_clone_repo()

    assert resolved == clone_target
    assert calls
    assert calls[0][:3] == ["git", "clone", "--depth"]
    assert calls[0][4] == "https://gh-secret@github.com/EfeErim/bitirmeprojesi.git"


def test_login_and_check_hf_token_warns_when_missing(monkeypatch):
    monkeypatch.setattr(bootstrap, "resolve_hf_token", lambda: None)
    lines: list[str] = []

    assert bootstrap.login_and_check_hf_token(print_fn=lines.append) is False
    assert lines == [
        "[HF] No token found. Set a Colab secret or env var named HF_TOKEN before running inference."
    ]


def test_login_and_check_hf_token_validates_identity(monkeypatch):
    calls: dict[str, object] = {}
    lines: list[str] = []

    monkeypatch.setattr(bootstrap, "resolve_hf_token", lambda: "hf-secret")

    fake_hf = ModuleType("huggingface_hub")

    def fake_login(*, token, add_to_git_credential):
        calls["login"] = (token, add_to_git_credential)

    class FakeHfApi:
        def __init__(self, token):
            calls["api_token"] = token

        def whoami(self):
            return {"name": "tester"}

    fake_hf.login = fake_login
    fake_hf.HfApi = FakeHfApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    assert bootstrap.login_and_check_hf_token(print_fn=lines.append) is True
    assert calls["login"] == ("hf-secret", False)
    assert calls["api_token"] == "hf-secret"
    assert lines == ["[HF] Authenticated as tester"]


def test_mirror_checkpoint_state_to_repo_copies_only_best_checkpoint(tmp_path: Path):
    source_root = tmp_path / "source"
    destination_root = tmp_path / "repo" / "runs" / "run_1" / "checkpoint_state"

    best_source = source_root / "checkpoints" / "ckpt_best"
    other_source = source_root / "checkpoints" / "ckpt_other"
    best_source.mkdir(parents=True, exist_ok=True)
    other_source.mkdir(parents=True, exist_ok=True)
    (best_source / "checkpoint.pt").write_text("best", encoding="utf-8")
    (best_source / "adapter_meta.json").write_text("{}", encoding="utf-8")
    (other_source / "checkpoint.pt").write_text("other", encoding="utf-8")

    (source_root / "best_checkpoint.json").parent.mkdir(parents=True, exist_ok=True)
    (source_root / "best_checkpoint.json").write_text(
        json.dumps({"name": "ckpt_best", "path": str(best_source)}) + "\n",
        encoding="utf-8",
    )
    (source_root / "latest_checkpoint.json").write_text(
        json.dumps({"name": "ckpt_other", "path": str(other_source)}) + "\n",
        encoding="utf-8",
    )
    (source_root / "checkpoint_index.json").write_text(
        json.dumps(
            [
                {"name": "ckpt_best", "path": str(best_source)},
                {"name": "ckpt_other", "path": str(other_source)},
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    mirrored_root = bootstrap.mirror_checkpoint_state_to_repo(source_root, destination_root)

    assert mirrored_root == destination_root
    best_checkpoint_file = destination_root / "checkpoints" / "ckpt_best" / "checkpoint.pt"
    assert best_checkpoint_file.exists()
    assert best_checkpoint_file.read_text(encoding="utf-8") == "best"
    assert not (destination_root / "checkpoints" / "best").exists()
    assert not (destination_root / "checkpoints" / "ckpt_other").exists()

    mirrored_best = bootstrap._read_json_dict(destination_root / "best_checkpoint.json")
    mirrored_latest = bootstrap._read_json_dict(destination_root / "latest_checkpoint.json")
    mirrored_index = (destination_root / "checkpoint_index.json").read_text(encoding="utf-8")

    assert mirrored_best["path"] == str(destination_root / "checkpoints" / "ckpt_best")
    assert mirrored_latest["path"] == str(destination_root / "checkpoints" / "ckpt_best")
    assert "ckpt_other" not in mirrored_index


def test_push_repo_run_to_github_skips_pt_files(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "config").mkdir(parents=True)
    (repo_root / "scripts").mkdir(parents=True)
    run_dir = repo_root / "runs" / "run_1"
    keep_file = run_dir / "outputs" / "colab_notebook_training" / "artifacts" / "summary.json"
    pt_file = run_dir / "checkpoint_state" / "checkpoints" / "best" / "checkpoint.pt"
    keep_file.parent.mkdir(parents=True, exist_ok=True)
    pt_file.parent.mkdir(parents=True, exist_ok=True)
    keep_file.write_text("{}", encoding="utf-8")
    pt_file.write_text("weights", encoding="utf-8")

    monkeypatch.setattr(bootstrap, "resolve_github_token", lambda: "gh-secret")

    calls: list[list[str]] = []

    def fake_run(command, cwd=None, check=True, stdout=None, stderr=None, text=None):
        calls.append(list(command))
        args = list(command[1:])
        if args == ["branch", "--show-current"]:
            return subprocess.CompletedProcess(command, 0, stdout="master\n")
        if args == ["remote", "get-url", "origin"]:
            return subprocess.CompletedProcess(command, 0, stdout="https://github.com/EfeErim/bitirmeprojesi.git\n")
        if args[:4] == ["diff", "--cached", "--name-only", "--"]:
            staged_stdout = "runs/run_1/outputs/colab_notebook_training/artifacts/summary.json\n"
            return subprocess.CompletedProcess(command, 0, stdout=staged_stdout)
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)

    result = bootstrap.push_repo_run_to_github(repo_root, "run_1", print_fn=lambda _: None)

    assert result["pushed"] is True
    add_calls = [call for call in calls if call[1] == "add"]
    assert add_calls
    added_args = " ".join(" ".join(call) for call in add_calls)
    assert "summary.json" in added_args
    assert "checkpoint.pt" not in added_args

    commit_calls = [call for call in calls if call[1] == "commit"]
    assert commit_calls
    assert commit_calls[0][-2:] == ["--", "runs/run_1"]

    push_calls = [call for call in calls if call[1] == "push"]
    assert push_calls
    assert "gh-secret@" in push_calls[0][2]
