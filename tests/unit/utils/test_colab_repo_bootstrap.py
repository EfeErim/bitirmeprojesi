import os
import sys
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
