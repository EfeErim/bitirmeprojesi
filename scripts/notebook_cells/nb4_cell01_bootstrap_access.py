# Auto-extracted from colab_notebooks/4_simple_direct_adapter_test_ui.ipynb cell 1.
# Keep notebook execute-only cells thin; edit behavior here.

from pathlib import Path
import os
import sys
import urllib.request

DOWNLOAD_TARGET = Path("/content/bitirmeprojesi")
DEFAULT_REPO_URL = "https://github.com/EfeErim/bitirmeprojesi.git"
REPO_URL = DEFAULT_REPO_URL
REPO_REF = os.environ.get("AADS_REPO_REF", "master")
DOWNLOAD_MANIFEST = "scripts/notebook4_raw_download_manifest.txt"


def _download_file(raw_base: str, rel_path: str, dest_root: Path) -> Path:
    dest_path = Path(dest_root) / rel_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{raw_base}/{rel_path}"
    print(f"Downloading {url} -> {dest_path}")
    with _open_url(url) as response, open(dest_path, "wb") as handle:
        handle.write(response.read())
    return dest_path


def _github_repo_parts(repo_url: str) -> tuple[str, str]:
    repo = repo_url.rstrip(".git").rstrip("/")
    prefix = "https://github.com/"
    if not repo.startswith(prefix):
        raise RuntimeError("Unsupported REPO_URL format for raw fetch: " + repo_url)
    owner_repo = repo[len(prefix) :].strip("/").split("/")
    if len(owner_repo) < 2:
        raise RuntimeError("Unsupported REPO_URL format for raw fetch: " + repo_url)
    return owner_repo[0], owner_repo[1]


def _open_url(url: str):
    headers = {"User-Agent": "aads-notebook4-bootstrap"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers))


def _download_needed_files(raw_base: str, dest_root: Path) -> list[str]:
    manifest_url = f"{raw_base}/{DOWNLOAD_MANIFEST}"
    print(f"Downloading {manifest_url} -> {dest_root / DOWNLOAD_MANIFEST}")
    with _open_url(manifest_url) as response:
        manifest_text = response.read().decode("utf-8")
    paths = [
        line.strip()
        for line in manifest_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not paths:
        raise RuntimeError(f"No Notebook 4 source files listed in {manifest_url}")
    (dest_root / DOWNLOAD_MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    (dest_root / DOWNLOAD_MANIFEST).write_text(manifest_text, encoding="utf-8")
    for rel_path in paths:
        _download_file(raw_base, rel_path, dest_root)
    return paths


def _candidate_raw_bases() -> list[str]:
    refs = [REPO_REF]
    if REPO_REF != "master":
        refs.append("master")
    if "main" not in refs:
        refs.append("main")

    raw_bases: list[str] = []
    owner, repo = _github_repo_parts(REPO_URL)
    for ref in refs:
        raw_base = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}"
        if raw_base not in raw_bases:
            raw_bases.append(raw_base)
    return raw_bases


def _ensure_aads_repo_on_path() -> Path:
    print("Fetching Notebook 4 source files from GitHub raw...")
    DOWNLOAD_TARGET.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    downloaded: list[str] | None = None
    try:
        for raw_base in _candidate_raw_bases():
            try:
                downloaded = _download_needed_files(raw_base, DOWNLOAD_TARGET)
                print(f"Notebook 4 source URL: {raw_base}")
                break
            except Exception as exc:
                errors.append(f"{raw_base}: {exc}")
        if downloaded is None:
            raise RuntimeError("; ".join(errors))
    except Exception as exc:
        raise RuntimeError(f"Failed to download Notebook 4 source files: {exc}") from exc

    if str(DOWNLOAD_TARGET) not in sys.path:
        sys.path.insert(0, str(DOWNLOAD_TARGET))
    os.chdir(DOWNLOAD_TARGET)
    print(f"Notebook 4 source ready: {DOWNLOAD_TARGET} ({len(downloaded)} files)")
    return DOWNLOAD_TARGET


repo_root = _ensure_aads_repo_on_path()
ROOT = repo_root
SEARCH_ROOTS = [str(ROOT / "models/adapters"), str(ROOT / "runs")]

from scripts.colab_repo_bootstrap import (
    collect_notebook_access_report,
    login_and_check_hf_token,
    print_notebook_access_report,
    resolve_hf_token,
)
from src.core.config_manager import get_config

CONFIG_FOR_ACCESS = get_config(environment="colab")
BACKBONE_MODEL_NAME = str(
    dict(dict(CONFIG_FOR_ACCESS.get("training", {})).get("continual", {}))
    .get("backbone", {})
    .get("model_name", "")
).strip()
ACCESS_REPORT = collect_notebook_access_report(
    repo_root=ROOT,
    hf_model_ids=[BACKBONE_MODEL_NAME] if BACKBONE_MODEL_NAME.strip() else [],
)
print_notebook_access_report(ACCESS_REPORT, print_fn=print)
if BACKBONE_MODEL_NAME:
    print(f"[KONTROL] Adapter backbone modeli: {BACKBONE_MODEL_NAME}")
if resolve_hf_token():
    login_and_check_hf_token(print_fn=print)
else:
    print("[HF] Token bulunamadi. Gated backbone icin Colab secret olarak HF_TOKEN ekleyin ve runtime'i yeniden baslatin.")
