import re
from typing import Any, Callable, Dict, Optional

_SUPPORTED_TRANSFORMERS_MIN = (4, 56, 0)
_SUPPORTED_TRANSFORMERS_MAX_EXCLUSIVE = (6, 0, 0)
_SUPPORTED_TRANSFORMERS_RANGE = ">=4.56.0,<6.0.0"


def _parse_transformers_version(version: str) -> tuple[int, int, int]:
    match = re.match(r"^\s*(\d+)\.(\d+)(?:\.(\d+))?", str(version))
    if match is None:
        raise ValueError(f"Unrecognized transformers version: {version!r}")
    major, minor, patch = match.groups(default="0")
    return int(major), int(minor), int(patch)


def check_vlm_dependencies(
    import_module: Callable[[str], Any] = __import__,
    transformers_version_provider: Optional[Callable[[], str]] = None,
) -> Dict[str, Any]:
    """Collect dependency diagnostics for VLM runtime without direct logging side effects."""
    result: Dict[str, Any] = {
        "transformers_warning": None,
        "missing_deps": [],
        "install_command": None,
    }

    try:
        version = (
            transformers_version_provider()
            if transformers_version_provider is not None
            else import_module("transformers").__version__
        )
        parsed_version = _parse_transformers_version(version)
        if not (_SUPPORTED_TRANSFORMERS_MIN <= parsed_version < _SUPPORTED_TRANSFORMERS_MAX_EXCLUSIVE):
            result["transformers_warning"] = (
                f"transformers {version} is outside the supported range {_SUPPORTED_TRANSFORMERS_RANGE}. "
                "DINOv3 backbone support landed in 4.56.0+, and this repo is not yet validated on "
                "Transformers 6.x. Install: !pip install 'transformers>=4.56,<6'"
            )
    except Exception as exc:
        result["transformers_warning"] = f"Could not check transformers version: {exc}"

    optional_packages = {
        "open_clip": "open-clip-torch",
    }
    missing_deps = []
    for package_name, pip_name in optional_packages.items():
        try:
            import_module(package_name)
        except ImportError:
            missing_deps.append(pip_name)

    result["missing_deps"] = missing_deps
    if missing_deps:
        result["install_command"] = f"!pip install {' '.join(missing_deps)}"

    return result
