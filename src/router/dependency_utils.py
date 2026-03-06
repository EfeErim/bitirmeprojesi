from typing import Any, Callable, Dict, Optional


def check_vlm_dependencies(
    import_module: Callable[[str], Any] = __import__,
    transformers_version_provider: Optional[Callable[[], str]] = None,
) -> Dict[str, Any]:
    """Collect dependency diagnostics for VLM runtime without direct logging side effects."""
    result: Dict[str, Any] = {
        'transformers_warning': None,
        'missing_deps': [],
        'install_command': None,
    }

    try:
        version = transformers_version_provider() if transformers_version_provider is not None else import_module('transformers').__version__
        major, minor, _patch = map(int, version.split('.')[:3])
        if (major, minor) < (4, 41):
            result['transformers_warning'] = (
                f"transformers {version} may not have SAM3. Recommend >=4.41.0. "
                "Install: !pip install transformers --upgrade"
            )
    except Exception as exc:
        result['transformers_warning'] = f"Could not check transformers version: {exc}"

    optional_packages = {
        'open_clip': 'open-clip-torch',
    }
    missing_deps = []
    for package_name, pip_name in optional_packages.items():
        try:
            import_module(package_name)
        except ImportError:
            missing_deps.append(pip_name)

    result['missing_deps'] = missing_deps
    if missing_deps:
        result['install_command'] = f"!pip install {' '.join(missing_deps)}"

    return result
