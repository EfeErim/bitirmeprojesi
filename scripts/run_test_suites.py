#!/usr/bin/env python3
"""Run modular pytest suites with per-suite status reporting.

Default behavior runs the `quick` group to keep local feedback fast.
Use `--suite all` when you explicitly want the full test matrix.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SuiteConfig:
    name: str
    description: str
    paths: tuple[str, ...]
    extra_pytest_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class SuiteResult:
    name: str
    command: tuple[str, ...]
    exit_code: int
    tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration_seconds: float

    @property
    def status(self) -> str:
        return "PASS" if self.exit_code == 0 else "FAIL"


def _build_suite_definitions() -> dict[str, SuiteConfig]:
    suites = {
        "unit/adapters": SuiteConfig(
            name="unit/adapters",
            description="Adapter lifecycle and prototype behavior",
            paths=("tests/unit/adapters",),
        ),
        "unit/dataset": SuiteConfig(
            name="unit/dataset",
            description="Dataset prep/load/cache behavior",
            paths=("tests/unit/dataset",),
        ),
        "unit/debugging": SuiteConfig(
            name="unit/debugging",
            description="Debug collector and monitoring helpers",
            paths=("tests/unit/debugging",),
        ),
        "unit/evaluation": SuiteConfig(
            name="unit/evaluation",
            description="Evaluation metrics behavior",
            paths=("tests/unit/evaluation",),
        ),
        "unit/monitoring": SuiteConfig(
            name="unit/monitoring",
            description="Runtime monitoring metrics",
            paths=("tests/unit/monitoring",),
        ),
        "unit/ood": SuiteConfig(
            name="unit/ood",
            description="OOD logic and thresholds",
            paths=("tests/unit/ood", "tests/unit/test_ood.py"),
        ),
        "unit/pipeline": SuiteConfig(
            name="unit/pipeline",
            description="Pipeline assembly and strict router init",
            paths=("tests/unit/pipeline",),
        ),
        "unit/router": SuiteConfig(
            name="unit/router",
            description="Router policy and strict VLM loading",
            paths=("tests/unit/router", "tests/unit/test_router.py"),
        ),
        "unit/security": SuiteConfig(
            name="unit/security",
            description="Security utility checks",
            paths=("tests/unit/security",),
        ),
        "unit/training": SuiteConfig(
            name="unit/training",
            description="Phase 2/3 training behavior",
            paths=("tests/unit/training",),
        ),
        "unit/utils": SuiteConfig(
            name="unit/utils",
            description="Utility behavior and optimization helpers",
            paths=("tests/unit/utils",),
        ),
        "unit/validation": SuiteConfig(
            name="unit/validation",
            description="Config/schema validation checks",
            paths=("tests/unit/validation",),
        ),
        "unit/visualization": SuiteConfig(
            name="unit/visualization",
            description="Visualization utilities",
            paths=("tests/unit/visualization",),
        ),
        "colab/environment": SuiteConfig(
            name="colab/environment",
            description="Colab environment compatibility checks",
            paths=("tests/colab/test_environment.py",),
        ),
        "colab/data": SuiteConfig(
            name="colab/data",
            description="Colab data pipeline tests",
            paths=("tests/colab/test_data_pipeline.py",),
        ),
        "colab/smoke": SuiteConfig(
            name="colab/smoke",
            description="Colab training smoke tests",
            paths=("tests/colab/test_smoke_training.py",),
        ),
        "integration/core": SuiteConfig(
            name="integration/core",
            description="Core integration checks",
            paths=(
                "tests/integration/test_configuration_integration.py",
                "tests/integration/test_configuration_final.py",
                "tests/integration/test_full_pipeline.py",
            ),
            extra_pytest_args=("--runintegration",),
        ),
        "integration/heavy": SuiteConfig(
            name="integration/heavy",
            description="Heavy-model integration checks",
            paths=("tests/integration/test_colab_integration.py",),
            extra_pytest_args=("--runintegration", "--runheavymodel"),
        ),
    }
    return suites


SUITES = _build_suite_definitions()

GROUPS = {
    "quick": (
        "unit/validation",
        "unit/router",
        "unit/ood",
        "unit/pipeline",
    ),
    "unit": tuple(name for name in SUITES if name.startswith("unit/")),
    "colab": tuple(name for name in SUITES if name.startswith("colab/")),
    "integration": tuple(name for name in SUITES if name.startswith("integration/")),
    "all": tuple(SUITES.keys()),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run modular AADS pytest suites with clear per-suite reporting.",
    )
    parser.add_argument(
        "--suite",
        action="append",
        default=[],
        help=(
            "Suite or group to run (repeatable). "
            f"Suites: {', '.join(sorted(SUITES.keys()))}. "
            f"Groups: {', '.join(sorted(GROUPS.keys()))}. "
            "Default: quick"
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available suites and groups, then exit.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after first failing suite.",
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Extra argument passed through to pytest (repeatable).",
    )
    parser.add_argument(
        "--quiet-pytest",
        action="store_true",
        help="Run pytest with -q for less per-test noise.",
    )
    return parser.parse_args()


def _expand_targets(raw_targets: list[str]) -> list[SuiteConfig]:
    selected = raw_targets or ["quick"]
    ordered_names: list[str] = []

    for target in selected:
        if target in SUITES:
            candidates = [target]
        elif target in GROUPS:
            candidates = list(GROUPS[target])
        else:
            raise ValueError(
                f"Unknown suite/group '{target}'. Use --list to inspect valid names."
            )

        for name in candidates:
            if name not in ordered_names:
                ordered_names.append(name)

    return [SUITES[name] for name in ordered_names]


def _safe_xml_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")


def _parse_junit_xml(xml_path: Path) -> tuple[int, int, int, int, int, float]:
    if not xml_path.exists():
        return 0, 0, 0, 0, 0, 0.0

    root = ET.parse(xml_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites and root.tag == "testsuites":
        suites = list(root.iter("testsuite"))

    tests = failures = skipped = errors = 0
    duration = 0.0

    for suite in suites:
        tests += int(suite.attrib.get("tests", 0))
        failures += int(suite.attrib.get("failures", 0))
        skipped += int(suite.attrib.get("skipped", 0))
        errors += int(suite.attrib.get("errors", 0))
        duration += float(suite.attrib.get("time", 0.0))

    passed = max(tests - failures - errors - skipped, 0)
    return tests, passed, failures, skipped, errors, duration


def _print_listing() -> None:
    print("\nAvailable suite groups:")
    for group_name in sorted(GROUPS):
        print(f"  - {group_name}: {', '.join(GROUPS[group_name])}")

    print("\nAvailable concrete suites:")
    for suite_name in sorted(SUITES):
        suite = SUITES[suite_name]
        joined_paths = ", ".join(suite.paths)
        print(f"  - {suite.name}: {suite.description}")
        print(f"    paths: {joined_paths}")
        if suite.extra_pytest_args:
            print(f"    extra args: {' '.join(suite.extra_pytest_args)}")


def _format_duration(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _print_suite_results(results: list[SuiteResult]) -> None:
    headers = ("Suite", "Status", "Passed/Total", "Failed", "Skipped", "Errors", "Time")
    rows = []
    for result in results:
        rows.append(
            (
                result.name,
                result.status,
                f"{result.passed}/{result.tests}",
                str(result.failed),
                str(result.skipped),
                str(result.errors),
                _format_duration(result.duration_seconds),
            )
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt_line(values: tuple[str, ...]) -> str:
        return "  ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    print("\n[SUMMARY] Suite execution results")
    print(fmt_line(headers))
    print(fmt_line(tuple("-" * width for width in widths)))
    for row in rows:
        print(fmt_line(row))

    total_tests = sum(item.tests for item in results)
    total_passed = sum(item.passed for item in results)
    total_failed = sum(item.failed for item in results)
    total_skipped = sum(item.skipped for item in results)
    total_errors = sum(item.errors for item in results)
    total_time = sum(item.duration_seconds for item in results)
    failing = [item.name for item in results if item.exit_code != 0]

    print(
        "\n[TOTAL] "
        f"tests={total_tests} passed={total_passed} failed={total_failed} "
        f"skipped={total_skipped} errors={total_errors} "
        f"time={_format_duration(total_time)}"
    )
    if failing:
        print(f"[FAILED SUITES] {', '.join(failing)}")
    else:
        print("[STATUS] All selected suites passed.")


def _run_suite(
    root: Path,
    suite: SuiteConfig,
    python_executable: str,
    junit_dir: Path,
    base_pytest_args: list[str],
) -> SuiteResult:
    junit_path = junit_dir / f"{_safe_xml_name(suite.name)}.xml"

    command = [
        python_executable,
        "-m",
        "pytest",
        *suite.paths,
        *suite.extra_pytest_args,
        *base_pytest_args,
        f"--junitxml={junit_path.as_posix()}",
    ]

    print(f"\n[RUN] {suite.name}")
    print(f"[INFO] {suite.description}")
    print(f"[CMD] {' '.join(command)}")
    sys.stdout.flush()

    start = time.perf_counter()
    completed = subprocess.run(command, cwd=str(root), check=False)
    wall_seconds = time.perf_counter() - start

    tests, passed, failed, skipped, errors, xml_seconds = _parse_junit_xml(junit_path)
    duration = xml_seconds if xml_seconds > 0 else wall_seconds
    result = SuiteResult(
        name=suite.name,
        command=tuple(command),
        exit_code=completed.returncode,
        tests=tests,
        passed=passed,
        failed=failed,
        skipped=skipped,
        errors=errors,
        duration_seconds=duration,
    )

    print(
        f"[{result.status}] {result.name} "
        f"(passed={result.passed}/{result.tests}, failed={result.failed}, "
        f"skipped={result.skipped}, errors={result.errors}, "
        f"time={_format_duration(result.duration_seconds)})"
    )
    return result


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    junit_dir = root / ".runtime_tmp" / "test_suites_junit"
    junit_dir.mkdir(parents=True, exist_ok=True)

    if args.list:
        _print_listing()
        return 0

    try:
        selected_suites = _expand_targets(args.suite)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 2

    print("[INFO] Selected suites:")
    for suite in selected_suites:
        print(f"  - {suite.name}")

    base_pytest_args = list(args.pytest_arg)
    if args.quiet_pytest:
        base_pytest_args.append("-q")

    all_results: list[SuiteResult] = []
    python_executable = str(Path(sys.executable))

    for suite in selected_suites:
        result = _run_suite(
            root=root,
            suite=suite,
            python_executable=python_executable,
            junit_dir=junit_dir,
            base_pytest_args=base_pytest_args,
        )
        all_results.append(result)
        if args.fail_fast and result.exit_code != 0:
            break

    _print_suite_results(all_results)
    return 1 if any(item.exit_code != 0 for item in all_results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
