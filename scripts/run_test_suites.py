#!/usr/bin/env python3
"""Run modular pytest suites for AADS v6 continual runtime."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SuiteConfig:
    name: str
    description: str
    paths: tuple[str, ...]
    extra_pytest_args: tuple[str, ...] = ()


SUITES: dict[str, SuiteConfig] = {
    'unit/validation': SuiteConfig(
        name='unit/validation',
        description='Configuration and schema checks',
        paths=(
            'tests/unit/validation/test_schemas.py',
            'tests/unit/validation/test_config_manager_ood_alias.py',
        ),
    ),
    'unit/training': SuiteConfig(
        name='unit/training',
        description='Continual training and quantization checks',
        paths=(
            'tests/unit/training/test_continual_sd_lora.py',
            'tests/unit/training/test_int8_quantization.py',
        ),
    ),
    'unit/adapter': SuiteConfig(
        name='unit/adapter',
        description='Continual adapter and fusion checks',
        paths=(
            'tests/unit/adapter/test_continual_adapter.py',
            'tests/unit/adapter/test_multi_scale_fusion.py',
        ),
    ),
    'unit/ood': SuiteConfig(
        name='unit/ood',
        description='Continual OOD checks',
        paths=('tests/unit/ood',),
    ),
    'unit/pipeline': SuiteConfig(
        name='unit/pipeline',
        description='Pipeline regression checks',
        paths=('tests/unit/pipeline/test_continual_pipeline_contract.py',),
    ),
    'unit/router': SuiteConfig(
        name='unit/router',
        description='Router policy checks',
        paths=('tests/unit/router',),
    ),
    'colab/smoke': SuiteConfig(
        name='colab/smoke',
        description='Colab continual smoke checks',
        paths=('tests/colab/test_smoke_training.py',),
    ),
    'integration/core': SuiteConfig(
        name='integration/core',
        description='Core continual integration checks',
        paths=('tests/integration/test_colab_integration.py',),
        extra_pytest_args=('--runintegration',),
    ),
}

GROUPS = {
    'quick': ('unit/validation', 'unit/training', 'unit/adapter', 'unit/ood'),
    'unit': ('unit/validation', 'unit/training', 'unit/adapter', 'unit/ood', 'unit/pipeline', 'unit/router'),
    'colab': ('colab/smoke',),
    'integration': ('integration/core',),
    'all': tuple(SUITES.keys()),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run modular v6 pytest suites.')
    parser.add_argument('--suite', action='append', default=[])
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--fail-fast', action='store_true')
    parser.add_argument('--pytest-arg', action='append', default=[])
    parser.add_argument('--quiet-pytest', action='store_true')
    return parser.parse_args()


def expand_targets(raw_targets: list[str]) -> list[SuiteConfig]:
    selected = raw_targets or ['quick']
    ordered: list[str] = []
    for target in selected:
        if target in SUITES:
            candidates = [target]
        elif target in GROUPS:
            candidates = list(GROUPS[target])
        else:
            raise ValueError(f"Unknown suite/group '{target}'")
        for item in candidates:
            if item not in ordered:
                ordered.append(item)
    return [SUITES[name] for name in ordered]


def list_targets() -> None:
    print('Groups:')
    for name in sorted(GROUPS):
        print(f'  - {name}: {", ".join(GROUPS[name])}')
    print('\nSuites:')
    for name in sorted(SUITES):
        cfg = SUITES[name]
        print(f'  - {cfg.name}: {cfg.description}')
        print(f'    paths: {", ".join(cfg.paths)}')


def run_suite(root: Path, suite: SuiteConfig, extra_args: list[str], quiet: bool) -> int:
    command = [sys.executable, '-m', 'pytest', *suite.paths, *suite.extra_pytest_args]
    if quiet:
        command.append('-q')
    command.extend(extra_args)
    print(f"\n[RUN] {suite.name}")
    print('[CMD]', ' '.join(command))
    return subprocess.run(command, cwd=str(root), check=False).returncode


def main() -> int:
    args = parse_args()
    if args.list:
        list_targets()
        return 0

    try:
        suites = expand_targets(args.suite)
    except ValueError as exc:
        print(f'[ERROR] {exc}')
        return 2

    root = Path(__file__).resolve().parents[1]
    failures = []
    for suite in suites:
        code = run_suite(root, suite, args.pytest_arg, args.quiet_pytest)
        if code != 0:
            failures.append(suite.name)
            if args.fail_fast:
                break

    if failures:
        print(f"\n[FAILED SUITES] {', '.join(failures)}")
        return 1

    print('\n[STATUS] All selected suites passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
