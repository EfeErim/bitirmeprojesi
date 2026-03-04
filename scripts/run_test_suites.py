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
    allow_no_tests: bool = False
    archived: bool = False


SUITES: dict[str, SuiteConfig] = {
    'unit/validation': SuiteConfig(
        name='unit/validation',
        description='Configuration and schema checks',
        paths=(
            'tests/unit/validation/test_schemas.py',
            'tests/unit/validation/test_config_manager_ood_alias.py',
            'tests/unit/validation/test_validation_comprehensive.py',
        ),
    ),
    'unit/training': SuiteConfig(
        name='unit/training',
        description='Continual training and low-bit guard checks',
        paths=(
            'tests/unit/training/test_continual_sd_lora.py',
            'tests/unit/training/test_low_bit_guardrails.py',
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
        paths=(
            'tests/unit/pipeline/test_continual_pipeline_contract.py',
            'tests/unit/pipeline/test_pipeline_comprehensive.py',
            'tests/unit/pipeline/test_pipeline_strict_router_init.py',
        ),
    ),
    'unit/router': SuiteConfig(
        name='unit/router',
        description='Router policy checks',
        paths=('tests/unit/router',),
    ),
    'unit/adapters': SuiteConfig(
        name='unit/adapters',
        description='Adapter utility checks',
        paths=('tests/unit/adapters',),
    ),
    'unit/dataset': SuiteConfig(
        name='unit/dataset',
        description='Dataset and layout checks',
        paths=('tests/unit/dataset',),
    ),
    'unit/debugging': SuiteConfig(
        name='unit/debugging',
        description='Debugging and monitoring helpers',
        paths=('tests/unit/debugging',),
    ),
    'unit/evaluation': SuiteConfig(
        name='unit/evaluation',
        description='Evaluation metrics checks',
        paths=('tests/unit/evaluation',),
    ),
    'unit/monitoring': SuiteConfig(
        name='unit/monitoring',
        description='Monitoring metrics checks',
        paths=('tests/unit/monitoring',),
    ),
    'unit/security': SuiteConfig(
        name='unit/security',
        description='Security utility checks',
        paths=('tests/unit/security',),
    ),
    'unit/utils': SuiteConfig(
        name='unit/utils',
        description='Utility and optimization checks',
        paths=('tests/unit/utils',),
    ),
    'unit/visualization': SuiteConfig(
        name='unit/visualization',
        description='Visualization surface checks',
        paths=('tests/unit/visualization',),
    ),
    'unit/fixtures': SuiteConfig(
        name='unit/fixtures',
        description='Shared fixture module import checks',
        paths=('tests/fixtures/test_fixtures.py',),
        allow_no_tests=True,
    ),
    'colab/smoke': SuiteConfig(
        name='colab/smoke',
        description='Colab continual smoke checks',
        paths=('tests/colab/test_smoke_training.py',),
    ),
    'colab/environment': SuiteConfig(
        name='colab/environment',
        description='Colab runtime and setup checks',
        paths=(
            'tests/colab/test_data_pipeline.py',
            'tests/colab/test_environment.py',
        ),
    ),
    'integration/core': SuiteConfig(
        name='integration/core',
        description='Core continual integration checks',
        paths=(
            'tests/integration/test_colab_integration.py',
            'tests/integration/test_continual_trainer_metric_gates.py',
            'tests/integration/test_full_pipeline_v6.py',
            'tests/integration/test_continual_trainer_real_backbone.py',
        ),
        extra_pytest_args=('--runintegration',),
    ),
    'archive/v5_legacy': SuiteConfig(
        name='archive/v5_legacy',
        description='Archived legacy test surface (coverage mapping only)',
        paths=('tests/archive/v5_legacy',),
        archived=True,
    ),
}

ACTIVE_V6_ALL = (
    'unit/validation',
    'unit/training',
    'unit/adapter',
    'unit/adapters',
    'unit/ood',
    'unit/dataset',
    'unit/pipeline',
    'unit/router',
    'unit/debugging',
    'unit/evaluation',
    'unit/monitoring',
    'unit/security',
    'unit/utils',
    'unit/visualization',
    'unit/fixtures',
    'colab/smoke',
    'colab/environment',
    'integration/core',
)

GROUPS = {
    'quick': ('unit/validation', 'unit/training', 'unit/adapter', 'unit/ood'),
    'unit': (
        'unit/validation',
        'unit/training',
        'unit/adapter',
        'unit/adapters',
        'unit/ood',
        'unit/dataset',
        'unit/pipeline',
        'unit/router',
        'unit/debugging',
        'unit/evaluation',
        'unit/monitoring',
        'unit/security',
        'unit/utils',
        'unit/visualization',
        'unit/fixtures',
    ),
    'colab': ('colab/smoke', 'colab/environment'),
    'integration': ('integration/core',),
    'archive': ('archive/v5_legacy',),
    'legacy': ('archive/v5_legacy',),
    'active': ACTIVE_V6_ALL,
    'all': ACTIVE_V6_ALL,
}

# Compatibility aliases for legacy suite names.
TARGET_ALIASES = {
    'unit_validation': 'unit/validation',
    'legacy/v5': 'archive/v5_legacy',
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
    for raw_target in selected:
        target = TARGET_ALIASES.get(raw_target, raw_target)
        if target in SUITES:
            candidates = [target]
        elif target in GROUPS:
            candidates = list(GROUPS[target])
        else:
            raise ValueError(f"Unknown suite/group '{raw_target}'")
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
    if suite.archived:
        print('[INFO] Running archived legacy suite (opt-in, excluded from --suite all).')
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
    skipped_no_tests = []
    for suite in suites:
        code = run_suite(root, suite, args.pytest_arg, args.quiet_pytest)
        if code == 5 and suite.allow_no_tests:
            skipped_no_tests.append(suite.name)
            print(f'[SKIP] {suite.name}: no tests collected (optional structural suite).')
            continue
        if code != 0:
            failures.append(suite.name)
            if args.fail_fast:
                break

    if skipped_no_tests:
        print(f"\n[SKIPPED - NO TESTS] {', '.join(skipped_no_tests)}")

    if failures:
        print(f"\n[FAILED SUITES] {', '.join(failures)}")
        return 1

    print('\n[STATUS] All selected suites passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
