"""Compare OOD methods (ensemble/knn/energy) using production_readiness.json method-level pooled metrics.

Outputs: .runtime_tmp/knn_vs_ensemble_summary.csv
"""
import json
from pathlib import Path
import csv

RUNTIME = Path('.runtime_tmp')
RUNTIME.mkdir(parents=True, exist_ok=True)
OUT = RUNTIME / 'knn_vs_ensemble_summary.csv'

# target runs (from user's list)
TARGET_RUNS = [
    ('grape','fruit','grape_fruit_2026-05-24_14-47-58'),
    ('grape','leaf','grape_leaf_2026-05-24_14-41-21'),
    ('apricot','leaf','apricot_leaf_2026-05-24_14-15-18'),
    ('tomato','fruit','tomato_fruit_2026-05-24_15-06-48')
]

rows = []
for crop, part, run in TARGET_RUNS:
    # try outputs path then telemetry path
    candidates = [
        Path('runs')/crop/part/run/ 'outputs' / 'colab_notebook_training' / 'artifacts' / 'production_readiness.json',
    ]
    # correct builder: some runs have production_readiness under outputs/...; also under telemetry/artifacts/production_readiness.json
    p1 = Path('runs')/crop/part/run/'outputs'/'colab_notebook_training'/'artifacts'/'production_readiness.json'
    p2 = Path('runs')/crop/part/run/'telemetry'/'artifacts'/'production_readiness.json'
    prod = None
    if p1.exists(): prod = p1
    elif p2.exists(): prod = p2
    else:
        # search
        for p in (Path('runs')/crop/part/run).rglob('production_readiness.json'):
            prod = p
            break
    if not prod:
        print('production_readiness.json not found for', run)
        continue
    with open(prod, 'r', encoding='utf-8') as f:
        pr = json.load(f)
    omc = pr.get('ood_method_comparison', {})
    methods = omc.get('methods', {})
    for m, info in methods.items():
        pooled = info.get('pooled_metrics', {})
        rows.append({
            'run': run,
            'crop': crop,
            'part': part,
            'method': m,
            'ood_auroc': pooled.get('ood_auroc'),
            'ood_fpr': pooled.get('ood_false_positive_rate'),
            'ood_samples': pooled.get('ood_samples'),
            'in_distribution_samples': pooled.get('in_distribution_samples')
        })

with open(OUT, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['run','crop','part','method','ood_auroc','ood_fpr','ood_samples','in_distribution_samples'])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print('Wrote', OUT)
