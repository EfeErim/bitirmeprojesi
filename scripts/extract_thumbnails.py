"""Extract thumbnails for worst OOD slices per run.

Usage: python scripts/extract_thumbnails.py

Saves thumbnails to .runtime_tmp/samples/<run>/<slice>/
"""
import csv
from pathlib import Path
from PIL import Image
import random
import os
import json

RUNTIME = Path('.runtime_tmp')
SAMPLES_DIR = RUNTIME / 'samples'
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# runs and default thumbnail count
TARGETS = [
    ('grape','fruit','grape_fruit_2026-05-24_14-47-58'),
    ('grape','leaf','grape_leaf_2026-05-24_14-41-21'),
    ('apricot','leaf','apricot_leaf_2026-05-24_14-15-18'),
    ('tomato','fruit','tomato_fruit_2026-05-24_15-06-48')
]
N = 20

REPORT = {}


def local_path(image_path):
    # convert /content/bitirmeprojesi/... to workspace path
    if image_path.startswith('/content/bitirmeprojesi'):
        rel = image_path.replace('/content/bitirmeprojesi/', '').lstrip('/')
        # on Windows workspace root is current dir
        p = Path(rel)
        return Path.cwd() / p
    # handle absolute workspace-like
    if image_path.startswith('data/') or image_path.startswith('runs/'):
        return Path.cwd() / image_path
    return Path(image_path)

for crop, part, run in TARGETS:
    print('Processing', run)
    # find predictions.csv
    preds = None
    p1 = Path('runs')/crop/part/run/'outputs'/'colab_notebook_training'/'artifacts'/'test'/'predictions.csv'
    p2 = Path('runs')/crop/part/run/'telemetry'/'artifacts'/'test'/'predictions.csv'
    p3 = Path('runs')/crop/part/run/'outputs'/'colab_notebook_training'/'artifacts'/'test'/'predictions.csv'
    for candidate in [p1,p2,p3]:
        if candidate.exists():
            preds = candidate
            break
    if not preds:
        # search
        for p in (Path('runs')/crop/part/run).rglob('predictions.csv'):
            preds = p
            break
    if not preds:
        print('predictions.csv not found for', run)
        continue
    # load production_readiness to get worst slice name
    prod = None
    for p in (Path('runs')/crop/part/run).rglob('production_readiness.json'):
        prod = p
        break
    worst_slice = None
    if prod:
        try:
            with open(prod,'r',encoding='utf-8') as f:
                pr = json.load(f)
            w = pr.get('ood_method_comparison', {}).get('methods', {}).get('ensemble', {}).get('worst_slice', {})
            worst_slice = w.get('slice_name') if isinstance(w, dict) else None
        except Exception:
            worst_slice = None
    # fallback: ask user (but we'll pick first ood_type in breakdown)
    if not worst_slice and prod:
        with open(prod,'r',encoding='utf-8') as f:
            pr = json.load(f)
        types = list(pr.get('ood_type_breakdown', {}).keys())
        worst_slice = types[0] if types else None
    if not worst_slice:
        print('Could not determine worst slice for', run)
        continue

    out_dir = SAMPLES_DIR / run / worst_slice
    out_dir.mkdir(parents=True, exist_ok=True)

    # read predictions and filter by ood_type
    rows = []
    with open(preds, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            ot = r.get('ood_type') or r.get('sample_origin')
            if ot == worst_slice or (r.get('sample_origin') and r.get('sample_origin') != 'in_distribution' and worst_slice in r.get('image_path')):
                rows.append(r)
    if not rows:
        # try matching by ood_primary_score_method? fallback to any ood sample
        with open(preds, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get('sample_origin') and r.get('sample_origin') != 'in_distribution':
                    rows.append(r)
    # sample up to N
    sample = rows if len(rows) <= N else random.sample(rows, N)
    stats = []
    for i, r in enumerate(sample):
        imgp = r.get('image_path')
        lp = local_path(imgp)
        outp = out_dir / f'{i+1:02d}_{Path(lp).name}'
        try:
            with Image.open(lp) as im:
                im.thumbnail((512,512))
                im.save(outp)
            stats.append({'image': str(lp), 'saved': str(outp), 'ok': True})
        except Exception as e:
            stats.append({'image': str(lp), 'error': str(e), 'ok': False})
    REPORT[run] = {'worst_slice': worst_slice, 'count_found': len(rows), 'sampled': len(sample), 'out_dir': str(out_dir), 'stats': stats}

# write report
with open(RUNTIME / 'samples_report.json', 'w', encoding='utf-8') as f:
    json.dump(REPORT, f, indent=2, ensure_ascii=False)

print('Done. Report at .runtime_tmp/samples_report.json')
