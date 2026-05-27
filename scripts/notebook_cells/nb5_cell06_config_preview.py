# Auto-extracted from colab_notebooks/5_calibrate_router_handoff_thresholds.ipynb cell 6.
# Keep notebook execute-only cells thin; edit behavior here.

holdout_recommended = (globals().get('holdout_result') or {}).get('recommended', {})
dev_recommended = (globals().get('calibration_result') or {}).get('recommended', {})
recommended = {}
source = ''
if holdout_recommended and holdout_recommended.get('eligible'):
    recommended = holdout_recommended
    source = 'holdout'
elif dev_recommended and dev_recommended.get('eligible'):
    recommended = dev_recommended
    source = 'dev'
overrides = dict(recommended.get('overrides') or {})
if overrides:
    print(f'[CONFIG PREVIEW] Apply these {source}-validated eligible dotted-path values after review:')
    print(json.dumps(overrides, indent=2))
else:
    rejected = holdout_recommended or dev_recommended or {}
    print('[CONFIG PREVIEW] No eligible override recommendation is available.')
    if rejected:
        print(json.dumps({
            'best_variant_id': rejected.get('variant_id'),
            'eligible': rejected.get('eligible'),
            'eligibility_reasons': rejected.get('eligibility_reasons'),
            'metrics': rejected.get('metrics'),
        }, indent=2))
