# Auto-extracted from colab_notebooks/5_calibrate_router_handoff_thresholds.ipynb cell 6.
# Keep notebook execute-only cells thin; edit behavior here.

holdout_recommended = (globals().get('holdout_result') or {}).get('recommended', {})
recommended = holdout_recommended or (globals().get('calibration_result') or {}).get('recommended', {})
overrides = dict(recommended.get('overrides') or {})
if overrides:
    source = 'holdout' if holdout_recommended else 'dev'
    print(f'[CONFIG PREVIEW] Apply these {source}-validated dotted-path values to the relevant config after review:')
    print(json.dumps(overrides, indent=2))
else:
    print('[CONFIG PREVIEW] No holdout-accepted override recommendation is available yet.')
