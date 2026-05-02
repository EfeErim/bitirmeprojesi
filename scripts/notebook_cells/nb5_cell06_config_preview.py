# Auto-extracted from colab_notebooks/5_calibrate_router_handoff_thresholds.ipynb cell 6.
# Keep notebook execute-only cells thin; edit behavior here.

recommended = (globals().get('calibration_result') or {}).get('recommended', {})
overrides = dict(recommended.get('overrides') or {})
if overrides:
    print('[CONFIG PREVIEW] Apply these dotted-path values to the relevant config after review:')
    print(json.dumps(overrides, indent=2))
else:
    print('[CONFIG PREVIEW] No override recommendation is available yet.')
