# Auto-extracted from colab_notebooks/5_calibrate_router_handoff_thresholds.ipynb cell 7.
# Keep notebook execute-only cells thin; edit behavior here.

if PUBLISH_RESULTS_TO_GIT:
    import shutil
    import subprocess
    from datetime import datetime, timezone

    publish_root = Path(PUBLISH_RESULTS_ROOT)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target_dir = publish_root / stamp
    target_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for path_value in (BASELINE_EVAL_OUTPUT, CALIBRATION_OUTPUT, HOLDOUT_VALIDATION_OUTPUT):
        source_path = Path(path_value)
        if source_path.exists():
            destination = target_dir / source_path.name
            shutil.copy2(source_path, destination)
            copied.append(str(destination))

    summary = {
        'created_at': stamp,
        'copied': copied,
        'calibration_recommended': (globals().get('calibration_result') or {}).get('recommended', {}),
        'holdout_recommended': (globals().get('holdout_result') or {}).get('recommended', {}),
    }
    (target_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    copied.append(str(target_dir / 'summary.json'))
    print('[PUBLISH] copied result artifacts')
    print(json.dumps(copied, indent=2))

    if AUTO_COMMIT_PUSH_RESULTS:
        subprocess.run(['git', 'add', str(target_dir)], check=True)
        commit_message = f'Add router calibration results {stamp}'
        commit = subprocess.run(['git', 'commit', '-m', commit_message], check=False)
        if commit.returncode == 0:
            push = subprocess.run(['git', 'push', 'origin', 'master'], check=False)
            print(f'[PUBLISH] git push returncode={push.returncode}')
        else:
            print(f'[PUBLISH] git commit skipped/failed returncode={commit.returncode}')
else:
    print('[PUBLISH] skipped')
