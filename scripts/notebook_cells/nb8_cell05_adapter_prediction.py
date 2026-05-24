# Auto-extracted from colab_notebooks/8_auto_router_adapter_prediction.ipynb cell 5.
# Keep notebook execute-only cells thin; edit behavior here.

import json

from scripts.colab_auto_router_adapter_prediction import run_auto_router_adapter_prediction

ROUTER_RESULT = dict(globals().get("result") or {})
ADAPTER_ROOT = globals().get("ADAPTER_ROOT", None)
RETURN_OOD = bool(globals().get("RETURN_OOD", True))
PRINT_JSON_RESULT = bool(globals().get("PRINT_JSON_RESULT", False))

if ANALYSIS_IMAGE_PATH is None:
    raise ValueError("ANALYSIS_IMAGE_PATH is required before adapter prediction.")
if not ROUTER_RESULT:
    raise ValueError("Run the Notebook 1 router analysis cell before adapter prediction.")

auto_result = run_auto_router_adapter_prediction(
    ANALYSIS_IMAGE_PATH,
    router_result=ROUTER_RESULT,
    config_env=CONFIG_ENV,
    device=DEVICE,
    adapter_root=ADAPTER_ROOT,
    return_ood=RETURN_OOD,
    status_printer=print,
)

print(
    "[AUTO] "
    f"status={auto_result.get('status')} "
    f"crop={auto_result.get('crop') or 'unknown'} "
    f"part={auto_result.get('part') or 'unknown'}"
)

diagnosis = auto_result.get("diagnosis")
if diagnosis:
    print(
        "[ADAPTER] "
        f"diagnosis={diagnosis} "
        f"confidence={float(auto_result.get('confidence', 0.0) or 0.0):.3f}"
    )
else:
    message = str(auto_result.get("message", "") or "")
    if message:
        print(f"[ADAPTER] prediction unavailable: {message}")

ood = auto_result.get("ood_analysis")
if isinstance(ood, dict):
    print(
        "[OOD] "
        f"is_ood={bool(ood.get('is_ood', False))} "
        f"method={ood.get('score_method')} "
        f"score={float(ood.get('primary_score', 0.0) or 0.0):.3f} "
        f"threshold={float(ood.get('decision_threshold', 0.0) or 0.0):.3f}"
    )

input_guard = auto_result.get("input_guard")
if isinstance(input_guard, dict):
    print(f"[INPUT_GUARD] decision={input_guard.get('decision')} is_plant_like={input_guard.get('is_plant_like')}")

if PRINT_JSON_RESULT:
    print(json.dumps(auto_result, indent=2))
