#!/usr/bin/env python3
"""Quick sanity check for VLM profile policy behavior.

This script loads base/colab configs, applies each profile, and prints:
- active profile
- SAM3 stage order
- threshold adaptation policy and effective threshold
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.router.vlm_pipeline import VLMPipeline


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_config(config_path: Path, input_threshold: float = 0.4):
    config = load_config(config_path)
    pipeline = VLMPipeline(config=config, device="cpu")

    profiles_cfg = pipeline._base_vlm_config.get("profiles", {})
    profiles = ["balanced"]
    if isinstance(profiles_cfg, dict):
        profiles = sorted(set(["balanced", *profiles_cfg.keys()]))

    print(f"\n=== {config_path.name} ===")
    for profile_name in profiles:
        pipeline.set_runtime_profile(profile_name, suppress_warning=True)
        stage_order = pipeline._sam3_stage_order()
        multiplier = pipeline._policy_value("execution", "confidence_threshold_multiplier", 1.0)
        min_thr = pipeline._policy_value("execution", "confidence_threshold_min", 0.0)
        max_thr = pipeline._policy_value("execution", "confidence_threshold_max", 1.0)
        effective = pipeline._resolve_effective_confidence_threshold(input_threshold)

        print(f"profile={profile_name}")
        print(f"  stage_order={stage_order}")
        print(
            "  threshold_policy="
            f"multiplier={multiplier}, min={min_thr}, max={max_thr}, "
            f"input={input_threshold}, effective={effective:.4f}"
        )


def main():
    for rel in ("config/base.json", "config/colab.json"):
        evaluate_config(ROOT / rel)


if __name__ == "__main__":
    main()
