---
name: aads-training-lifecycle
description: Maintain AADS v6 continual SD-LoRA training behavior and adapter lifecycle invariants across local and Colab flows. Use for continual engine logic, quantization/fusion/OOD threshold behavior, and training runtime component updates.
---

# AADS Training Lifecycle

## Workflow

1. Identify continual scope.
- Determine whether change targets continual engine config, trainer runtime, adapter lifecycle, or cross-module behavior.
- Load `references/training-invariants.md`.

2. Preserve lifecycle invariants.
- Preserve v6 continual adaptation semantics.
- Preserve OOD-threshold path compatibility with adapters.
- Preserve trainer runtime/component contract behavior.

3. Enforce local/Colab parity.
- Review paired local and `colab_*` trainer variants when behavior changes.
- Include notebook/script entrypoint checks when user-facing training flow changes.

4. Run phase-aligned validation.
- Run targeted training unit tests.
- Run Colab smoke training checks when applicable.
- Add integration checks only when required.

5. Emit training delta summary.
- Report continual behavior changes and invariants status.

## Output Contract

Return sections in this order:
1. Continual scope
2. Files changed
3. Invariants preserved/changed
4. Validation commands and results
5. Risks and follow-up checks
6. Local-Colab parity notes

## References

- Read `references/training-invariants.md` before implementing.
