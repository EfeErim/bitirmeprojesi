# Presentation Outline

Last updated: 2026-06-16

Use this file for M4. The presentation should use Turkish narration with English technical terms.

## Slide Plan

1. Title and one-sentence goal
   - Turkish narration: bitki fotoğrafından bitki/part tanıma, doğru adapter seçme, hastalık tahmini veya güvenli unknown.
   - English technical terms: router, adapter, OOD, abstention.

2. Problem and motivation
   - Users may upload different plant photos.
   - A wrong confident disease label is worse than a clear unknown/review result.

3. Final scope
   - Demo product, not web/mobile production app.
   - Supported final crop/part surfaces: tomato, strawberry, grape, apricot, fruit and leaf where reliable.
   - Colab + GitHub + handoff guide.

4. System architecture
   - Image input.
   - Router crop/part decision.
   - Adapter selection.
   - Disease prediction.
   - OOD/unknown/review status.

5. Training and artifact flow
   - Notebook 0 dataset preparation.
   - Notebook 2 adapter training.
   - Notebook 3 adapter validation.
   - `production_readiness.json` as deployability evidence.

6. Live demo flow
   - Notebook 8.
   - Show supported example.
   - Show difficult example.
   - Show unknown/unsupported example.
   - Show output fields: crop, part, disease, confidence/OOD, status.

7. Demo checklist results
   - Number of images.
   - Answered count.
   - Abstained/reviewed count.
   - Failed count.
   - Per-target support labels.

8. Safety and unknown policy
   - Explain why unknown can be correct.
   - Mention selective classification / risk-coverage at a high level.
   - Make the failure definition explicit: wrong disease, known disease marked unknown without reason, unknown mapped to known disease.

9. Evidence-gate and Notebook 16
   - Keep this short.
   - Position Notebook 16 as review-signal research/evidence analysis.
   - State that ROI/bbox evidence does not override runtime disease decisions unless promotion gates pass.

10. Handoff package
    - GitHub repo.
    - Colab notebooks.
    - Demo checklist.
    - Handoff guide.
    - Final validation checklist.

11. Limitations
    - Supported crops/parts only.
    - User photo guidance matters.
    - Some target surfaces may be `low_confidence` or `experimental`.
    - External model access can require Colab/token setup.

12. Final conclusion
    - The project is complete when it gives reliable answers where supported and safe unknown/review behavior where unsupported.

## Required Visuals

- One architecture diagram.
- One Notebook 8 output screenshot.
- One demo checklist result table.
- One supported-target status table.
- One fallback screenshot/output for external dependency risk.

## Demo Script

1. Open Notebook 8.
2. Run a clear supported example.
3. Explain router and adapter output.
4. Run a difficult supported example.
5. Run an unsupported or unknown example.
6. Explain why unknown/review is safer than a wrong disease label.
7. Show checklist summary instead of relying only on the live run.

## Rehearsal Checklist

- Notebook opens cleanly.
- Required credentials/tokens are ready.
- Demo images are accessible.
- Fallback screenshots are ready.
- Expected outputs are known before the presentation.
- No code edits are required during the demo.
