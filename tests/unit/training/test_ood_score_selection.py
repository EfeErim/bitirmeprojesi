from src.training.services.ood_score_selection import (
    apply_primary_score_method_to_evaluation,
    normalize_requested_primary_score_method,
    select_best_ood_score_method,
)
from src.training.types import EvaluationArtifactsPayload, ValidationReport


def _evaluation_payload() -> EvaluationArtifactsPayload:
    return EvaluationArtifactsPayload(
        report=ValidationReport.from_dict({}),
        y_true=[0, 1],
        y_pred=[0, 1],
        ood_labels=[0, 1],
        ood_scores=[0.1, 0.9],
        ood_primary_score_method="ensemble",
        ood_scores_by_method={
            "ensemble": [0.4, 0.6],
            "energy": [0.1, 0.9],
            "knn": [0.2, 0.8],
        },
        context={},
    )


def test_normalize_requested_primary_score_method_accepts_auto():
    assert normalize_requested_primary_score_method("AUTO") == "auto"


def test_select_best_ood_score_method_prefers_best_auroc_then_lower_fpr():
    selected = select_best_ood_score_method(
        {
            "ensemble": {"ood_auroc": 0.91, "ood_false_positive_rate": 0.07},
            "energy": {"ood_auroc": 0.95, "ood_false_positive_rate": 0.09},
            "knn": {"ood_auroc": 0.95, "ood_false_positive_rate": 0.11},
        }
    )

    assert selected == "energy"


def test_apply_primary_score_method_to_evaluation_rewrites_selected_scores():
    rewritten = apply_primary_score_method_to_evaluation(
        _evaluation_payload(),
        "energy",
        requested_primary_score_method="auto",
        selection_source="real_ood_split",
    )

    assert rewritten is not None
    assert rewritten.ood_primary_score_method == "energy"
    assert rewritten.ood_scores == [0.1, 0.9]
    assert rewritten.context["ood_requested_primary_score_method"] == "auto"
    assert rewritten.context["ood_primary_score_selection_source"] == "real_ood_split"
