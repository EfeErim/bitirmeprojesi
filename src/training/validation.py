"""Validation helpers for continual training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from src.ood.sure_scoring import compute_ds_f1
from src.training.types import EvaluationArtifactsPayload, ValidationReport


@dataclass
class _ArtifactMetricState:
    conformal_hits: int = 0
    conformal_total: int = 0
    conformal_set_size_total: int = 0
    mixed_ood_labels: List[int] = field(default_factory=list)
    mixed_ood_scores: List[float] = field(default_factory=list)
    semantic_labels: List[int] = field(default_factory=list)
    semantic_preds: List[int] = field(default_factory=list)
    confidence_labels: List[int] = field(default_factory=list)
    confidence_preds: List[int] = field(default_factory=list)


def _build_confusion_matrix(labels: torch.Tensor, preds: torch.Tensor, num_classes: int) -> torch.Tensor:
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    if labels.numel() <= 0 or preds.numel() <= 0:
        return confusion
    valid_mask = (labels >= 0) & (labels < num_classes) & (preds >= 0) & (preds < num_classes)
    if not bool(valid_mask.any().item()):
        return confusion
    valid_labels = labels[valid_mask]
    valid_preds = preds[valid_mask]
    flat_index = (valid_labels * num_classes + valid_preds).to(torch.long)
    counts = torch.bincount(flat_index, minlength=num_classes * num_classes)
    confusion = counts.reshape(num_classes, num_classes)
    return confusion


def _update_detector_artifact_state_for_batch(
    state: _ArtifactMetricState,
    *,
    detector: Any,
    idx_to_class: Dict[int, str],
    features: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    ood_loader_present: bool,
    is_ood_loader: bool,
) -> None:
    ood = detector.score(features, logits, predicted_labels=predictions)

    if ood_loader_present:
        batch_ood_labels = [1 if is_ood_loader else 0] * int(labels.shape[0])
        state.mixed_ood_labels.extend(batch_ood_labels)
        state.mixed_ood_scores.extend(float(score) for score in ood["ensemble_score"].detach().cpu().tolist())

    if getattr(detector, "sure_enabled", False) and ood_loader_present and "sure_semantic_ood" in ood:
        state.semantic_labels.extend([1 if is_ood_loader else 0] * int(labels.shape[0]))
        state.semantic_preds.extend(int(flag) for flag in ood["sure_semantic_ood"].detach().cpu().tolist())
        if is_ood_loader:
            state.confidence_labels.extend([1] * int(labels.shape[0]))
        else:
            state.confidence_labels.extend(int(flag) for flag in (predictions != labels).detach().cpu().tolist())
        state.confidence_preds.extend(int(flag) for flag in ood["sure_confidence_reject"].detach().cpu().tolist())

    if (
        not is_ood_loader
        and getattr(detector, "conformal_enabled", False)
        and getattr(detector, "conformal_qhat", None) is not None
    ):
        for feat, logit, label in zip(features, logits, labels):
            pred_set = detector.build_conformal_set(feat, logit, idx_to_class)
            true_label = idx_to_class.get(int(label.item()), str(int(label.item())))
            state.conformal_hits += int(true_label in pred_set)
            state.conformal_total += 1
            state.conformal_set_size_total += len(pred_set)


def _compute_classification_metrics(
    confusion: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float, float, float]:
    tp = confusion.diag().to(torch.float32)
    support = confusion.sum(dim=1).to(torch.float32)
    predicted_support = confusion.sum(dim=0).to(torch.float32)

    precision = torch.zeros_like(tp)
    recall = torch.zeros_like(tp)
    precision_mask = predicted_support > 0
    recall_mask = support > 0
    precision[precision_mask] = tp[precision_mask] / predicted_support[precision_mask]
    recall[recall_mask] = tp[recall_mask] / support[recall_mask]

    f1 = torch.zeros_like(tp)
    denom = precision + recall
    valid_f1 = denom > 0
    f1[valid_f1] = (2.0 * precision[valid_f1] * recall[valid_f1]) / denom[valid_f1]

    supported_class_mask = support > 0
    if bool(supported_class_mask.any().item()):
        macro_precision = float(precision[supported_class_mask].mean().item())
        macro_recall = float(recall[supported_class_mask].mean().item())
        macro_f1 = float(f1[supported_class_mask].mean().item())
        balanced_accuracy = macro_recall
    else:
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0
        balanced_accuracy = 0.0

    return tp, support, precision, recall, f1, macro_precision, macro_recall, macro_f1, balanced_accuracy


def _build_per_class_metrics(
    trainer: Any,
    *,
    num_classes: int,
    recall: torch.Tensor,
    recall_mask: torch.Tensor,
    support: torch.Tensor,
) -> tuple[Dict[str, float], Dict[str, int]]:
    idx_to_class = {idx: name for name, idx in trainer.class_to_idx.items()}
    per_class_accuracy: Dict[str, float] = {}
    per_class_support: Dict[str, int] = {}
    for class_index in range(num_classes):
        class_name = str(idx_to_class.get(class_index, class_index))
        per_class_accuracy[class_name] = (
            float(recall[class_index].item()) if bool(recall_mask[class_index].item()) else 0.0
        )
        per_class_support[class_name] = int(support[class_index].item())
    return per_class_accuracy, per_class_support


def _rank_worst_classes(
    *,
    num_classes: int,
    class_to_idx: Dict[str, int],
    per_class_accuracy: Dict[str, float],
    per_class_support: Dict[str, int],
) -> List[Dict[str, Any]]:
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    def _worst_class_sort_key(item: Dict[str, Any]) -> tuple[float, int, str]:
        return (float(item["accuracy"]), -int(item["support"]), str(item["class_name"]))

    return sorted(
        [
            {
                "class_name": class_name,
                "class_index": int(class_index),
                "accuracy": float(per_class_accuracy[class_name]),
                "support": int(per_class_support[class_name]),
            }
            for class_index, class_name in [(idx, str(idx_to_class.get(idx, idx))) for idx in range(num_classes)]
            if per_class_support[class_name] > 0
        ],
        key=_worst_class_sort_key,
    )


def _can_collect_detector_artifacts(trainer: Any, detector: Any) -> bool:
    if detector is None or not hasattr(detector, "calibration_issue"):
        return False
    if detector.calibration_issue() is not None:
        return False
    idx_to_class = {idx: name for name, idx in dict(getattr(trainer, "class_to_idx", {})).items()}
    return bool(idx_to_class)


def _update_detector_artifact_state(
    state: _ArtifactMetricState,
    *,
    detector: Any,
    idx_to_class: Dict[int, str],
    trainer: Any,
    eval_loader: Iterable[Dict[str, torch.Tensor]],
    ood_loader_present: bool,
    is_ood_loader: bool,
) -> None:
    trainer.set_eval_mode()
    with torch.no_grad():
        for batch in eval_loader:
            images = batch["images"].to(trainer.device)
            labels = batch["labels"].to(trainer.device)
            features = trainer.encode(images)
            logits = trainer.classifier(features)
            predictions = torch.argmax(logits, dim=1)
            _update_detector_artifact_state_for_batch(
                state,
                detector=detector,
                idx_to_class=idx_to_class,
                features=features,
                logits=logits,
                labels=labels,
                predictions=predictions,
                ood_loader_present=ood_loader_present,
                is_ood_loader=is_ood_loader,
            )


def _finalize_artifact_metric_state(payload: EvaluationArtifactsPayload, state: _ArtifactMetricState) -> None:
    if state.conformal_total > 0:
        payload.conformal_empirical_coverage = float(state.conformal_hits) / float(state.conformal_total)
        payload.conformal_avg_set_size = float(state.conformal_set_size_total) / float(state.conformal_total)
        payload.context["conformal_eval_samples"] = int(state.conformal_total)

    if state.mixed_ood_labels and 0 in state.mixed_ood_labels and 1 in state.mixed_ood_labels:
        payload.ood_labels = list(state.mixed_ood_labels)
        payload.ood_scores = list(state.mixed_ood_scores)
        payload.context["ood_eval_in_distribution_samples"] = int(sum(1 for item in state.mixed_ood_labels if item == 0))
        payload.context["ood_eval_ood_samples"] = int(sum(1 for item in state.mixed_ood_labels if item == 1))

    if state.semantic_labels and 0 in state.semantic_labels and 1 in state.semantic_labels and state.confidence_labels:
        ds_f1 = compute_ds_f1(
            torch.tensor(state.semantic_labels, dtype=torch.long),
            torch.tensor(state.confidence_labels, dtype=torch.long),
            torch.tensor(state.semantic_preds, dtype=torch.long),
            torch.tensor(state.confidence_preds, dtype=torch.long),
        )
        payload.sure_ds_f1 = float(ds_f1["ds_f1"])
        payload.context["sure_semantic_f1"] = float(ds_f1["semantic_f1"])
        payload.context["sure_confidence_f1"] = float(ds_f1["confidence_f1"])


def _evaluate_model_core(
    trainer: Any,
    loader: Iterable[Dict[str, torch.Tensor]],
    *,
    detector: Any = None,
    artifact_state: Optional[_ArtifactMetricState] = None,
    idx_to_class: Optional[Dict[int, str]] = None,
    ood_loader_present: bool = False,
) -> Optional[Tuple[ValidationReport, List[int], List[int]]]:
    if getattr(trainer, "classifier", None) is None:
        return None

    trainer.set_eval_mode()

    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    all_labels: List[torch.Tensor] = []
    all_preds: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(trainer.device)
            labels = batch["labels"].to(trainer.device)

            if artifact_state is not None and detector is not None:
                features = trainer.encode(images)
                logits = trainer.classifier(features)
            else:
                features = None
                logits = trainer.forward_logits(images)

            loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=0.0)

            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * float(batch_size)
            total_samples += batch_size

            predictions = torch.argmax(logits, dim=1)
            total_correct += int((predictions == labels).sum().item())
            all_labels.append(labels.detach().cpu())
            all_preds.append(predictions.detach().cpu())

            if artifact_state is not None and detector is not None and features is not None:
                _update_detector_artifact_state_for_batch(
                    artifact_state,
                    detector=detector,
                    idx_to_class=idx_to_class or {},
                    features=features,
                    logits=logits,
                    labels=labels,
                    predictions=predictions,
                    ood_loader_present=ood_loader_present,
                    is_ood_loader=False,
                )

    if total_samples <= 0:
        return None

    val_loss = float(total_loss / max(1, total_samples))
    val_accuracy = float(total_correct / max(1, total_samples))

    labels = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long)
    preds = torch.cat(all_preds, dim=0) if all_preds else torch.empty(0, dtype=torch.long)

    num_classes = max(1, int(getattr(trainer.classifier, "out_features", 1)))
    confusion = (
        _build_confusion_matrix(labels, preds, num_classes)
        if labels.numel() > 0
        else torch.zeros((num_classes, num_classes), dtype=torch.long)
    )
    (
        _tp,
        support,
        _precision,
        recall,
        f1,
        macro_precision,
        macro_recall,
        macro_f1,
        balanced_accuracy,
    ) = _compute_classification_metrics(confusion)
    recall_mask = support > 0

    weighted_f1 = float((f1 * support).sum().item() / max(1.0, float(support.sum().item())))

    per_class_accuracy, per_class_support = _build_per_class_metrics(
        trainer,
        num_classes=num_classes,
        recall=recall,
        recall_mask=recall_mask,
        support=support,
    )
    ranked_worst = _rank_worst_classes(
        num_classes=num_classes,
        class_to_idx=trainer.class_to_idx,
        per_class_accuracy=per_class_accuracy,
        per_class_support=per_class_support,
    )

    report = ValidationReport(
        val_loss=val_loss,
        val_accuracy=val_accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        balanced_accuracy=balanced_accuracy,
        per_class_accuracy=per_class_accuracy,
        per_class_support=per_class_support,
        worst_classes=ranked_worst[:3],
    )
    return report, labels.tolist(), preds.tolist()


def evaluate_model(trainer: Any, loader: Iterable[Dict[str, torch.Tensor]]) -> Optional[ValidationReport]:
    result = _evaluate_model_core(trainer, loader)
    return None if result is None else result[0]


def evaluate_model_with_predictions(
    trainer: Any,
    loader: Iterable[Dict[str, torch.Tensor]],
) -> Optional[Tuple[ValidationReport, List[int], List[int]]]:
    return _evaluate_model_core(trainer, loader)


def evaluate_model_with_artifact_metrics(
    trainer: Any,
    loader: Iterable[Dict[str, torch.Tensor]],
    *,
    ood_loader: Optional[Iterable[Dict[str, torch.Tensor]]] = None,
) -> Optional[EvaluationArtifactsPayload]:
    detector = getattr(trainer, "ood_detector", None)
    idx_to_class = {idx: name for name, idx in dict(getattr(trainer, "class_to_idx", {})).items()}
    can_collect = _can_collect_detector_artifacts(trainer, detector)
    state = _ArtifactMetricState() if can_collect else None

    result = _evaluate_model_core(
        trainer,
        loader,
        detector=detector if can_collect else None,
        artifact_state=state,
        idx_to_class=idx_to_class,
        ood_loader_present=ood_loader is not None,
    )
    if result is None:
        return None

    report, y_true, y_pred = result
    payload = EvaluationArtifactsPayload(report=report, y_true=y_true, y_pred=y_pred)
    if not can_collect or state is None:
        return payload

    if ood_loader is not None:
        _update_detector_artifact_state(
            state,
            detector=detector,
            idx_to_class=idx_to_class,
            trainer=trainer,
            eval_loader=ood_loader,
            ood_loader_present=True,
            is_ood_loader=True,
        )
    _finalize_artifact_metric_state(payload, state)

    return payload
