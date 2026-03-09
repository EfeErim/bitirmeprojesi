"""Validation helpers for continual training."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from src.ood.sure_scoring import compute_ds_f1
from src.training.types import EvaluationArtifactsPayload, ValidationReport


def _evaluate_model_core(
    trainer: Any,
    loader: Iterable[Dict[str, torch.Tensor]],
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
            logits = trainer.forward_logits(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)

            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * float(batch_size)
            total_samples += batch_size

            predictions = torch.argmax(logits, dim=1)
            total_correct += int((predictions == labels).sum().item())
            all_labels.append(labels.detach().cpu())
            all_preds.append(predictions.detach().cpu())

    if total_samples <= 0:
        return None

    val_loss = float(total_loss / max(1, total_samples))
    val_accuracy = float(total_correct / max(1, total_samples))

    labels = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long)
    preds = torch.cat(all_preds, dim=0) if all_preds else torch.empty(0, dtype=torch.long)

    num_classes = max(1, int(getattr(trainer.classifier, "out_features", 1)))
    if labels.numel() > 0:
        confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
        for label, pred in zip(labels.tolist(), preds.tolist()):
            if 0 <= label < num_classes and 0 <= pred < num_classes:
                confusion[label, pred] += 1
    else:
        confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

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

    weighted_f1 = float((f1 * support).sum().item() / max(1.0, float(support.sum().item())))

    idx_to_class = {idx: name for name, idx in trainer.class_to_idx.items()}
    per_class_accuracy: Dict[str, float] = {}
    per_class_support: Dict[str, int] = {}
    for class_index in range(num_classes):
        class_name = str(idx_to_class.get(class_index, class_index))
        per_class_accuracy[class_name] = (
            float(recall[class_index].item()) if bool(recall_mask[class_index].item()) else 0.0
        )
        per_class_support[class_name] = int(support[class_index].item())

    def _worst_class_sort_key(item: Dict[str, Any]) -> tuple[float, int, str]:
        return (float(item["accuracy"]), -int(item["support"]), str(item["class_name"]))

    ranked_worst = sorted(
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
    result = _evaluate_model_core(trainer, loader)
    if result is None:
        return None

    report, y_true, y_pred = result
    payload = EvaluationArtifactsPayload(report=report, y_true=y_true, y_pred=y_pred)

    detector = getattr(trainer, "ood_detector", None)
    if detector is None or not hasattr(detector, "calibration_issue"):
        return payload
    if detector.calibration_issue() is not None:
        return payload

    idx_to_class = {idx: name for name, idx in dict(getattr(trainer, "class_to_idx", {})).items()}
    if not idx_to_class:
        return payload

    conformal_hits = 0
    conformal_total = 0
    conformal_set_size_total = 0
    mixed_ood_labels: List[int] = []
    mixed_ood_scores: List[float] = []
    semantic_labels: List[int] = []
    semantic_preds: List[int] = []
    confidence_labels: List[int] = []
    confidence_preds: List[int] = []

    def _score_loader(
        eval_loader: Iterable[Dict[str, torch.Tensor]],
        *,
        is_ood_loader: bool,
    ) -> None:
        nonlocal conformal_hits, conformal_total, conformal_set_size_total

        trainer.set_eval_mode()
        with torch.no_grad():
            for batch in eval_loader:
                images = batch["images"].to(trainer.device)
                labels = batch["labels"].to(trainer.device)
                features = trainer.encode(images)
                logits = trainer.classifier(features)
                predictions = torch.argmax(logits, dim=1)
                ood = detector.score(features, logits, predicted_labels=predictions)

                if ood_loader is not None:
                    batch_ood_labels = [1 if is_ood_loader else 0] * int(images.shape[0])
                    mixed_ood_labels.extend(batch_ood_labels)
                    mixed_ood_scores.extend(float(score) for score in ood["ensemble_score"].detach().cpu().tolist())

                if getattr(detector, "sure_enabled", False) and ood_loader is not None and "sure_semantic_ood" in ood:
                    semantic_labels.extend([1 if is_ood_loader else 0] * int(images.shape[0]))
                    semantic_preds.extend(int(flag) for flag in ood["sure_semantic_ood"].detach().cpu().tolist())
                    if is_ood_loader:
                        confidence_labels.extend([1] * int(images.shape[0]))
                    else:
                        confidence_labels.extend(
                            int(flag)
                            for flag in (predictions != labels).detach().cpu().tolist()
                        )
                    confidence_preds.extend(
                        int(flag) for flag in ood["sure_confidence_reject"].detach().cpu().tolist()
                    )

                if (
                    not is_ood_loader
                    and getattr(detector, "conformal_enabled", False)
                    and getattr(detector, "conformal_qhat", None) is not None
                ):
                    for feat, logit, label in zip(features, logits, labels):
                        pred_set = detector.build_conformal_set(feat, logit, idx_to_class)
                        true_label = idx_to_class.get(int(label.item()), str(int(label.item())))
                        conformal_hits += int(true_label in pred_set)
                        conformal_total += 1
                        conformal_set_size_total += len(pred_set)

    _score_loader(loader, is_ood_loader=False)
    if ood_loader is not None:
        _score_loader(ood_loader, is_ood_loader=True)

    if conformal_total > 0:
        payload.conformal_empirical_coverage = float(conformal_hits) / float(conformal_total)
        payload.conformal_avg_set_size = float(conformal_set_size_total) / float(conformal_total)
        payload.context["conformal_eval_samples"] = int(conformal_total)

    if mixed_ood_labels and 0 in mixed_ood_labels and 1 in mixed_ood_labels:
        payload.ood_labels = list(mixed_ood_labels)
        payload.ood_scores = list(mixed_ood_scores)
        payload.context["ood_eval_in_distribution_samples"] = int(sum(1 for item in mixed_ood_labels if item == 0))
        payload.context["ood_eval_ood_samples"] = int(sum(1 for item in mixed_ood_labels if item == 1))

    if semantic_labels and 0 in semantic_labels and 1 in semantic_labels and confidence_labels:
        ds_f1 = compute_ds_f1(
            torch.tensor(semantic_labels, dtype=torch.long),
            torch.tensor(confidence_labels, dtype=torch.long),
            torch.tensor(semantic_preds, dtype=torch.long),
            torch.tensor(confidence_preds, dtype=torch.long),
        )
        payload.sure_ds_f1 = float(ds_f1["ds_f1"])
        payload.context["sure_semantic_f1"] = float(ds_f1["semantic_f1"])
        payload.context["sure_confidence_f1"] = float(ds_f1["confidence_f1"])

    return payload
