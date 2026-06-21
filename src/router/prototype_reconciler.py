"""Taxonomy-aware prototype reconciliation for router handoff decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.router.label_normalization import normalize_part_label
from src.router.prototype_bank import DEFAULT_BACKEND, euclidean_distance, image_vector
from src.router.taxonomy_registry import make_target_id, normalize_crop_name
from src.shared.json_utils import read_json, write_json

DEFAULT_MIN_SIMILARITY = 0.20
DEFAULT_MIN_MARGIN = 0.03
DEFAULT_ALLOW_TAXONOMY_CORRECTION = True
TRUSTED_ROUTER_STATUSES = {"ok", "trusted_hint_skipped", "skipped"}


@dataclass(frozen=True)
class PrototypeMatch:
    target_id: str | None
    crop: str | None
    part: str | None
    similarity: float
    distance: float
    second_target_id: str | None = None
    second_similarity: float | None = None
    margin: float = 0.0


@dataclass(frozen=True)
class ReconcileDecision:
    decision: str
    crop: str | None
    part: str | None
    reason: str
    taxonomy_relation: str
    prototype_match: PrototypeMatch | None
    vlm_crop: str | None
    vlm_part: str | None
    min_similarity: float
    min_margin: float

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        match = payload.pop("prototype_match")
        payload["prototype_match"] = match
        payload["prototype_crop"] = match.get("crop") if isinstance(match, dict) else None
        payload["prototype_part"] = match.get("part") if isinstance(match, dict) else None
        payload["prototype_target"] = match.get("target_id") if isinstance(match, dict) else None
        payload["prototype_similarity"] = match.get("similarity") if isinstance(match, dict) else None
        payload["prototype_margin"] = match.get("margin") if isinstance(match, dict) else None
        payload["reconciled_crop"] = self.crop
        payload["reconciled_part"] = self.part
        payload["reconcile_decision"] = self.decision
        return payload


def _norm_text(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"unknown", "none", "null"}:
        return None
    return text or None


def _similarity_from_distance(distance: float) -> float:
    return round(1.0 / (1.0 + max(0.0, float(distance))), 8)


def load_target_taxonomy(registry_payload: dict[str, Any] | Path | str | None) -> dict[str, dict[str, Any]]:
    if registry_payload is None:
        return {}
    if isinstance(registry_payload, (str, Path)):
        registry_payload = read_json(registry_payload, default={}, expect_type=dict)
    entries = registry_payload.get("targets", []) if isinstance(registry_payload, dict) else []
    return {
        str(entry.get("target_id") or ""): dict(entry)
        for entry in entries
        if isinstance(entry, dict) and str(entry.get("target_id") or "").strip()
    }


def taxonomy_relation(
    router_crop: str | None,
    prototype_target_id: str | None,
    target_taxonomy: dict[str, dict[str, Any]],
) -> str:
    router_crop_key = normalize_crop_name(router_crop) if _norm_text(router_crop) else ""
    if not router_crop_key:
        return "router_unknown"
    if not prototype_target_id or prototype_target_id not in target_taxonomy:
        return "prototype_unknown"
    prototype = target_taxonomy[prototype_target_id]
    prototype_crop = normalize_crop_name(prototype.get("crop_canonical_name") or prototype.get("crop") or "")
    names = {
        prototype_crop,
        *[normalize_crop_name(value) for value in prototype.get("common_names", [])],
        *[normalize_crop_name(value) for value in prototype.get("synonyms", [])],
    }
    if router_crop_key in names:
        return "same_crop"

    router_family = router_genus = None
    for entry in target_taxonomy.values():
        entry_names = {
            normalize_crop_name(entry.get("crop_canonical_name") or ""),
            *[normalize_crop_name(value) for value in entry.get("common_names", [])],
            *[normalize_crop_name(value) for value in entry.get("synonyms", [])],
        }
        if router_crop_key in entry_names:
            router_family = _norm_text(entry.get("family"))
            router_genus = _norm_text(entry.get("genus"))
            break

    prototype_family = _norm_text(prototype.get("family"))
    prototype_genus = _norm_text(prototype.get("genus"))
    if router_genus and prototype_genus and router_genus == prototype_genus:
        return "same_genus"
    if router_family and prototype_family and router_family == prototype_family:
        return "same_family"
    return "distant_or_unknown"


def nearest_target(
    image_path: str | Path,
    prototype_payload: dict[str, Any] | Path | str,
) -> PrototypeMatch:
    if isinstance(prototype_payload, (str, Path)):
        prototype_payload = read_json(prototype_payload, default={}, expect_type=dict)
    embedding_backend = str(prototype_payload.get("embedding_backend") or DEFAULT_BACKEND)
    source_roots = prototype_payload.get("source_roots", {}) if isinstance(prototype_payload.get("source_roots"), dict) else {}
    query_vector = image_vector(
        Path(image_path),
        embedding_backend=embedding_backend,
        embedding_model_id=str(source_roots.get("embedding_model_id") or "imageomics/bioclip-2.5-vith14"),
        device=str(source_roots.get("embedding_device") or "cpu"),
    )
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for target_id, target in sorted((prototype_payload.get("target_prototypes") or {}).items()):
        if not isinstance(target, dict):
            continue
        centroid = tuple(float(value) for value in target.get("centroid", []) if value is not None)
        if not centroid:
            continue
        distance = euclidean_distance(query_vector, centroid)
        scored.append((distance, str(target_id), target))

    if not scored:
        return PrototypeMatch(target_id=None, crop=None, part=None, similarity=0.0, distance=0.0)

    scored.sort(key=lambda item: (item[0], item[1]))
    best_distance, best_target_id, best_target = scored[0]
    best_similarity = _similarity_from_distance(best_distance)
    second_target_id = None
    second_similarity = None
    if len(scored) > 1:
        second_distance, second_target_id, _ = scored[1]
        second_similarity = _similarity_from_distance(second_distance)
    margin = round(best_similarity - float(second_similarity or 0.0), 8)
    return PrototypeMatch(
        target_id=best_target_id,
        crop=_norm_text(best_target.get("crop")),
        part=_norm_text(best_target.get("part")),
        similarity=best_similarity,
        distance=round(best_distance, 8),
        second_target_id=second_target_id,
        second_similarity=second_similarity,
        margin=margin,
    )


def reconcile_router_handoff(
    *,
    image_path: str | Path,
    router_crop: str | None,
    router_part: str | None,
    router_status: str | None,
    prototype_payload: dict[str, Any] | Path | str,
    registry_payload: dict[str, Any] | Path | str | None,
    supported_targets: set[str] | None = None,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    min_margin: float = DEFAULT_MIN_MARGIN,
    allow_taxonomy_correction: bool = DEFAULT_ALLOW_TAXONOMY_CORRECTION,
) -> ReconcileDecision:
    vlm_crop = normalize_crop_name(router_crop) if router_crop else None
    if vlm_crop in {"unknown", "none", "null"}:
        vlm_crop = None
    vlm_part = normalize_part_label(router_part) if _norm_text(router_part) else None
    if vlm_part in {"unknown", "none", "null"}:
        vlm_part = None
    status = str(router_status or "").strip().lower()
    match = nearest_target(image_path, prototype_payload)
    target_taxonomy = load_target_taxonomy(registry_payload)
    relation = taxonomy_relation(vlm_crop, match.target_id, target_taxonomy)
    supported = set(supported_targets or set((target_taxonomy or {}).keys()))

    if not match.target_id or match.target_id not in supported:
        return ReconcileDecision(
            decision="abstain",
            crop=vlm_crop,
            part=vlm_part,
            reason="prototype_target_not_supported",
            taxonomy_relation=relation,
            prototype_match=match,
            vlm_crop=vlm_crop,
            vlm_part=vlm_part,
            min_similarity=min_similarity,
            min_margin=min_margin,
        )

    if match.similarity < min_similarity or match.margin < min_margin:
        return ReconcileDecision(
            decision="abstain",
            crop=vlm_crop,
            part=vlm_part,
            reason="prototype_evidence_weak",
            taxonomy_relation=relation,
            prototype_match=match,
            vlm_crop=vlm_crop,
            vlm_part=vlm_part,
            min_similarity=min_similarity,
            min_margin=min_margin,
        )

    prototype_crop = match.crop
    prototype_part = match.part
    router_target = make_target_id(vlm_crop, vlm_part) if vlm_crop and vlm_part else None
    router_is_trusted = status in TRUSTED_ROUTER_STATUSES
    router_target_is_supported = bool(router_target and router_target in supported)
    if router_target == match.target_id and router_is_trusted:
        return ReconcileDecision(
            decision="accept_router",
            crop=vlm_crop,
            part=vlm_part,
            reason="router_and_prototype_agree",
            taxonomy_relation=relation,
            prototype_match=match,
            vlm_crop=vlm_crop,
            vlm_part=vlm_part,
            min_similarity=min_similarity,
            min_margin=min_margin,
        )

    if vlm_part and prototype_part and vlm_part != prototype_part and router_is_trusted and router_target_is_supported:
        return ReconcileDecision(
            decision="abstain",
            crop=vlm_crop,
            part=vlm_part,
            reason="part_conflict",
            taxonomy_relation=relation,
            prototype_match=match,
            vlm_crop=vlm_crop,
            vlm_part=vlm_part,
            min_similarity=min_similarity,
            min_margin=min_margin,
        )

    taxonomy_promotable = relation in {"same_crop", "same_genus", "same_family", "router_unknown"}
    prototype_override_promotable = not router_is_trusted or not router_target_is_supported
    if allow_taxonomy_correction and (taxonomy_promotable or prototype_override_promotable):
        return ReconcileDecision(
            decision="use_prototype",
            crop=prototype_crop,
            part=prototype_part,
            reason=(
                "prototype_corrected_router_handoff"
                if taxonomy_promotable
                else "prototype_overrode_untrusted_router_handoff"
            ),
            taxonomy_relation=relation,
            prototype_match=match,
            vlm_crop=vlm_crop,
            vlm_part=vlm_part,
            min_similarity=min_similarity,
            min_margin=min_margin,
        )

    return ReconcileDecision(
        decision="abstain",
        crop=vlm_crop,
        part=vlm_part,
        reason="taxonomy_relation_not_promotable",
        taxonomy_relation=relation,
        prototype_match=match,
        vlm_crop=vlm_crop,
        vlm_part=vlm_part,
        min_similarity=min_similarity,
        min_margin=min_margin,
    )


def write_reconciliation_report(payload: dict[str, Any], output_path: Path) -> Path:
    return write_json(output_path, payload, ensure_ascii=False, sort_keys=False)
