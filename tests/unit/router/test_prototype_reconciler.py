from pathlib import Path

from PIL import Image

from src.router.prototype_bank import build_prototype_bank
from src.router.prototype_reconciler import reconcile_router_handoff, taxonomy_relation
from src.router.taxonomy_registry import build_taxonomy_registry


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


def test_reconciler_accepts_router_when_prototype_agrees(tmp_path: Path):
    dataset_root = tmp_path / "datasets"
    image_path = dataset_root / "tomato__leaf" / "train" / "healthy" / "a.png"
    _write_image(image_path, (20, 90, 40))
    prototype_payload = build_prototype_bank(dataset_root=dataset_root, created_at="20260617T000000Z")
    registry_payload = build_taxonomy_registry(dataset_root=dataset_root, adapter_root=None, created_at="20260617T000000Z")

    decision = reconcile_router_handoff(
        image_path=image_path,
        router_crop="tomato",
        router_part="leaf",
        router_status="ok",
        prototype_payload=prototype_payload,
        registry_payload=registry_payload,
        min_similarity=0.1,
        min_margin=0.0,
    )

    assert decision.decision == "accept_router"
    assert decision.crop == "tomato"
    assert decision.part == "leaf"
    assert decision.to_payload()["reconciled_crop"] == "tomato"


def test_reconciler_corrects_unknown_router_from_strong_prototype(tmp_path: Path):
    dataset_root = tmp_path / "datasets"
    tomato_image = dataset_root / "tomato__fruit" / "train" / "healthy" / "a.png"
    grape_image = dataset_root / "grape__fruit" / "train" / "healthy" / "b.png"
    _write_image(tomato_image, (190, 30, 30))
    _write_image(grape_image, (40, 20, 120))
    prototype_payload = build_prototype_bank(dataset_root=dataset_root, created_at="20260617T000000Z")
    registry_payload = build_taxonomy_registry(dataset_root=dataset_root, adapter_root=None, created_at="20260617T000000Z")

    decision = reconcile_router_handoff(
        image_path=tomato_image,
        router_crop="unknown",
        router_part="fruit",
        router_status="unknown_crop",
        prototype_payload=prototype_payload,
        registry_payload=registry_payload,
        min_similarity=0.1,
        min_margin=0.01,
    )

    assert decision.decision == "use_prototype"
    assert decision.crop == "tomato"
    assert decision.part == "fruit"
    assert decision.reason == "prototype_corrected_router_handoff"


def test_reconciler_abstains_on_part_conflict(tmp_path: Path):
    dataset_root = tmp_path / "datasets"
    image_path = dataset_root / "tomato__leaf" / "train" / "healthy" / "a.png"
    _write_image(image_path, (20, 90, 40))
    prototype_payload = build_prototype_bank(dataset_root=dataset_root, created_at="20260617T000000Z")
    registry_payload = build_taxonomy_registry(dataset_root=dataset_root, adapter_root=None, created_at="20260617T000000Z")

    decision = reconcile_router_handoff(
        image_path=image_path,
        router_crop="tomato",
        router_part="fruit",
        router_status="ok",
        prototype_payload=prototype_payload,
        registry_payload=registry_payload,
        min_similarity=0.1,
        min_margin=0.0,
    )

    assert decision.decision == "abstain"
    assert decision.reason == "part_conflict"


def test_taxonomy_relation_detects_same_family(tmp_path: Path):
    registry_payload = {
        "targets": [
            {
                "target_id": "apricot__leaf",
                "crop_canonical_name": "apricot",
                "common_names": ["apricot"],
                "synonyms": [],
                "genus": "Prunus",
                "family": "Rosaceae",
            },
            {
                "target_id": "strawberry__leaf",
                "crop_canonical_name": "strawberry",
                "common_names": ["strawberry"],
                "synonyms": [],
                "genus": "Fragaria",
                "family": "Rosaceae",
            },
        ]
    }
    taxonomy = {entry["target_id"]: entry for entry in registry_payload["targets"]}

    assert taxonomy_relation("strawberry", "apricot__leaf", taxonomy) == "same_family"
