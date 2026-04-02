"""Human-readable guided artifact indexes for notebook and workflow outputs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from src.shared.json_utils import read_json, write_json

_PRIORITY_ORDER = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}

_GUIDED_START_HERE = "guided/00_start_here.md"
_GUIDED_CATALOG = "guided/02_file_catalog.json"


def _relative_path(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except Exception:
        try:
            return Path(os.path.relpath(path.resolve(), base_dir.resolve())).as_posix()
        except Exception:
            return path.as_posix()


def _path_format(path: Path) -> str:
    if path.is_dir():
        return "directory"
    suffix = path.suffix.lower().lstrip(".")
    return suffix or "file"


def _entry_from_path(
    path: Path,
    *,
    base_dir: Path,
    category: str,
    priority: str,
    title_tr: str,
    description_tr: str,
    reader_goal: str,
    generated_by: str,
    decision_importance: str,
    read_order: int,
) -> Dict[str, Any]:
    return {
        "relative_path": _relative_path(path, base_dir),
        "category": str(category),
        "priority": str(priority),
        "title_tr": str(title_tr),
        "description_tr": str(description_tr),
        "reader_goal": str(reader_goal),
        "format": _path_format(path),
        "generated_by": str(generated_by),
        "decision_importance": str(decision_importance),
        "read_order": int(read_order),
        "exists": bool(path.exists()),
        "is_directory": bool(path.is_dir()),
        "size_bytes": None if not path.exists() or path.is_dir() else int(path.stat().st_size),
    }


def _coerce_extra_entry(entry: Dict[str, Any], *, base_dir: Path) -> Dict[str, Any]:
    raw_path = entry.get("path")
    if not raw_path:
        raise ValueError("Extra guided entry requires a 'path' field.")
    resolved_path = Path(raw_path)
    return _entry_from_path(
        resolved_path,
        base_dir=base_dir,
        category=str(entry.get("category", "supporting")),
        priority=str(entry.get("priority", "medium")),
        title_tr=str(entry.get("title_tr", resolved_path.name)),
        description_tr=str(entry.get("description_tr", "")),
        reader_goal=str(entry.get("reader_goal", "")),
        generated_by=str(entry.get("generated_by", "guided_artifact_catalog")),
        decision_importance=str(entry.get("decision_importance", "supporting_context")),
        read_order=int(entry.get("read_order", 999)),
    )


def _dedupe_entries(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        relative_path = str(entry.get("relative_path", "")).strip().replace("\\", "/")
        if not relative_path:
            continue
        normalized = dict(entry)
        normalized["relative_path"] = relative_path
        merged[relative_path] = {**merged.get(relative_path, {}), **normalized}
    return sorted(
        merged.values(),
        key=lambda item: (
            _PRIORITY_ORDER.get(str(item.get("priority", "low")), 99),
            int(item.get("read_order", 999)),
            str(item.get("relative_path", "")),
        ),
    )


def _load_training_overview(artifact_root: Path) -> Dict[str, Any]:
    summary = read_json(artifact_root / "training" / "summary.json", default={}, expect_type=dict)
    readiness = read_json(artifact_root / "production_readiness.json", default={}, expect_type=dict)
    benchmark = read_json(artifact_root / "ood_benchmark" / "summary.json", default={}, expect_type=dict)
    payload = {
        "run_id": summary.get("run_id", ""),
        "run_label": summary.get("run_label", summary.get("run_id", "")),
        "crop_name": summary.get("crop_name", ""),
        "part_name": summary.get("part_name", "unspecified"),
        "classification_split": dict(readiness).get("classification_split", ""),
        "readiness_status": dict(readiness).get("status", ""),
        "readiness_passed": dict(readiness).get("passed"),
        "ood_evidence_source": dict(readiness).get("ood_evidence_source", summary.get("ood_evidence_source", "")),
        "class_count": summary.get("class_count"),
        "class_names": list(summary.get("class_names", [])) if isinstance(summary.get("class_names"), list) else [],
        "adapter_dir": summary.get("adapter_dir", ""),
        "artifact_dir": summary.get("artifact_dir", ""),
        "checkpoint_count": summary.get("checkpoint_count"),
        "final_metrics": dict(summary.get("final_metrics", {})) if isinstance(summary.get("final_metrics"), dict) else {},
        "ood_benchmark_status": dict(benchmark).get("status", ""),
    }
    return {key: value for key, value in payload.items() if value not in (None, "", [], {})}


def _load_prep_overview(artifact_root: Path) -> Dict[str, Any]:
    summary = read_json(artifact_root / "prep_summary.json", default={}, expect_type=dict)
    nested_summary = dict(summary.get("summary", {})) if isinstance(summary.get("summary"), dict) else {}
    payload = {
        "crop_name": summary.get("crop_name", ""),
        "source_root": summary.get("source_root", ""),
        "runtime_ready": summary.get("runtime_ready"),
        "prepared_runtime_root": summary.get("prepared_runtime_root", ""),
        "blocking_issue_count": nested_summary.get("blocking_issues"),
        "readable_images": nested_summary.get("readable_images"),
        "excluded_images": nested_summary.get("excluded_images"),
        "same_class_review_pairs": nested_summary.get("same_class_review_pairs"),
        "cross_class_conflicts": nested_summary.get("cross_class_conflicts"),
    }
    return {key: value for key, value in payload.items() if value not in (None, "", [], {})}


def _find_training_entries(artifact_root: Path, *, base_dir: Path, generated_by: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []

    def add(
        relative_path: str,
        *,
        category: str,
        priority: str,
        title_tr: str,
        description_tr: str,
        reader_goal: str,
        decision_importance: str,
        read_order: int,
    ) -> None:
        path = artifact_root / relative_path
        if not path.exists():
            return
        entries.append(
            _entry_from_path(
                path,
                base_dir=base_dir,
                category=category,
                priority=priority,
                title_tr=title_tr,
                description_tr=description_tr,
                reader_goal=reader_goal,
                generated_by=generated_by,
                decision_importance=decision_importance,
                read_order=read_order,
            )
        )

    add(
        "production_readiness.json",
        category="ood_and_readiness",
        priority="critical",
        title_tr="Nihai Ã¼retim hazÄ±rlÄ±ÄŸÄ± kararÄ±",
        description_tr="Deploy kararÄ± iÃ§in Ã¶nce bakÄ±lmasÄ± gereken ana dosya.",
        reader_goal="Modelin Ã¼retime Ã§Ä±kmaya hazÄ±r olup olmadÄ±ÄŸÄ±nÄ± anlamak",
        decision_importance="deploy_decision",
        read_order=1,
    )
    add(
        "training/summary.json",
        category="training",
        priority="critical",
        title_tr="EÄŸitim Ã¶zeti",
        description_tr="KoÅŸunun kimliÄŸi, sÄ±nÄ±flarÄ±, temel metrikleri ve export yollarÄ±nÄ± Ã¶zetler.",
        reader_goal="KoÅŸunun ne Ã¼rettiÄŸini tek dosyada gÃ¶rmek",
        decision_importance="run_overview",
        read_order=2,
    )
    add(
        "test/metric_gate.json",
        category="test",
        priority="critical",
        title_tr="Test gate Ã¶zeti",
        description_tr="Held-out test performansÄ±nÄ±n gate kararÄ±nÄ± iÃ§erir.",
        reader_goal="AsÄ±l test kararÄ±nÄ± gÃ¶rmek",
        decision_importance="deploy_decision",
        read_order=3,
    )
    add(
        "validation/metric_gate.json",
        category="validation",
        priority="high",
        title_tr="Validation gate Ã¶zeti",
        description_tr="Validation split Ã¼zerinde Ã¶lÃ§Ã¼len yardÄ±mcÄ± gate kararÄ±.",
        reader_goal="Test Ã¶ncesi referans performansÄ± gÃ¶rmek",
        decision_importance="supporting_diagnostic",
        read_order=4,
    )
    add(
        "ood_benchmark/summary.json",
        category="ood_and_readiness",
        priority="high",
        title_tr="OOD benchmark Ã¶zeti",
        description_tr="GerÃ§ek OOD yoksa kullanÄ±lan held-out benchmark sonucunu Ã¶zetler.",
        reader_goal="OOD fallback kanÄ±tÄ±nÄ± incelemek",
        decision_importance="ood_decision",
        read_order=5,
    )
    add(
        "provenance_slice_breakdown.json",
        category="ood_and_readiness",
        priority="medium",
        title_tr="Provenance dilim kirilimi",
        description_tr="Authoritative ID split uzerinde provenance slice metriklerini raporlar.",
        reader_goal="Kaynak veya domain slice kaymalarini incelemek",
        decision_importance="supporting_diagnostic",
        read_order=6,
    )
    add(
        "training/history.json",
        category="training",
        priority="high",
        title_tr="Epoch bazlÄ± eÄŸitim geÃ§miÅŸi",
        description_tr="Epoch seviyesinde loss ve performans eÄŸrilerini JSON olarak saklar.",
        reader_goal="EÄŸitim trendini programatik olarak incelemek",
        decision_importance="training_diagnosis",
        read_order=10,
    )
    add(
        "training/history.csv",
        category="training",
        priority="medium",
        title_tr="EÄŸitim geÃ§miÅŸi CSV",
        description_tr="Epoch metriklerinin tablo formatÄ±ndaki karÅŸÄ±lÄ±ÄŸÄ±.",
        reader_goal="Tablo halinde eÄŸitim eÄŸrilerini incelemek",
        decision_importance="training_diagnosis",
        read_order=11,
    )
    add(
        "training/results.csv",
        category="training",
        priority="medium",
        title_tr="EÄŸitim sonuÃ§ tablosu",
        description_tr="Ã–zet epoch sonuÃ§larÄ±nÄ± CSV halinde sunar.",
        reader_goal="Epoch sonuÃ§larÄ±nÄ± hÄ±zlÄ±ca taramak",
        decision_importance="training_diagnosis",
        read_order=12,
    )
    add(
        "training/batch_metrics.csv",
        category="training",
        priority="medium",
        title_tr="Batch metrikleri",
        description_tr="AdÄ±m bazlÄ± loss, LR ve throughput kayÄ±tlarÄ±nÄ± iÃ§erir.",
        reader_goal="Ä°nce taneli eÄŸitim davranÄ±ÅŸÄ±nÄ± incelemek",
        decision_importance="training_diagnosis",
        read_order=13,
    )
    add(
        "training/results.png",
        category="training",
        priority="medium",
        title_tr="EÄŸitim grafik paneli",
        description_tr="Loss, accuracy ve throughput eÄŸrilerini tek gÃ¶rselde toplar.",
        reader_goal="EÄŸitimin genel seyrini gÃ¶rsel olarak gÃ¶rmek",
        decision_importance="training_diagnosis",
        read_order=14,
    )
    for curve_path in sorted((artifact_root / "training").glob("training_curves*.png")):
        entries.append(
            _entry_from_path(
                curve_path,
                base_dir=base_dir,
                category="training",
                priority="low",
                title_tr=f"EÄŸitim eÄŸrisi gÃ¶rseli: {curve_path.name}",
                description_tr="Notebook tarafÄ±ndan ara/final aÅŸamada kaydedilen eÄŸitim eÄŸrisi gÃ¶rseli.",
                reader_goal="Belirli epoch anlarÄ±ndaki eÄŸitim eÄŸrilerini gÃ¶rmek",
                generated_by=generated_by,
                decision_importance="training_diagnosis",
                read_order=20,
            )
        )

    for split_name, order_base in (("validation", 30), ("test", 40)):
        add(
            f"{split_name}/classification_report.txt",
            category=split_name,
            priority="medium",
            title_tr=f"{split_name.title()} sÄ±nÄ±flandÄ±rma raporu",
            description_tr="Precision, recall ve F1 Ã¶zetini metin formatÄ±nda sunar.",
            reader_goal="SÄ±nÄ±f bazlÄ± metrikleri metin olarak okumak",
            decision_importance="quality_diagnosis",
            read_order=order_base,
        )
        add(
            f"{split_name}/classification_report.json",
            category=split_name,
            priority="medium",
            title_tr=f"{split_name.title()} sÄ±nÄ±flandÄ±rma raporu JSON",
            description_tr="AynÄ± raporun makine-okur JSON sÃ¼rÃ¼mÃ¼.",
            reader_goal="Raporu programatik olarak tÃ¼ketmek",
            decision_importance="quality_diagnosis",
            read_order=order_base + 1,
        )
        add(
            f"{split_name}/per_class_metrics.csv",
            category=split_name,
            priority="medium",
            title_tr=f"{split_name.title()} sÄ±nÄ±f bazlÄ± metrikler",
            description_tr="Her sÄ±nÄ±f iÃ§in precision, recall, F1 ve support deÄŸerlerini iÃ§erir.",
            reader_goal="Hangi sÄ±nÄ±fta sorun olduÄŸunu gÃ¶rmek",
            decision_importance="quality_diagnosis",
            read_order=order_base + 2,
        )
        add(
            f"{split_name}/confusion_matrix.csv",
            category=split_name,
            priority="medium",
            title_tr=f"{split_name.title()} confusion matrix CSV",
            description_tr="Ham confusion matrix deÄŸerlerini tablo halinde tutar.",
            reader_goal="KarÄ±ÅŸan sÄ±nÄ±f Ã§iftlerini sayÄ±sal gÃ¶rmek",
            decision_importance="quality_diagnosis",
            read_order=order_base + 3,
        )
        add(
            f"{split_name}/confusion_matrix.png",
            category=split_name,
            priority="medium",
            title_tr=f"{split_name.title()} confusion matrix gÃ¶rseli",
            description_tr="Ham confusion matrix'in gÃ¶rsel hali.",
            reader_goal="Hangi sÄ±nÄ±flarÄ±n karÄ±ÅŸtÄ±ÄŸÄ±nÄ± hÄ±zlÄ± gÃ¶rmek",
            decision_importance="quality_diagnosis",
            read_order=order_base + 4,
        )
        add(
            f"{split_name}/confusion_matrix_normalized.png",
            category=split_name,
            priority="medium",
            title_tr=f"{split_name.title()} normalize confusion matrix",
            description_tr="SÄ±nÄ±f bÃ¼yÃ¼klÃ¼ÄŸÃ¼nden baÄŸÄ±msÄ±z normalize confusion matrix gÃ¶rseli.",
            reader_goal="Oransal hata desenini gÃ¶rmek",
            decision_importance="quality_diagnosis",
            read_order=order_base + 5,
        )
        add(
            f"{split_name}/ood_type_breakdown.json",
            category=split_name,
            priority="medium",
            title_tr=f"{split_name.title()} OOD tip kÄ±rÄ±lÄ±mÄ±",
            description_tr="OOD klasÃ¶r alt tiplerine gÃ¶re metrik kÄ±rÄ±lÄ±mÄ±nÄ± iÃ§erir.",
            reader_goal="Hangi OOD alt tipinde sorun olduÄŸunu gÃ¶rmek",
            decision_importance="ood_decision",
            read_order=order_base + 6,
        )
        add(
            f"{split_name}/ood_method_comparison.json",
            category=split_name,
            priority="medium",
            title_tr=f"{split_name.title()} OOD method comparison",
            description_tr="Pooled ve slice-aware OOD score yontem karsilastirmasini icerir.",
            reader_goal="Ensemble, energy ve knn kanitlarini karsilastirmak",
            decision_importance="ood_decision",
            read_order=order_base + 7,
        )
        add(
            f"{split_name}/ood_evidence_summary.json",
            category=split_name,
            priority="high",
            title_tr=f"{split_name.title()} OOD kanÄ±t Ã¶zeti",
            description_tr="Bu split iÃ§in OOD Ã¶rnek sayÄ±larÄ± ve Ã¶zet metrikleri gÃ¶sterir.",
            reader_goal="OOD kanÄ±tÄ±nÄ±n yeterliliÄŸini gÃ¶rmek",
            decision_importance="ood_decision",
            read_order=order_base + 7,
        )
        add(
            f"{split_name}/predictions.csv",
            category=split_name,
            priority="medium",
            title_tr=f"{split_name.title()} sample predictions",
            description_tr="Her ornek icin tahmin, etiket ve guven bilgisini CSV olarak sunar.",
            reader_goal="Yanlis tahmin edilen ornekleri tek tek incelemek",
            decision_importance="quality_diagnosis",
            read_order=order_base + 8,
        )
        add(
            f"{split_name}/hard_examples.csv",
            category=split_name,
            priority="high",
            title_tr=f"{split_name.title()} hard examples",
            description_tr="Yanlis siniflanan veya kacirilan OOD orneklerini oncelikli inceleme icin listeler.",
            reader_goal="Feedback ile duzeltilecek zor ornekleri onceliklendirmek",
            decision_importance="quality_diagnosis",
            read_order=order_base + 9,
        )
        add(
            f"{split_name}/hard_examples_thumbnails",
            category=split_name,
            priority="medium",
            title_tr=f"{split_name.title()} hard example thumbnails",
            description_tr="Zor orneklerin hizli gozden gecirme icin kucuk onizlemelerini tutar.",
            reader_goal="Zor ornekleri gorsel olarak hizlica taramak",
            decision_importance="quality_diagnosis",
            read_order=order_base + 10,
        )

    add(
        "ood_benchmark/per_fold.csv",
        category="ood_and_readiness",
        priority="medium",
        title_tr="OOD benchmark fold tablosu",
        description_tr="Held-out benchmark fold sonuÃ§larÄ±nÄ± tek tabloda toplar.",
        reader_goal="Fold bazÄ±nda OOD performansÄ±nÄ± incelemek",
        decision_importance="ood_decision",
        read_order=50,
    )
    add(
        "ood_benchmark/progress.json",
        category="ood_and_readiness",
        priority="low",
        title_tr="OOD benchmark ilerleme kaydÄ±",
        description_tr="Uzun benchmark Ã§alÄ±ÅŸÄ±rken ara durum bilgisini tutar.",
        reader_goal="Benchmark sÃ¼recinin hangi aÅŸamada kaldÄ±ÄŸÄ±nÄ± gÃ¶rmek",
        decision_importance="runtime_diagnostic",
        read_order=51,
    )
    for fold_dir in sorted((artifact_root / "ood_benchmark" / "folds").glob("*")):
        if not fold_dir.is_dir():
            continue
        fold_name = fold_dir.name
        for filename, title, order in (
            ("metric_gate.json", "OOD fold gate Ã¶zeti", 60),
            ("failure.json", "OOD fold hata Ã¶zeti", 61),
            ("failure_traceback.txt", "OOD fold traceback", 62),
        ):
            target = fold_dir / filename
            if not target.exists():
                continue
            entries.append(
                _entry_from_path(
                    target,
                    base_dir=base_dir,
                    category="ood_and_readiness",
                    priority="low" if "failure" not in filename else "high",
                    title_tr=f"{title}: {fold_name}",
                    description_tr=f"Held-out fold '{fold_name}' iÃ§in Ã¼retilen yardÄ±mcÄ± artefact.",
                    reader_goal="Tek bir fold Ã¶zelinde ayrÄ±ntÄ± incelemek",
                    generated_by=generated_by,
                    decision_importance="ood_decision" if "metric_gate" in filename else "runtime_diagnostic",
                    read_order=order,
                )
            )
    return entries


def _find_prep_entries(artifact_root: Path, *, base_dir: Path, generated_by: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []

    def add(
        relative_path: str,
        *,
        category: str,
        priority: str,
        title_tr: str,
        description_tr: str,
        reader_goal: str,
        decision_importance: str,
        read_order: int,
    ) -> None:
        path = artifact_root / relative_path
        if not path.exists():
            return
        entries.append(
            _entry_from_path(
                path,
                base_dir=base_dir,
                category=category,
                priority=priority,
                title_tr=title_tr,
                description_tr=description_tr,
                reader_goal=reader_goal,
                generated_by=generated_by,
                decision_importance=decision_importance,
                read_order=read_order,
            )
        )

    add(
        "prep_summary.json",
        category="summary",
        priority="critical",
        title_tr="Prep ana Ã¶zeti",
        description_tr="Notebook 0 koÅŸusunun en Ã¶nemli audit Ã¶zetini iÃ§erir.",
        reader_goal="Dataset prep sonucunu tek dosyada gÃ¶rmek",
        decision_importance="prep_gate",
        read_order=1,
    )
    add(
        "proposed_split_manifest.json",
        category="split_plan",
        priority="critical",
        title_tr="Ã–nerilen split manifesti",
        description_tr="Her gÃ¶rÃ¼ntÃ¼nÃ¼n hangi split'e gideceÄŸini ve blokajlarÄ± iÃ§erir.",
        reader_goal="GerÃ§ek split planÄ±nÄ± incelemek",
        decision_importance="prep_gate",
        read_order=2,
    )
    add(
        "ood_handoff_checklist.json",
        category="split_plan",
        priority="high",
        title_tr="OOD handoff kontrol listesi",
        description_tr="Prep sonrasÄ± OOD veri hazÄ±rlÄ±ÄŸÄ± iÃ§in kalan adÄ±mlarÄ± gÃ¶sterir.",
        reader_goal="OOD tarafÄ±nda ne eksik olduÄŸunu gÃ¶rmek",
        decision_importance="prep_gate",
        read_order=3,
    )
    add(
        "label_normalization_report.json",
        category="label_normalization",
        priority="high",
        title_tr="Etiket normalizasyon raporu",
        description_tr="Ham sÄ±nÄ±f klasÃ¶r adlarÄ±nÄ±n normalize sÄ±nÄ±f adlarÄ±na nasÄ±l eÅŸlendiÄŸini gÃ¶sterir.",
        reader_goal="SÄ±nÄ±f isimlerinin doÄŸru yorumlandÄ±ÄŸÄ±nÄ± doÄŸrulamak",
        decision_importance="data_quality",
        read_order=10,
    )
    add(
        "class_health_report.json",
        category="class_health",
        priority="high",
        title_tr="SÄ±nÄ±f saÄŸlÄ±k raporu",
        description_tr="Her sÄ±nÄ±fÄ±n aile sayÄ±sÄ±, hedef split daÄŸÄ±lÄ±mÄ± ve risk Ã¶zetini iÃ§erir.",
        reader_goal="Hangi sÄ±nÄ±flarÄ±n kÄ±rÄ±lgan olduÄŸunu gÃ¶rmek",
        decision_importance="data_quality",
        read_order=11,
    )
    add(
        "dataset_manifest.csv",
        category="manifests",
        priority="medium",
        title_tr="TÃ¼m gÃ¶rÃ¼ntÃ¼ manifesti",
        description_tr="Taranan tÃ¼m gÃ¶rÃ¼ntÃ¼lerin kalite ve hash alanlarÄ±nÄ± iÃ§erir.",
        reader_goal="Ham veri kayÄ±tlarÄ±nÄ± tablo halinde incelemek",
        decision_importance="data_audit",
        read_order=20,
    )
    add(
        "family_manifest.csv",
        category="manifests",
        priority="medium",
        title_tr="Aile manifesti",
        description_tr="GÃ¶rÃ¼ntÃ¼ aileleri ve split atamalarÄ±nÄ± satÄ±r bazÄ±nda gÃ¶sterir.",
        reader_goal="Aile bazlÄ± grouping sonucunu gÃ¶rmek",
        decision_importance="data_audit",
        read_order=21,
    )
    add(
        "same_class_review_candidates.csv",
        category="review_items",
        priority="medium",
        title_tr="AynÄ± sÄ±nÄ±f inceleme adaylarÄ±",
        description_tr="Borderline benzerlik gÃ¶steren aynÄ± sÄ±nÄ±f gÃ¶rÃ¼ntÃ¼ Ã§iftleri.",
        reader_goal="Manuel inceleme gerektiren Ã§iftleri gÃ¶rmek",
        decision_importance="data_quality",
        read_order=30,
    )
    add(
        "same_class_review_clusters.csv",
        category="review_items",
        priority="medium",
        title_tr="AynÄ± sÄ±nÄ±f inceleme kÃ¼meleri",
        description_tr="Manuel veya otomatik Ã§Ã¶zÃ¼m iÃ§in gruplanmÄ±ÅŸ review kÃ¼meleri.",
        reader_goal="Review iÅŸini kÃ¼me bazÄ±nda organize etmek",
        decision_importance="data_quality",
        read_order=31,
    )
    add(
        "same_class_auto_resolved_clusters.csv",
        category="review_items",
        priority="low",
        title_tr="Otomatik Ã§Ã¶zÃ¼len review kÃ¼meleri",
        description_tr="DÃ¼ÅŸÃ¼k riskli olduÄŸu iÃ§in otomatik Ã§Ã¶zÃ¼len kÃ¼meler.",
        reader_goal="Hangi review kÃ¼melerinin otomatik kapandÄ±ÄŸÄ±nÄ± gÃ¶rmek",
        decision_importance="data_quality",
        read_order=32,
    )
    add(
        "same_class_high_risk_clusters.csv",
        category="review_items",
        priority="high",
        title_tr="YÃ¼ksek riskli review kÃ¼meleri",
        description_tr="Manuel kontrol gerektiren yÃ¼ksek risk kÃ¼meleri.",
        reader_goal="Ã–ncelikli insan inceleme listesini gÃ¶rmek",
        decision_importance="prep_gate",
        read_order=33,
    )
    add(
        "cross_class_conflicts.csv",
        category="review_items",
        priority="critical",
        title_tr="SÄ±nÄ±flar arasÄ± Ã§akÄ±ÅŸmalar",
        description_tr="FarklÄ± sÄ±nÄ±flara dÃ¼ÅŸen ama Ã§akÄ±ÅŸan veya kopya gÃ¶rÃ¼nen Ã¶rnekler.",
        reader_goal="Split ve etiket hatalarÄ±nÄ± gÃ¶rmek",
        decision_importance="prep_gate",
        read_order=34,
    )
    add(
        "exact_duplicates.csv",
        category="review_items",
        priority="medium",
        title_tr="Exact duplicate listesi",
        description_tr="AynÄ± hash'e sahip gÃ¶rÃ¼ntÃ¼ gruplarÄ±nÄ± listeler.",
        reader_goal="Birebir kopyalarÄ± temizlemek",
        decision_importance="data_quality",
        read_order=35,
    )
    return entries


def _render_section_markdown(
    *,
    heading: str,
    intro: str,
    entries: Sequence[Dict[str, Any]],
) -> str:
    lines = [f"# {heading}", "", intro.strip(), ""]
    if not entries:
        lines.extend(["Bu bÃ¶lÃ¼m iÃ§in mevcut artefact bulunamadÄ±.", ""])
        return "\n".join(lines).rstrip() + "\n"
    for entry in entries:
        lines.append(f"## {entry['title_tr']}")
        lines.append(f"- Yol: `{entry['relative_path']}`")
        lines.append(f"- Ã–ncelik: `{entry['priority']}`")
        lines.append(f"- Format: `{entry['format']}`")
        lines.append(f"- AmaÃ§: {entry['reader_goal']}")
        lines.append(f"- AÃ§Ä±klama: {entry['description_tr']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_guided_file(path: Path, content: str | Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, dict):
        return write_json(path, content, ensure_ascii=False, sort_keys=False)
    path.write_text(str(content), encoding="utf-8")
    return path


def _copy_guided_to_telemetry(telemetry: Any, *, artifact_root: Path, written_paths: Sequence[Path]) -> None:
    if telemetry is None or not hasattr(telemetry, "copy_artifact_file"):
        return
    for path in written_paths:
        relative_path = path.relative_to(artifact_root).as_posix()
        telemetry.copy_artifact_file(path, relative_path)


def _prepare_guided_catalog(
    *,
    root: Path,
    base_dir: Path,
    overview_loader: Any,
    entry_finder: Any,
    overview_updates: Dict[str, Any] | None,
    extra_entries: Sequence[Dict[str, Any]] | None,
    generated_by: str,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    overview = {**overview_loader(root), **dict(overview_updates or {})}
    catalog_entries = entry_finder(root, base_dir=base_dir, generated_by=generated_by)
    for item in list(extra_entries or []):
        catalog_entries.append(_coerce_extra_entry(dict(item), base_dir=base_dir))
    return overview, _dedupe_entries(catalog_entries)


def _build_guided_payloads(
    *,
    base_dir: Path,
    catalog_kind: str,
    overview_kind: str,
    overview_schema_version: str,
    overview: Dict[str, Any],
    overview_filename: str,
    primary_files: Dict[str, str],
    catalog_entries: Sequence[Dict[str, Any]],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    catalog_payload = {
        "schema_version": "v1_guided_file_catalog",
        "catalog_kind": catalog_kind,
        "catalog_base": str(base_dir.resolve()),
        "entry_count": len(catalog_entries),
        "entries": list(catalog_entries),
    }
    overview_payload = {
        "schema_version": overview_schema_version,
        "overview_kind": overview_kind,
        "catalog_base": str(base_dir.resolve()),
        "primary_files": {
            "start_here": _GUIDED_START_HERE,
            overview_filename.split("/")[-1].replace(".json", "").replace("01_", ""): overview_filename,
            "file_catalog": _GUIDED_CATALOG,
            **primary_files,
        },
        **overview,
    }
    return catalog_payload, overview_payload


def _write_guided_bundle(
    *,
    guided_dir: Path,
    overview_filename: str,
    overview_payload: Dict[str, Any],
    catalog_payload: Dict[str, Any],
    start_here: str,
    category_to_doc: Dict[str, tuple[str, str, str]],
    catalog_entries: Sequence[Dict[str, Any]],
) -> List[Path]:
    written_paths = [
        _write_guided_file(guided_dir / "00_start_here.md", start_here.rstrip() + "\n"),
        _write_guided_file(guided_dir / Path(overview_filename).name, overview_payload),
        _write_guided_file(guided_dir / "02_file_catalog.json", catalog_payload),
    ]
    for category, (filename, heading, intro) in category_to_doc.items():
        written_paths.append(
            _write_guided_file(
                guided_dir / filename,
                _render_section_markdown(
                    heading=heading,
                    intro=intro,
                    entries=[entry for entry in catalog_entries if entry.get("category") == category],
                ),
            )
        )
    return written_paths


def _merge_guided_telemetry(
    *,
    telemetry: Any,
    artifact_root: Path,
    written_paths: Sequence[Path],
    catalog_entries: Sequence[Dict[str, Any]],
    overview_payload: Dict[str, Any],
    overview_key: str,
    overview_filename: str,
) -> None:
    _copy_guided_to_telemetry(telemetry, artifact_root=artifact_root, written_paths=written_paths)
    if telemetry is not None and hasattr(telemetry, "merge_artifact_catalog"):
        telemetry.merge_artifact_catalog(catalog_entries)
    if telemetry is not None and hasattr(telemetry, "merge_summary_metadata"):
        telemetry.merge_summary_metadata(
            {
                "guided_artifacts": {
                    "start_here": _GUIDED_START_HERE,
                    overview_key: overview_filename,
                    "file_catalog": _GUIDED_CATALOG,
                },
                "guided_catalog_entry_count": len(catalog_entries),
                overview_key: overview_payload,
            }
        )


def refresh_training_guided_artifacts(
    artifact_root: str | Path,
    *,
    telemetry: Any = None,
    overview_updates: Dict[str, Any] | None = None,
    extra_entries: Sequence[Dict[str, Any]] | None = None,
    generated_by: str = "src.training.services.reporting",
) -> Dict[str, Any]:
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    guided_dir = root / "guided"
    base_dir = root

    overview, catalog_entries = _prepare_guided_catalog(
        root=root,
        base_dir=base_dir,
        overview_loader=_load_training_overview,
        entry_finder=_find_training_entries,
        overview_updates=overview_updates,
        extra_entries=extra_entries,
        generated_by=generated_by,
    )
    catalog_payload, overview_payload = _build_guided_payloads(
        base_dir=base_dir,
        catalog_kind="training",
        overview_kind="training",
        overview_schema_version="v1_guided_run_overview",
        overview=overview,
        overview_filename="guided/01_run_overview.json",
        primary_files={
            "training_summary": "training/summary.json",
            "production_readiness": "production_readiness.json",
            "test_metric_gate": "test/metric_gate.json",
        },
        catalog_entries=catalog_entries,
    )

    start_here = "\n".join(
        [
            "# Buradan BaÅŸla",
            "",
            "Ham artefact dosyalarÄ± korunur. Ã–nce ÅŸu sÄ±rayÄ± izleyin:",
            "",
            "1. `guided/01_run_overview.json`",
            "2. `production_readiness.json`",
            "3. `training/summary.json`",
            "4. `test/metric_gate.json`",
            "5. `ood_benchmark/summary.json` veya `test/ood_evidence_summary.json`",
            "",
            "TanÄ± iÃ§in:",
            "- `guided/10_training.md`",
            "- `guided/20_validation.md`",
            "- `guided/30_test.md`",
            "- `guided/40_ood_and_readiness.md`",
            "- `guided/50_adapter_export.md`",
            "- `guided/60_logs_and_checkpoints.md`",
            "",
            f"Katalog tabanÄ±: `{base_dir.resolve().as_posix()}`",
            "",
        ]
    )

    category_to_doc = {
        "training": ("10_training.md", "EÄŸitim DosyalarÄ±", "Bu bÃ¶lÃ¼m eÄŸitim geÃ§miÅŸi ve eÄŸri dosyalarÄ±nÄ± Ã¶zetler."),
        "validation": ("20_validation.md", "Validation DosyalarÄ±", "Bu bÃ¶lÃ¼m validation split artefactlerini toplar."),
        "test": ("30_test.md", "Test DosyalarÄ±", "Bu bÃ¶lÃ¼m held-out test artefactlerini toplar."),
        "ood_and_readiness": (
            "40_ood_and_readiness.md",
            "OOD ve Readiness",
            "Bu bÃ¶lÃ¼m OOD kanÄ±tÄ± ve nihai readiness kararÄ±nÄ± toplar.",
        ),
        "adapter_export": (
            "50_adapter_export.md",
            "Adapter Export",
            "Bu bÃ¶lÃ¼m adapter export ve notebook export dosyalarÄ±nÄ± listeler.",
        ),
        "logs_and_checkpoints": (
            "60_logs_and_checkpoints.md",
            "Log ve Checkpoint",
            "Bu bÃ¶lÃ¼m runtime loglarÄ± ve checkpoint kayÄ±tlarÄ±nÄ± listeler.",
        ),
    }

    written_paths = _write_guided_bundle(
        guided_dir=guided_dir,
        overview_filename="guided/01_run_overview.json",
        overview_payload=overview_payload,
        catalog_payload=catalog_payload,
        start_here=start_here,
        category_to_doc=category_to_doc,
        catalog_entries=catalog_entries,
    )
    _merge_guided_telemetry(
        telemetry=telemetry,
        artifact_root=root,
        written_paths=written_paths,
        catalog_entries=catalog_entries,
        overview_payload=overview_payload,
        overview_key="run_overview",
        overview_filename="guided/01_run_overview.json",
    )
    return {
        "catalog": catalog_payload,
        "overview": overview_payload,
        "paths": {path.name: str(path) for path in written_paths},
    }


def refresh_prep_guided_artifacts(
    artifact_root: str | Path,
    *,
    telemetry: Any = None,
    overview_updates: Dict[str, Any] | None = None,
    extra_entries: Sequence[Dict[str, Any]] | None = None,
    generated_by: str = "scripts.prepare_grouped_runtime_dataset",
) -> Dict[str, Any]:
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    guided_dir = root / "guided"
    base_dir = root

    overview, catalog_entries = _prepare_guided_catalog(
        root=root,
        base_dir=base_dir,
        overview_loader=_load_prep_overview,
        entry_finder=_find_prep_entries,
        overview_updates=overview_updates,
        extra_entries=extra_entries,
        generated_by=generated_by,
    )
    catalog_payload, overview_payload = _build_guided_payloads(
        base_dir=base_dir,
        catalog_kind="data_prep",
        overview_kind="data_prep",
        overview_schema_version="v1_guided_prep_overview",
        overview=overview,
        overview_filename="guided/01_prep_overview.json",
        primary_files={
            "prep_summary": "prep_summary.json",
            "split_manifest": "proposed_split_manifest.json",
        },
        catalog_entries=catalog_entries,
    )
    start_here = "\n".join(
        [
            "# Buradan BaÅŸla",
            "",
            "Notebook 0 Ã§Ä±ktÄ±larÄ±nda hiÃ§bir ham dosya silinmez. Ã–nce ÅŸu sÄ±rayla ilerleyin:",
            "",
            "1. `guided/01_prep_overview.json`",
            "2. `prep_summary.json`",
            "3. `proposed_split_manifest.json`",
            "4. `cross_class_conflicts.csv` ve `same_class_high_risk_clusters.csv`",
            "",
            f"Katalog tabanÄ±: `{base_dir.resolve().as_posix()}`",
            "",
        ]
    )
    category_to_doc = {
        "summary": ("10_summary.md", "Prep Ã–zeti", "Ana prep Ã¶zeti ve koÅŸu gÃ¶rÃ¼nÃ¼mÃ¼."),
        "label_normalization": (
            "20_label_normalization.md",
            "Etiket Normalizasyonu",
            "Ham sÄ±nÄ±f isimlerinin normalize sonuÃ§larÄ±.",
        ),
        "class_health": ("30_class_health.md", "SÄ±nÄ±f SaÄŸlÄ±ÄŸÄ±", "SÄ±nÄ±f bazlÄ± risk ve daÄŸÄ±lÄ±m raporlarÄ±."),
        "split_plan": ("40_split_plan.md", "Split PlanÄ±", "Ã–nerilen split ve OOD handoff kararlarÄ±."),
        "review_items": ("50_review_items.md", "Ä°nceleme Ã–ÄŸeleri", "Manuel inceleme gerektiren Ã§ift ve kÃ¼meler."),
        "manifests": ("60_manifests.md", "Manifestler", "Tam veri ve aile manifestleri."),
    }

    written_paths = _write_guided_bundle(
        guided_dir=guided_dir,
        overview_filename="guided/01_prep_overview.json",
        overview_payload=overview_payload,
        catalog_payload=catalog_payload,
        start_here=start_here,
        category_to_doc=category_to_doc,
        catalog_entries=catalog_entries,
    )
    _merge_guided_telemetry(
        telemetry=telemetry,
        artifact_root=root,
        written_paths=written_paths,
        catalog_entries=catalog_entries,
        overview_payload=overview_payload,
        overview_key="prep_overview",
        overview_filename="guided/01_prep_overview.json",
    )
    return {
        "catalog": catalog_payload,
        "overview": overview_payload,
        "paths": {path.name: str(path) for path in written_paths},
    }

