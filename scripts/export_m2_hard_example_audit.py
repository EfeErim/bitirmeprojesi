"""Export targeted M2 hard-example review packets."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared.csv_utils import write_csv_rows_with_order  # noqa: E402

DEFAULT_TARGETS = ("tomato__leaf", "tomato__fruit", "apricot__fruit", "strawberry__fruit")
REASON_WEIGHTS = {
    "prototype_evidence_weak": 180,
    "prototype_policy_not_calibrated": 160,
    "part_conflict": 150,
    "negative_prototype_too_close": 140,
    "prototype_overrode_untrusted_router_handoff": 120,
    "prototype_overrode_calibrated_part_conflict": 110,
}
CSV_COLUMNS = (
    "rank",
    "priority_score",
    "priority_reasons",
    "image_id",
    "resolved_image",
    "source",
    "expected_target",
    "expected_crop",
    "expected_part",
    "expected_class",
    "actual_status",
    "predicted_crop",
    "predicted_part",
    "predicted_disease",
    "prototype_target",
    "prototype_class_label",
    "prototype_similarity",
    "prototype_margin",
    "reconcile_reason",
    "failure_bucket",
    "pass_fail",
    "review_decision",
    "corrected_crop",
    "corrected_part",
    "corrected_class",
    "prototype_quality",
    "adapter_training_quality",
    "review_notes",
)


def newest_m2_run_id(results_root: Path = Path("docs/demo_results/m2")) -> str:
    candidates = [path.name for path in results_root.iterdir() if path.is_dir() and (path / "m2_demo_checklist_run.json").is_file()]
    if not candidates:
        raise FileNotFoundError(f"No M2 result folders with m2_demo_checklist_run.json under {results_root}")
    return sorted(candidates)[-1]


def load_m2_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Expected a top-level rows list in {path}")
    return [dict(row) for row in rows if isinstance(row, dict)]


def build_hard_example_rows(
    rows: list[dict[str, Any]],
    *,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    include_unsupported: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    candidates = []
    target_set = set(targets)
    for row in rows:
        if not include_unsupported and not _is_supported_row(row):
            continue
        if target_set and str(row.get("expected_target") or "") not in target_set:
            continue
        score, reasons = _score_row(row, target_set=target_set)
        if score <= 0:
            continue
        candidates.append((_sort_key(row, score), row, score, reasons))

    candidates.sort(key=lambda item: item[0])
    if limit is not None:
        candidates = candidates[:limit]

    audit_rows = []
    for rank, (_, row, score, reasons) in enumerate(candidates, start=1):
        audit_rows.append(
            {
                "rank": rank,
                "priority_score": score,
                "priority_reasons": "|".join(reasons),
                "image_id": _text(row.get("image_id")),
                "resolved_image": _text(row.get("resolved_image")),
                "source": _text(row.get("source")),
                "expected_target": _text(row.get("expected_target")),
                "expected_crop": _text(row.get("expected_crop")),
                "expected_part": _text(row.get("expected_part")),
                "expected_class": _text(row.get("expected_class")),
                "actual_status": _text(row.get("actual_status")),
                "predicted_crop": _text(row.get("predicted_crop")),
                "predicted_part": _text(row.get("predicted_part")),
                "predicted_disease": _text(row.get("predicted_disease")),
                "prototype_target": _text(row.get("prototype_target")),
                "prototype_class_label": _text(row.get("prototype_class_label")),
                "prototype_similarity": _number_text(row.get("prototype_similarity")),
                "prototype_margin": _number_text(row.get("prototype_margin")),
                "reconcile_reason": _text(row.get("reconcile_reason")),
                "failure_bucket": _text(row.get("failure_bucket")),
                "pass_fail": _text(row.get("pass_fail")),
                "review_decision": "",
                "corrected_crop": "",
                "corrected_part": "",
                "corrected_class": "",
                "prototype_quality": "",
                "adapter_training_quality": "",
                "review_notes": "",
            }
        )
    return audit_rows


def write_contact_sheet(rows: list[dict[str, Any]], output: str | Path, *, repo_root: Path, title: str = "") -> int:
    if not output:
        return 0
    from PIL import Image, ImageDraw

    thumbs = []
    for row in rows:
        image_path = resolve_local_image_path(row, repo_root=repo_root)
        if not image_path or not image_path.is_file():
            continue
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue
        image.thumbnail((220, 160))
        thumbs.append((row, image.copy()))

    cols = 4
    cell_width = 340
    cell_height = 255
    title_height = 42 if title else 0
    row_count = max(1, (len(thumbs) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * cell_width, title_height + row_count * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((8, 12), title, fill=(0, 0, 0))
    for index, (row, image) in enumerate(thumbs):
        x = (index % cols) * cell_width
        y = title_height + (index // cols) * cell_height
        sheet.paste(image, (x + (220 - image.width) // 2, y + 5))
        label = (
            f"#{row.get('rank')} {row.get('expected_target')}\n"
            f"GT: {row.get('expected_class')}\n"
            f"Pred: {row.get('predicted_disease') or row.get('actual_status')}\n"
            f"{row.get('reconcile_reason') or row.get('failure_bucket')}"
        )
        draw.multiline_text((x + 8, y + 170), label[:180], fill=(0, 0, 0), spacing=2)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)
    return len(thumbs)


def write_packets(rows: list[dict[str, Any]], output_dir: Path, *, repo_root: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("expected_target") or "unknown")].append(row)

    packets = []
    for index, (target, target_rows) in enumerate(sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])), start=1):
        packet_dir = output_dir / f"{index:02d}_{_slug(target)}"
        packet_dir.mkdir(parents=True, exist_ok=True)
        csv_path = packet_dir / "review_rows.csv"
        sheet_path = packet_dir / "contact_sheet.jpg"
        write_csv_rows_with_order(csv_path, target_rows, preferred_headers=CSV_COLUMNS, encoding="utf-8-sig")
        contact_sheet_count = write_contact_sheet(target_rows, sheet_path, repo_root=repo_root, title=f"{target} ({len(target_rows)} rows)")
        (packet_dir / "README.md").write_text(
            "\n".join(
                [
                    f"# `{target}` M2 Hard Examples",
                    "",
                    f"- rows: `{len(target_rows)}`",
                    f"- contact sheet items: `{contact_sheet_count}`",
                    "- Fill `review_decision` only after visual/domain review.",
                    "- Supported decisions: `keep`, `exclude_ambiguous`, `relabel:<class>`, `add_prototype_positive`, `add_prototype_hard_negative`, `add_adapter_train`.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        packets.append(
            {
                "target": target,
                "rows": len(target_rows),
                "packet_dir": str(packet_dir),
                "csv": str(csv_path),
                "contact_sheet": str(sheet_path),
                "contact_sheet_count": contact_sheet_count,
            }
        )

    summary = {
        "schema_version": "v1_m2_hard_example_packets",
        "packet_dir": str(output_dir),
        "packet_count": len(packets),
        "rows_written": sum(int(packet["rows"]) for packet in packets),
        "packets": packets,
    }
    (output_dir / "packet_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def write_review_index(rows: list[dict[str, Any]], packet_summary: dict[str, Any], output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    packet_root = Path(str(packet_summary.get("packet_dir") or output.parent))
    cards = []
    for packet in packet_summary.get("packets", []):
        contact_sheet = Path(str(packet.get("contact_sheet") or ""))
        csv_path = Path(str(packet.get("csv") or ""))
        image_ref = _relative_ref(contact_sheet, packet_root)
        csv_ref = _relative_ref(csv_path, packet_root)
        cards.append(
            "\n".join(
                [
                    '<section class="packet">',
                    f"<h2>{escape(str(packet.get('target') or 'unknown'))}</h2>",
                    f"<p><strong>Rows:</strong> {int(packet.get('rows') or 0)}</p>",
                    f'<p><a href="{escape(csv_ref)}">review_rows.csv</a></p>',
                    f'<img src="{escape(image_ref)}" alt="{escape(str(packet.get("target") or "packet"))} contact sheet">',
                    "</section>",
                ]
            )
        )
    by_target = Counter(str(row.get("expected_target") or "unknown") for row in rows)
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>M2 Hard-Example Review Packets</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    header {{ margin-bottom: 24px; }}
    .summary {{ display: flex; gap: 18px; flex-wrap: wrap; margin: 16px 0; }}
    .summary div {{ border: 1px solid #d7dde5; padding: 10px 12px; border-radius: 6px; }}
    .packet {{ border-top: 1px solid #d7dde5; padding: 18px 0 26px; }}
    .packet h2 {{ font-size: 18px; margin: 0 0 8px; }}
    .packet img {{ max-width: 100%; height: auto; border: 1px solid #d7dde5; }}
    a {{ color: #0f5ea8; }}
  </style>
</head>
<body>
  <header>
    <h1>M2 Hard-Example Review Packets</h1>
    <p>Static review index for targeted M2 prototype and adapter hard examples.</p>
    <div class="summary">
      <div><strong>Rows</strong><br>{rows}</div>
      <div><strong>Packets</strong><br>{packets}</div>
      <div><strong>Top targets</strong><br>{targets}</div>
    </div>
  </header>
  {cards}
</body>
</html>
""".format(
        rows=len(rows),
        packets=int(packet_summary.get("packet_count") or 0),
        targets=escape(", ".join(f"{target}: {count}" for target, count in by_target.most_common(4))),
        cards="\n".join(cards),
    )
    output.write_text(html, encoding="utf-8")
    return str(output)


def resolve_local_image_path(row: dict[str, Any], *, repo_root: Path) -> Path | None:
    for key in ("resolved_image", "source"):
        value = str(row.get(key) or "").strip()
        if not value:
            continue
        if value.startswith("staged_external:"):
            value = value.split(":", 1)[1]
        marker = "bitirmeprojesi/"
        normalized = value.replace("\\", "/")
        if marker in normalized:
            value = normalized.split(marker, 1)[1]
        path = Path(value)
        if path.is_absolute() and path.exists():
            return path
        candidate = repo_root / value
        if candidate.exists():
            return candidate
    return None


def _score_row(row: dict[str, Any], *, target_set: set[str]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    pass_fail = str(row.get("pass_fail") or "").lower()
    actual_status = str(row.get("actual_status") or "").lower()
    predicted_disease = str(row.get("predicted_disease") or "").strip()
    reconcile_reason = str(row.get("reconcile_reason") or "").strip()
    failure_bucket = str(row.get("failure_bucket") or "").strip()

    if pass_fail == "fail" and actual_status == "success" and predicted_disease:
        score += 1000
        reasons.append("answered_wrong")
    if str(row.get("expected_target") or "") in target_set:
        score += 200
        reasons.append("focus_target")
    if failure_bucket == "router":
        score += 120
        reasons.append("router_failure")
    if reconcile_reason in REASON_WEIGHTS:
        score += REASON_WEIGHTS[reconcile_reason]
        reasons.append(reconcile_reason)
    if pass_fail == "fail":
        score += 100
        reasons.append("failed_row")

    margin = _float_or_none(row.get("prototype_margin"))
    if margin is not None and margin < 0.04:
        score += 40
        reasons.append("low_prototype_margin")
    similarity = _float_or_none(row.get("prototype_similarity"))
    expected_target = str(row.get("expected_target") or "")
    prototype_target = str(row.get("prototype_target") or "")
    if similarity is not None and similarity >= 0.55 and prototype_target and prototype_target != expected_target:
        score += 70
        reasons.append("high_similarity_wrong_prototype_target")
    if pass_fail == "fail" and prototype_target == expected_target and actual_status in {"router_uncertain", "unknown_crop"}:
        score += 80
        reasons.append("prototype_correct_but_abstained")
    return score, list(dict.fromkeys(reasons))


def _sort_key(row: dict[str, Any], score: int) -> tuple[int, str, str]:
    return (-score, str(row.get("expected_target") or ""), str(row.get("image_id") or ""))


def _is_supported_row(row: dict[str, Any]) -> bool:
    target = str(row.get("expected_target") or "")
    crop = str(row.get("expected_crop") or "")
    part = str(row.get("expected_part") or "")
    return bool(target and "__" in target and crop not in {"", "unknown"} and part not in {"", "unknown", "unknown_part"})


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _number_text(value: Any) -> str:
    number = _float_or_none(value)
    return "" if number is None else f"{number:.8g}"


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    return text.strip("_").lower()[:96] or "unknown"


def _relative_ref(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="M2 result run id. Defaults to newest result folder.")
    parser.add_argument("--input", type=Path, help="Path to m2_demo_checklist_run.json.")
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS), help="Comma-separated expected_target values.")
    parser.add_argument("--limit", type=int, default=150, help="Maximum rows to export. Pass 0 for no limit.")
    parser.add_argument("--include-unsupported", action="store_true", help="Include unsupported/unknown expected rows.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--csv-output", type=Path, help="Audit CSV output.")
    parser.add_argument("--packet-output-dir", type=Path, help="Review packet directory.")
    parser.add_argument("--review-index-output", type=Path, help="Static review index output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = args.run_id or (args.input.parent.name if args.input else newest_m2_run_id())
    input_path = args.input or Path("docs/demo_results/m2") / run_id / "m2_demo_checklist_run.json"
    csv_output = args.csv_output or Path("docs/demo_results/m2") / run_id / "hard_example_audit.csv"
    packet_dir = args.packet_output_dir or Path(".runtime_tmp/m2_hard_example_audit") / run_id
    review_index = args.review_index_output or packet_dir / "index.html"
    targets = tuple(target.strip() for target in str(args.targets).split(",") if target.strip())
    limit = None if args.limit == 0 else args.limit

    rows = load_m2_rows(input_path)
    audit_rows = build_hard_example_rows(rows, targets=targets, include_unsupported=args.include_unsupported, limit=limit)
    write_csv_rows_with_order(csv_output, audit_rows, preferred_headers=CSV_COLUMNS, encoding="utf-8-sig")
    packet_summary = write_packets(audit_rows, packet_dir, repo_root=args.repo_root)
    index_output = write_review_index(audit_rows, packet_summary, review_index)

    print(
        json.dumps(
            {
                "schema_version": "v1_m2_hard_example_audit",
                "run_id": run_id,
                "input": str(input_path),
                "row_count": len(audit_rows),
                "csv_output": str(csv_output),
                "packet_output_dir": str(packet_dir),
                "packet_count": packet_summary.get("packet_count", 0),
                "review_index_output": index_output,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
