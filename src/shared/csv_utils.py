from __future__ import annotations

import csv
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def read_csv_rows(path: Path, *, encoding: str = "utf-8-sig") -> List[Dict[str, str]]:
    if not Path(path).is_file() or Path(path).stat().st_size <= 0:
        return []
    with Path(path).open("r", encoding=encoding, newline="") as handle:
        return list(csv.DictReader(handle))


def read_csv_rows_from_source(
    source_path: Path,
    *,
    zip_member_suffix: Optional[str] = None,
    encoding: str = "utf-8-sig",
) -> List[Dict[str, str]]:
    source_path = Path(source_path)
    if source_path.suffix.lower() == ".zip":
        if not zip_member_suffix:
            raise ValueError("zip_member_suffix is required when reading rows from a zip archive")
        with zipfile.ZipFile(source_path) as archive:
            members = [name for name in archive.namelist() if name.endswith(f"/{zip_member_suffix}")]
            if not members:
                raise FileNotFoundError(f"No {zip_member_suffix} found in zip: {source_path}")
            if len(members) > 1:
                raise RuntimeError(f"Zip contains multiple {zip_member_suffix} files: {members}")
            payload = archive.read(members[0]).decode(encoding)
        return list(csv.DictReader(payload.splitlines()))
    return read_csv_rows(source_path, encoding=encoding)


def read_csv_preview(
    path: Path,
    *,
    max_rows: int,
    fields: Sequence[str],
    encoding: str = "utf-8-sig",
) -> tuple[int, List[Dict[str, Any]]]:
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        return 0, []
    preview: List[Dict[str, Any]] = []
    row_count = 0
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_count += 1
            if len(preview) >= max_rows:
                continue
            selected = {
                field: row.get(field, "")
                for field in fields
                if str(row.get(field, "")).strip()
            }
            if selected:
                preview.append(selected)
    return row_count, preview