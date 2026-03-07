#!/usr/bin/env python3
"""Live Colab telemetry utilities with local spool + Drive synchronization."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.shared.artifacts import ArtifactStore
from src.shared.json_utils import ensure_parent, read_json, write_json

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)

def _append_line(path: Path, line: str) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        if not line.endswith("\n"):
            handle.write("\n")


@dataclass
class _SyncState:
    events_synced: int = 0
    logs_synced_bytes: int = 0
    last_sync_ts: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "events_synced": int(self.events_synced),
            "logs_synced_bytes": int(self.logs_synced_bytes),
            "last_sync_ts": self.last_sync_ts,
        }

    @classmethod
    def from_path(cls, path: Path) -> "_SyncState":
        if not path.exists():
            return cls()
        try:
            payload = read_json(path, default={}, expect_type=dict)
            return cls(
                events_synced=int(payload.get("events_synced", 0)),
                logs_synced_bytes=int(payload.get("logs_synced_bytes", 0)),
                last_sync_ts=str(payload.get("last_sync_ts", "")),
            )
        except Exception:
            return cls()


class ColabLiveTelemetry:
    """Write telemetry locally first, then synchronize to Drive best-effort."""

    def __init__(
        self,
        *,
        notebook_name: str,
        run_id: Optional[str] = None,
        drive_root: Optional[str | Path] = None,
        local_root: Optional[str | Path] = None,
        sync_interval_sec: float = 5.0,
    ) -> None:
        self.notebook_name = str(notebook_name)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sync_interval_sec = float(max(0.5, sync_interval_sec))

        drive_base = Path(
            drive_root
            or os.environ.get("AADS_DRIVE_LOG_ROOT", "/content/drive/MyDrive/aads_ulora")
        )
        local_base = Path(local_root or "/content/aads_ulora/telemetry_spool")

        self.drive_run_dir = drive_base / "telemetry" / self.run_id
        self.local_run_dir = local_base / self.run_id
        self.artifacts_dir = self.drive_run_dir / "artifacts"
        self.local_artifacts_dir = self.local_run_dir / "artifacts"

        self.local_events_path = self.local_run_dir / "events.jsonl"
        self.drive_events_path = self.drive_run_dir / "events.jsonl"
        self.local_log_path = self.local_run_dir / "runtime.log"
        self.drive_log_path = self.drive_run_dir / "runtime.log"
        self.local_latest_path = self.local_run_dir / "latest_status.json"
        self.drive_latest_path = self.drive_run_dir / "latest_status.json"
        self.local_summary_path = self.local_run_dir / "summary.json"
        self.drive_summary_path = self.drive_run_dir / "summary.json"
        self.local_artifact_index_path = self.local_run_dir / "artifact_index.json"
        self.drive_artifact_index_path = self.drive_run_dir / "artifact_index.json"
        self.local_sync_state_path = self.local_run_dir / "sync_state.json"

        self._sync_state = _SyncState.from_path(self.local_sync_state_path)
        self._artifact_index: List[Dict[str, Any]] = self._load_artifact_index()
        self._last_sync_attempt = 0.0
        self._drive_available = True
        self._sequence = 0

        self.local_run_dir.mkdir(parents=True, exist_ok=True)
        self.drive_run_dir.mkdir(parents=True, exist_ok=True)
        self._local_artifact_store = ArtifactStore(self.local_artifacts_dir)
        self._drive_artifact_store = ArtifactStore(self.artifacts_dir)
        self.emit_event(
            "run_started",
            {
                "notebook": self.notebook_name,
                "run_id": self.run_id,
                "drive_run_dir": str(self.drive_run_dir),
                "local_run_dir": str(self.local_run_dir),
            },
            phase="bootstrap",
        )

    def _load_artifact_index(self) -> List[Dict[str, Any]]:
        if not self.local_artifact_index_path.exists():
            return []
        try:
            payload = read_json(self.local_artifact_index_path, default=[], expect_type=list)
            if isinstance(payload, list):
                return [dict(item) for item in payload if isinstance(item, dict)]
        except Exception:
            pass
        return []

    def _store_sync_state(self) -> None:
        self._sync_state.last_sync_ts = _utc_now_iso()
        write_json(self.local_sync_state_path, self._sync_state.to_dict(), sort_keys=True)

    def _store_artifact_index(self) -> None:
        write_json(self.local_artifact_index_path, self._artifact_index, sort_keys=True)
        try:
            write_json(self.drive_artifact_index_path, self._artifact_index, sort_keys=True)
        except Exception:
            self._drive_available = False

    def _periodic_sync(self, force: bool = False) -> None:
        now = time.time()
        should_sync = force or (now - self._last_sync_attempt) >= self.sync_interval_sec
        if should_sync:
            self.sync_pending()
            self._last_sync_attempt = now

    def _append_event_local(self, payload: Dict[str, Any]) -> None:
        _append_line(self.local_events_path, _safe_json(payload))

    def _append_event_drive(self, payload: Dict[str, Any]) -> None:
        _append_line(self.drive_events_path, _safe_json(payload))

    def _append_log_local(self, line: str) -> None:
        _append_line(self.local_log_path, line)

    def _append_log_drive(self, line: str) -> None:
        _append_line(self.drive_log_path, line)

    def emit_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        *,
        phase: str = "runtime",
        level: str = "info",
        force_sync: bool = True,
    ) -> Dict[str, Any]:
        self._sequence += 1
        record = {
            "ts": _utc_now_iso(),
            "seq": int(self._sequence),
            "run_id": self.run_id,
            "notebook": self.notebook_name,
            "phase": str(phase),
            "event_type": str(event_type),
            "level": str(level).lower(),
            "payload": dict(payload or {}),
        }
        self._append_event_local(record)
        if force_sync:
            self._periodic_sync()
        return record

    def emit_log(self, message: str, *, phase: str = "runtime", level: str = "info") -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp} [{level.upper()}] [{phase}] {message}"
        self._append_log_local(line)
        self.emit_event("log_line", {"message": message}, phase=phase, level=level, force_sync=False)
        self._periodic_sync()

    def update_latest(self, status_payload: Dict[str, Any]) -> None:
        payload = {
            "ts": _utc_now_iso(),
            "run_id": self.run_id,
            "notebook": self.notebook_name,
            "status": dict(status_payload or {}),
        }
        write_json(self.local_latest_path, payload, sort_keys=True)
        try:
            write_json(self.drive_latest_path, payload, sort_keys=True)
            self._drive_available = True
        except Exception:
            self._drive_available = False
        self._periodic_sync()

    def write_json_artifact(self, relative_path: str, payload: Dict[str, Any]) -> Path:
        body = _safe_json(payload)
        return self.write_text_artifact(relative_path, body)

    def write_text_artifact(self, relative_path: str, text: str) -> Path:
        rel = Path(relative_path)
        local_path = self._local_artifact_store.write_text(rel, text)
        drive_path = self.artifacts_dir / rel
        try:
            drive_path = self._drive_artifact_store.write_text(rel, text)
            self._drive_available = True
        except Exception:
            self._drive_available = False
        self._artifact_index.append(
            {
                "ts": _utc_now_iso(),
                "relative_path": str(rel).replace("\\", "/"),
                "type": "text",
            }
        )
        self._store_artifact_index()
        self.emit_event("artifact_written", {"relative_path": str(rel)}, phase="artifact", force_sync=False)
        return drive_path if drive_path.exists() else local_path

    def write_binary_artifact(self, relative_path: str, content: bytes) -> Path:
        rel = Path(relative_path)
        local_path = self._local_artifact_store.write_bytes(rel, content)
        drive_path = self.artifacts_dir / rel
        try:
            drive_path = self._drive_artifact_store.write_bytes(rel, content)
            self._drive_available = True
        except Exception:
            self._drive_available = False
        self._artifact_index.append(
            {
                "ts": _utc_now_iso(),
                "relative_path": str(rel).replace("\\", "/"),
                "type": "binary",
            }
        )
        self._store_artifact_index()
        self.emit_event("artifact_written", {"relative_path": str(rel)}, phase="artifact", force_sync=False)
        return drive_path if drive_path.exists() else local_path

    def copy_artifact_file(self, source_path: str | Path, relative_path: str) -> Path:
        src = Path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"Artifact source does not exist: {src}")
        rel = Path(relative_path)
        local_path = self._local_artifact_store.copy_file(src, rel)
        drive_path = self.artifacts_dir / rel
        try:
            drive_path = self._drive_artifact_store.copy_file(src, rel)
            self._drive_available = True
        except Exception:
            self._drive_available = False
        self._artifact_index.append(
            {
                "ts": _utc_now_iso(),
                "relative_path": str(rel).replace("\\", "/"),
                "type": "file_copy",
            }
        )
        self._store_artifact_index()
        self.emit_event("artifact_copied", {"relative_path": str(rel)}, phase="artifact", force_sync=False)
        return drive_path if drive_path.exists() else local_path

    def sync_pending(self) -> None:
        self.drive_run_dir.mkdir(parents=True, exist_ok=True)
        self._sync_events()
        self._sync_logs()
        self._store_artifact_index()
        self._store_sync_state()

    def _sync_events(self) -> None:
        if not self.local_events_path.exists():
            return
        lines = self.local_events_path.read_text(encoding="utf-8").splitlines()
        start = max(0, int(self._sync_state.events_synced))
        if start >= len(lines):
            return
        unsynced = lines[start:]
        try:
            ensure_parent(self.drive_events_path)
            with self.drive_events_path.open("a", encoding="utf-8") as handle:
                for line in unsynced:
                    handle.write(line + "\n")
            self._sync_state.events_synced = len(lines)
            self._drive_available = True
        except Exception:
            self._drive_available = False

    def _sync_logs(self) -> None:
        if not self.local_log_path.exists():
            return
        local_bytes = self.local_log_path.read_bytes()
        offset = max(0, int(self._sync_state.logs_synced_bytes))
        if offset >= len(local_bytes):
            return
        chunk = local_bytes[offset:]
        try:
            ensure_parent(self.drive_log_path)
            with self.drive_log_path.open("ab") as handle:
                handle.write(chunk)
            self._sync_state.logs_synced_bytes = len(local_bytes)
            self._drive_available = True
        except Exception:
            self._drive_available = False

    def close(self, final_payload: Optional[Dict[str, Any]] = None) -> None:
        summary = {
            "ts": _utc_now_iso(),
            "run_id": self.run_id,
            "notebook": self.notebook_name,
            "final_payload": dict(final_payload or {}),
            "artifact_count": len(self._artifact_index),
            "drive_available": bool(self._drive_available),
        }
        write_json(self.local_summary_path, summary, sort_keys=True)
        try:
            write_json(self.drive_summary_path, summary, sort_keys=True)
            self._drive_available = True
        except Exception:
            self._drive_available = False
        self.emit_event("run_finished", summary, phase="final", force_sync=False)
        self.sync_pending()
