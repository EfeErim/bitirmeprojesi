#!/usr/bin/env python3
"""Live Colab telemetry utilities with local spool + Drive synchronization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _append_line(path: Path, line: str) -> None:
    _ensure_parent(path)
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
            payload = json.loads(path.read_text(encoding="utf-8"))
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
        heartbeat_sec: float = 15.0,
    ) -> None:
        self.notebook_name = str(notebook_name)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sync_interval_sec = float(max(0.5, sync_interval_sec))
        self.heartbeat_sec = float(max(1.0, heartbeat_sec))

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
        self._last_heartbeat = 0.0
        self._drive_available = True
        self._sequence = 0
        self._bg_heartbeat_thread: Optional[threading.Thread] = None
        self._bg_heartbeat_stop = threading.Event()

        self.local_run_dir.mkdir(parents=True, exist_ok=True)
        self.drive_run_dir.mkdir(parents=True, exist_ok=True)
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
            payload = json.loads(self.local_artifact_index_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return [dict(item) for item in payload if isinstance(item, dict)]
        except Exception:
            pass
        return []

    def _store_sync_state(self) -> None:
        self._sync_state.last_sync_ts = _utc_now_iso()
        self.local_sync_state_path.write_text(_safe_json(self._sync_state.to_dict()), encoding="utf-8")

    def _store_artifact_index(self) -> None:
        body = _safe_json(self._artifact_index)
        self.local_artifact_index_path.write_text(body, encoding="utf-8")
        try:
            self.drive_artifact_index_path.parent.mkdir(parents=True, exist_ok=True)
            self.drive_artifact_index_path.write_text(body, encoding="utf-8")
        except Exception:
            self._drive_available = False

    def _periodic_sync(self, force: bool = False) -> None:
        now = time.time()
        should_sync = force or (now - self._last_sync_attempt) >= self.sync_interval_sec
        should_heartbeat = (now - self._last_heartbeat) >= self.heartbeat_sec
        if should_sync:
            self.sync_pending()
            self._last_sync_attempt = now
        if should_heartbeat:
            self.emit_event(
                "heartbeat",
                {"drive_available": bool(self._drive_available)},
                phase="heartbeat",
                level="debug",
                force_sync=False,
            )
            self._last_heartbeat = now

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
        body = _safe_json(payload)
        self.local_latest_path.write_text(body, encoding="utf-8")
        try:
            self.drive_latest_path.parent.mkdir(parents=True, exist_ok=True)
            self.drive_latest_path.write_text(body, encoding="utf-8")
            self._drive_available = True
        except Exception:
            self._drive_available = False
        self._periodic_sync()

    def write_json_artifact(self, relative_path: str, payload: Dict[str, Any]) -> Path:
        body = _safe_json(payload)
        return self.write_text_artifact(relative_path, body)

    def write_text_artifact(self, relative_path: str, text: str) -> Path:
        rel = Path(relative_path)
        local_path = self.local_artifacts_dir / rel
        drive_path = self.artifacts_dir / rel
        _ensure_parent(local_path)
        local_path.write_text(text, encoding="utf-8")
        try:
            _ensure_parent(drive_path)
            drive_path.write_text(text, encoding="utf-8")
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
        local_path = self.local_artifacts_dir / rel
        drive_path = self.artifacts_dir / rel
        _ensure_parent(local_path)
        local_path.write_bytes(content)
        try:
            _ensure_parent(drive_path)
            drive_path.write_bytes(content)
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
        local_path = self.local_artifacts_dir / rel
        drive_path = self.artifacts_dir / rel
        _ensure_parent(local_path)
        shutil.copy2(src, local_path)
        try:
            _ensure_parent(drive_path)
            shutil.copy2(src, drive_path)
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
            _ensure_parent(self.drive_events_path)
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
            _ensure_parent(self.drive_log_path)
            with self.drive_log_path.open("ab") as handle:
                handle.write(chunk)
            self._sync_state.logs_synced_bytes = len(local_bytes)
            self._drive_available = True
        except Exception:
            self._drive_available = False

    # ------------------------------------------------------------------
    # Background heartbeat daemon – keeps the runtime "alive" for Colab's
    # backend manager even when the Python kernel is blocked inside a long
    # synchronous call such as adapter.train_increment().
    # ------------------------------------------------------------------

    def start_background_heartbeat(
        self,
        interval_sec: Optional[float] = None,
        stdout_echo: bool = True,
    ) -> None:
        """Launch a daemon thread that keeps the Colab runtime alive.

        The thread prints a brief status line to **stdout** on every tick.
        This is the primary keep-alive signal: Colab's backend monitors
        kernel stdout/IOPub activity and will *not* recycle a runtime that
        is producing output — even when the browser tab is fully closed.

        The thread also persists heartbeat events to local spool + Drive
        for post-hoc diagnostics.

        Call :meth:`stop_background_heartbeat` (or :meth:`close`) to stop.
        """
        if self._bg_heartbeat_thread is not None and self._bg_heartbeat_thread.is_alive():
            return  # already running
        self._bg_heartbeat_stop.clear()
        interval = float(interval_sec or self.heartbeat_sec)
        _echo = bool(stdout_echo)

        def _heartbeat_loop() -> None:
            seq = 0
            while not self._bg_heartbeat_stop.is_set():
                seq += 1
                try:
                    self.emit_event(
                        "heartbeat",
                        {"drive_available": bool(self._drive_available), "source": "background_thread"},
                        phase="heartbeat",
                        level="debug",
                        force_sync=True,
                    )
                except Exception:
                    pass  # best-effort; never crash the daemon
                # --- stdout keep-alive: the line Colab's backend sees ---
                if _echo:
                    try:
                        print(
                            f"[HEARTBEAT] #{seq} alive  "
                            f"run={self.run_id}  "
                            f"drive={'ok' if self._drive_available else 'NO'}",
                            flush=True,
                        )
                    except Exception:
                        pass
                self._bg_heartbeat_stop.wait(interval)

        self._bg_heartbeat_thread = threading.Thread(
            target=_heartbeat_loop, name="TelemetryBackgroundHeartbeat", daemon=True
        )
        self._bg_heartbeat_thread.start()
        logger.debug("Background heartbeat started (interval=%.1fs, stdout_echo=%s)", interval, _echo)

    def stop_background_heartbeat(self) -> None:
        """Stop the background heartbeat daemon thread."""
        self._bg_heartbeat_stop.set()
        if self._bg_heartbeat_thread is not None:
            self._bg_heartbeat_thread.join(timeout=max(1.0, self.heartbeat_sec + 1.0))
            self._bg_heartbeat_thread = None
        logger.debug("Background heartbeat stopped")

    def close(self, final_payload: Optional[Dict[str, Any]] = None) -> None:
        self.stop_background_heartbeat()
        summary = {
            "ts": _utc_now_iso(),
            "run_id": self.run_id,
            "notebook": self.notebook_name,
            "final_payload": dict(final_payload or {}),
            "artifact_count": len(self._artifact_index),
            "drive_available": bool(self._drive_available),
        }
        body = _safe_json(summary)
        self.local_summary_path.write_text(body, encoding="utf-8")
        try:
            self.drive_summary_path.parent.mkdir(parents=True, exist_ok=True)
            self.drive_summary_path.write_text(body, encoding="utf-8")
            self._drive_available = True
        except Exception:
            self._drive_available = False
        self.emit_event("run_finished", summary, phase="final", force_sync=False)
        self.sync_pending()
