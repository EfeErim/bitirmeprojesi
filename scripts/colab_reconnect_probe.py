#!/usr/bin/env python3
"""Helpers for diagnosing Colab reconnect stalls in notebook 2."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any, Dict, Optional, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return sum(1 for _ in path.open("r", encoding="utf-8"))
    except Exception:
        return 0


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _append_line(path: Path, line: str) -> None:
    _ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _resolve_run_dirs(
    *,
    state: Optional[Dict[str, Any]] = None,
    telemetry: Any = None,
) -> Tuple[Path, Path]:
    state = state or {}
    run_id = str(getattr(telemetry, "run_id", "") or "").strip()

    drive_run = getattr(telemetry, "drive_run_dir", None)
    local_run = getattr(telemetry, "local_run_dir", None)
    if drive_run is not None and local_run is not None:
        return Path(drive_run), Path(local_run)

    drive_hint = str(state.get("telemetry_run_dir", "") or "").strip()
    if drive_hint:
        drive_path = Path(drive_hint)
        if not run_id:
            run_id = drive_path.name
        local_root = Path("/content/aads_ulora/telemetry_spool")
        local_path = local_root / (run_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
        return drive_path, local_path

    if not run_id:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    drive_root = Path("/content/drive/MyDrive/aads_ulora") / "telemetry" / run_id
    local_root = Path("/content/aads_ulora/telemetry_spool") / run_id
    return drive_root, local_root


def _resolve_latest_manifest_path(
    *,
    drive_run_dir: Path,
    checkpoint_manager: Any = None,
) -> Path:
    manager_path = getattr(checkpoint_manager, "latest_manifest_path", None)
    if manager_path is not None:
        return Path(manager_path)
    return drive_run_dir / "latest_checkpoint.json"


def _read_gpu_info() -> Dict[str, Any]:
    try:
        import torch  # type: ignore
    except Exception:
        return {"cuda_available": False, "gpu_name": ""}
    cuda_ok = bool(torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else ""
    return {"cuda_available": cuda_ok, "gpu_name": gpu_name}


def capture_preflight_identity(
    *,
    state: Optional[Dict[str, Any]] = None,
    telemetry: Any = None,
    checkpoint_manager: Any = None,
    browser_info: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Collect preflight run identity and persist it to investigation artifacts."""
    state = state or {}
    drive_run_dir, local_run_dir = _resolve_run_dirs(state=state, telemetry=telemetry)
    latest_manifest_path = _resolve_latest_manifest_path(
        drive_run_dir=drive_run_dir,
        checkpoint_manager=checkpoint_manager,
    )

    latest_checkpoint_meta = None
    if checkpoint_manager is not None and hasattr(checkpoint_manager, "get_latest"):
        try:
            latest_checkpoint_meta = checkpoint_manager.get_latest()
        except Exception:
            latest_checkpoint_meta = None
    if latest_checkpoint_meta is None and isinstance(state.get("resume_manifest"), dict):
        latest_checkpoint_meta = dict(state["resume_manifest"])
    if latest_checkpoint_meta is None:
        latest_checkpoint_meta = _read_json(latest_manifest_path, None)

    widget_cfg = state.get("widget_config", {}) if isinstance(state.get("widget_config"), dict) else {}
    resume_mode = str(widget_cfg.get("resume_mode", "unknown"))
    payload: Dict[str, Any] = {
        "ts": _utc_now_iso(),
        "run_id": str(getattr(telemetry, "run_id", "") or ""),
        "drive_run_dir": str(drive_run_dir),
        "local_run_dir": str(local_run_dir),
        "resume_mode": resume_mode,
        "latest_checkpoint_meta": latest_checkpoint_meta if isinstance(latest_checkpoint_meta, dict) else {},
        "browser_info": str(browser_info or "").strip(),
        "gpu": _read_gpu_info(),
        "notes": dict(extra or {}),
    }

    local_path = local_run_dir / "investigation" / "preflight_identity.json"
    drive_path = drive_run_dir / "investigation" / "preflight_identity.json"
    _ensure_parent(local_path)
    _ensure_parent(drive_path)
    body = json.dumps(payload, indent=2, ensure_ascii=False)
    local_path.write_text(body, encoding="utf-8")
    try:
        drive_path.write_text(body, encoding="utf-8")
    except Exception:
        pass
    return payload


@dataclass
class ReconnectEvidenceProbe:
    """Periodically samples telemetry/checkpoint health for reconnect diagnostics."""

    drive_run_dir: Path
    local_run_dir: Path
    latest_manifest_path: Path
    interval_sec: float = 15.0
    echo: bool = True

    def __post_init__(self) -> None:
        self.interval_sec = float(max(1.0, self.interval_sec))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.local_probe_path = self.local_run_dir / "investigation" / "reconnect_probe.jsonl"
        self.drive_probe_path = self.drive_run_dir / "investigation" / "reconnect_probe.jsonl"
        self.local_trial_events_path = self.local_run_dir / "investigation" / "trial_timestamps.jsonl"
        self.drive_trial_events_path = self.drive_run_dir / "investigation" / "trial_timestamps.jsonl"

    @classmethod
    def from_runtime(
        cls,
        *,
        state: Optional[Dict[str, Any]] = None,
        telemetry: Any = None,
        checkpoint_manager: Any = None,
        interval_sec: float = 15.0,
        echo: bool = True,
    ) -> "ReconnectEvidenceProbe":
        drive_run_dir, local_run_dir = _resolve_run_dirs(state=state, telemetry=telemetry)
        latest_manifest_path = _resolve_latest_manifest_path(
            drive_run_dir=drive_run_dir,
            checkpoint_manager=checkpoint_manager,
        )
        return cls(
            drive_run_dir=Path(drive_run_dir),
            local_run_dir=Path(local_run_dir),
            latest_manifest_path=Path(latest_manifest_path),
            interval_sec=interval_sec,
            echo=echo,
        )

    def _read_checkpoint_meta(self) -> Dict[str, Any]:
        payload = _read_json(self.latest_manifest_path, {})
        if not isinstance(payload, dict):
            return {}
        return payload

    def _read_latest_status(self) -> Dict[str, Any]:
        paths = [self.drive_run_dir / "latest_status.json", self.local_run_dir / "latest_status.json"]
        for path in paths:
            payload = _read_json(path, None)
            if isinstance(payload, dict):
                return payload
        return {}

    def _resolve_events_path(self) -> Path:
        drive_path = self.drive_run_dir / "events.jsonl"
        if drive_path.exists():
            return drive_path
        return self.local_run_dir / "events.jsonl"

    def snapshot_once(self) -> Dict[str, Any]:
        events_path = self._resolve_events_path()
        latest_status_path = self.drive_run_dir / "latest_status.json"
        if not latest_status_path.exists():
            latest_status_path = self.local_run_dir / "latest_status.json"

        latest_status = self._read_latest_status()
        latest_checkpoint = self._read_checkpoint_meta()
        status_blob = latest_status.get("status", {}) if isinstance(latest_status.get("status", {}), dict) else {}
        checkpoint_exists = bool(latest_checkpoint)

        snapshot: Dict[str, Any] = {
            "ts": _utc_now_iso(),
            "events_path": str(events_path),
            "events_lines": _count_lines(events_path),
            "latest_status_path": str(latest_status_path),
            "latest_status_mtime_utc": (
                datetime.fromtimestamp(latest_status_path.stat().st_mtime, tz=timezone.utc).isoformat()
                if latest_status_path.exists()
                else ""
            ),
            "latest_phase": str(status_blob.get("phase", "")),
            "latest_epoch": int(status_blob.get("epoch", 0)) if status_blob.get("epoch") is not None else 0,
            "latest_batch": int(status_blob.get("batch", 0)) if status_blob.get("batch") is not None else 0,
            "latest_global_step": int(status_blob.get("global_step", 0))
            if status_blob.get("global_step") is not None
            else 0,
            "checkpoint_manifest_path": str(self.latest_manifest_path),
            "checkpoint_found": checkpoint_exists,
            "checkpoint_epoch": int(latest_checkpoint.get("epoch", 0)) if checkpoint_exists else 0,
            "checkpoint_global_step": int(latest_checkpoint.get("global_step", 0)) if checkpoint_exists else 0,
            "checkpoint_name": str(latest_checkpoint.get("name", "")) if checkpoint_exists else "",
        }
        line = _safe_json(snapshot)
        _append_line(self.local_probe_path, line)
        try:
            _append_line(self.drive_probe_path, line)
        except Exception:
            pass

        if self.echo:
            print(
                "[probe] "
                f"events={snapshot['events_lines']} "
                f"phase={snapshot['latest_phase'] or 'n/a'} "
                f"epoch={snapshot['latest_epoch']} "
                f"step={snapshot['latest_global_step']} "
                f"ckpt_step={snapshot['checkpoint_global_step']}"
            )
        return snapshot

    def record_trial_event(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        record = {
            "ts": _utc_now_iso(),
            "event_type": str(event_type),
            "payload": dict(payload or {}),
        }
        line = _safe_json(record)
        _append_line(self.local_trial_events_path, line)
        try:
            _append_line(self.drive_trial_events_path, line)
        except Exception:
            pass
        return record

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self.snapshot_once()
            self._stop_event.wait(self.interval_sec)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self.record_trial_event("probe_started", {"interval_sec": self.interval_sec})
        self._thread = threading.Thread(target=self._loop, name="ReconnectEvidenceProbe", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_sec + 1.0))
        self.record_trial_event("probe_stopped", {})
        self.snapshot_once()


def log_trial_timestamp(
    probe: ReconnectEvidenceProbe,
    *,
    label: str,
    close_ts_utc: str = "",
    reopen_ts_utc: str = "",
    first_reconnect_ts_utc: str = "",
    ui_state: str = "",
    ram_disk_busy: Optional[bool] = None,
    note: str = "",
) -> Dict[str, Any]:
    """Append a normalized reconnect trial event entry."""
    payload: Dict[str, Any] = {
        "label": str(label),
        "close_ts_utc": str(close_ts_utc).strip(),
        "reopen_ts_utc": str(reopen_ts_utc).strip(),
        "first_reconnect_ts_utc": str(first_reconnect_ts_utc).strip(),
        "ui_state": str(ui_state).strip(),
        "note": str(note).strip(),
    }
    if ram_disk_busy is not None:
        payload["ram_disk_busy"] = bool(ram_disk_busy)
    return probe.record_trial_event("trial_timestamp", payload)
