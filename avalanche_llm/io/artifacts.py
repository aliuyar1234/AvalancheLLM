from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..canon import get_canon
from .hashing import sha256_file, sha256_bytes
from .schema_validate import load_json_schema, validate_json


class RunError(RuntimeError):
    pass


def _now_utc() -> str:
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if ts.endswith("+00:00"):
        ts = ts[: -len("+00:00")] + "Z"
    return ts


def _git_code_version() -> str:
    try:
        import subprocess

        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            return out
    except Exception:
        pass
    return "no_git_metadata"


@dataclass
class RunWriter:
    run_dir: Path
    run_id: str
    phase_id: str
    stage_tag: str
    created_utc: str
    config_hash: str
    resolved_config_bytes: bytes
    allow_resume: bool
    _canon: dict[str, Any] = field(default_factory=get_canon, init=False)
    run_record: dict[str, Any] = field(default_factory=dict, init=False)

    def _path_run_record(self) -> Path:
        return self.run_dir / str(self._canon["OUTPUT"]["RUN_RECORD_JSON"])

    def _path_config_resolved(self) -> Path:
        return self.run_dir / str(self._canon["OUTPUT"]["CONFIG_RESOLVED_YAML"])

    def init_run_dir(self) -> None:
        if self.run_dir.exists():
            if not self.allow_resume:
                raise RunError(f"Run directory already exists: {self.run_dir}")
            record_path = self._path_run_record()
            if not record_path.is_file():
                raise RunError(
                    "allow_resume is set but the run record is missing: "
                    f"{self._canon['OUTPUT']['RUN_RECORD_JSON']}"
                )
            existing = json.loads(record_path.read_text(encoding="utf-8"))
            if existing.get("stage_status") == "complete":
                raise RunError("Refusing to resume a completed run")
            self.run_record = existing
            return

        self.run_dir.mkdir(parents=True, exist_ok=False)

        for sub in self._canon["OUTPUT"]["RUN_SUBDIR"].values():
            (self.run_dir / str(sub)).mkdir(parents=True, exist_ok=False)

    def write_resolved_config(self) -> None:
        path = self._path_config_resolved()
        if path.exists():
            if not self.allow_resume:
                raise RunError(f"Resolved config already exists: {path}")
            existing = path.read_bytes()
            if existing != self.resolved_config_bytes:
                raise RunError("Resolved config bytes differ for resumed run")
            return
        path.write_bytes(self.resolved_config_bytes)

    def start_run_record(
        self, *, model: dict[str, Any], dataset: dict[str, Any], determinism: dict[str, Any] | None = None
    ) -> None:
        def _superset(existing: Any, required: Any) -> bool:
            if isinstance(required, dict):
                if not isinstance(existing, dict):
                    return False
                for k, v in required.items():
                    if k not in existing:
                        return False
                    if not _superset(existing[k], v):
                        return False
                return True
            if isinstance(required, list):
                return isinstance(existing, list) and existing == required
            return existing == required

        if self.allow_resume and self.run_record:
            if self.run_record.get("run_id") != self.run_id:
                raise RunError("Resumed run_id mismatch in existing run_record.json")
            if self.run_record.get("phase_id") != self.phase_id:
                raise RunError("Resumed phase_id mismatch in existing run_record.json")
            if self.run_record.get("stage_tag") != self.stage_tag:
                raise RunError("Resumed stage_tag mismatch in existing run_record.json")
            hashes = self.run_record.get("hashes", {})
            if hashes.get("config_sha256") != self.config_hash:
                raise RunError("Resumed config_sha256 mismatch in existing run_record.json")
            if not _superset(self.run_record.get("model"), model):
                raise RunError("Resumed model block mismatch in existing run_record.json")
            if not _superset(self.run_record.get("dataset"), dataset):
                raise RunError("Resumed dataset block mismatch in existing run_record.json")
            if determinism is not None and not _superset(self.run_record.get("determinism"), determinism):
                raise RunError("Resumed determinism block mismatch in existing run_record.json")
            return

        project = self._canon["PROJECT"]
        try:
            import torch
        except Exception:  # pragma: no cover
            torch = None
        try:
            import transformers
        except Exception:  # pragma: no cover
            transformers = None

        self.run_record = {
            "project_id": str(project["PROJECT_ID"]),
            "pack_version": str(project["PACK_VERSION"]),
            "run_id": self.run_id,
            "phase_id": self.phase_id,
            "created_utc": self.created_utc,
            "stage_tag": self.stage_tag,
            "stage_status": "started",
            "model": model,
            "dataset": dataset,
            "conditions": [],
            "artifacts": {},
            "hashes": {
                "config_sha256": self.config_hash,
                "code_version": _git_code_version(),
            },
            "versions": {
                "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                "torch": getattr(torch, "__version__", None),
                "transformers": getattr(transformers, "__version__", None),
            },
            "determinism": determinism or {"PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")},
            "artifact_index": [],
        }
        self._write_run_record()

    def _write_run_record(self) -> None:
        path = self._path_run_record()
        text = json.dumps(self.run_record, indent=2, ensure_ascii=False)
        path.write_text(text + "\n", encoding="utf-8", newline="\n")

    def flush_run_record(self) -> None:
        self._write_run_record()

    def register_artifact(self, *, logical_name: str, relative_path: str) -> None:
        rel = Path(relative_path)
        abs_path = self.run_dir / rel
        if not abs_path.is_file():
            raise RunError(f"Artifact missing on disk: {abs_path}")
        digest = sha256_file(abs_path)
        size = abs_path.stat().st_size
        entry = {
            "relative_path": rel.as_posix(),
            "sha256": digest,
            "bytes": int(size),
            "created_utc": _now_utc(),
        }
        self.run_record.setdefault("artifacts", {})[logical_name] = rel.as_posix()
        self.run_record.setdefault("artifact_index", []).append(entry)
        self._write_run_record()

    def finish(self) -> None:
        self.run_record["stage_status"] = "complete"
        self._write_run_record()
        self._validate_run_record_schema()

    def _validate_run_record_schema(self) -> None:
        schema_candidates = list(Path.cwd().rglob("run_record.schema.json"))
        if not schema_candidates:
            raise RunError("Could not locate run_record.schema.json for validation")
        if len(schema_candidates) > 1:
            raise RunError(
                "Multiple run_record.schema.json files found; cannot validate deterministically"
            )
        schema = load_json_schema(schema_candidates[0])
        validate_json(self.run_record, schema)

    def write_json(self, *, relative_path: str, obj: Any) -> str:
        rel = Path(relative_path)
        abs_path = self.run_dir / rel
        if abs_path.exists():
            raise RunError(f"Refusing to overwrite artifact: {abs_path}")
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
        abs_path.write_bytes(data)
        return rel.as_posix()

    def write_bytes(self, *, relative_path: str, data: bytes) -> str:
        rel = Path(relative_path)
        abs_path = self.run_dir / rel
        if abs_path.exists():
            raise RunError(f"Refusing to overwrite artifact: {abs_path}")
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(data)
        return rel.as_posix()

    def write_text(self, *, relative_path: str, text: str) -> str:
        return self.write_bytes(relative_path=relative_path, data=text.encode("utf-8"))

    def content_sha256(self, data: bytes) -> str:
        return sha256_bytes(data)
