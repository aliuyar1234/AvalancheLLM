from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone


class RunIdError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunId:
    run_id: str
    created_utc: str


def _utc_ts_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def make_run_id(
    *,
    stage_tag: str,
    mode: str,
    resolved_config_bytes: bytes,
    counter: int | None = None,
) -> RunId:
    stage_tag_norm = stage_tag.strip()
    if not stage_tag_norm:
        raise RunIdError("stage_tag must be non-empty")

    created_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if created_utc.endswith("+00:00"):
        created_utc = created_utc[: -len("+00:00")] + "Z"

    if mode == "content_hash":
        h = hashlib.sha256(resolved_config_bytes + stage_tag_norm.encode("utf-8")).hexdigest()[:12]
        return RunId(run_id=f"RUN_{stage_tag_norm}_{h}", created_utc=created_utc)
    if mode == "timestamp_counter":
        if counter is None:
            raise RunIdError("counter is required for timestamp_counter mode")
        stamp = _utc_ts_compact()
        return RunId(run_id=f"RUN_{stamp}_{counter:04d}", created_utc=created_utc)
    raise RunIdError(f"Unknown run_id_mode: {mode}")

