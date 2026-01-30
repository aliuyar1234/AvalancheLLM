from __future__ import annotations

import hashlib
from pathlib import Path

from avalanche_llm.io.artifacts import RunWriter


def test_run_record_schema_validation(tmp_path: Path) -> None:
    resolved = b"pipeline:\\n  seq_len: 1\\n"
    config_hash = hashlib.sha256(resolved).hexdigest()
    writer = RunWriter(
        run_dir=tmp_path / "RUN_TEST",
        run_id="RUN_TEST",
        phase_id="PHASE_TEST",
        stage_tag="SXX",
        created_utc="2026-01-01T00:00:00Z",
        config_hash=config_hash,
        resolved_config_bytes=resolved,
        allow_resume=False,
    )
    writer.init_run_dir()
    writer.write_resolved_config()
    writer.start_run_record(model={}, dataset={})
    writer.finish()

