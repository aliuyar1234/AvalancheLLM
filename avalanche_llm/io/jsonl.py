from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_utc() -> str:
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if ts.endswith("+00:00"):
        ts = ts[: -len("+00:00")] + "Z"
    return ts


@dataclass
class JsonlLogger:
    path: Path
    run_id: str

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "ts_utc": _now_utc(),
            "run_id": self.run_id,
            "event_type": event_type,
            "payload": payload,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

