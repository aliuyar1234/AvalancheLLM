from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema


class SchemaError(RuntimeError):
    pass


def validate_json(instance: Any, schema: dict[str, Any]) -> None:
    try:
        jsonschema.validate(instance=instance, schema=schema)
    except jsonschema.ValidationError as e:
        raise SchemaError(str(e)) from e


def load_json_schema(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SchemaError(f"Schema file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

