from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .canon import canon_get
from .canon import get_canon


class ConfigError(RuntimeError):
    pass


def _resolve_node(node: Any) -> Any:
    if isinstance(node, str) and node.startswith("CANON."):
        return canon_get(node)
    if isinstance(node, list):
        return [_resolve_node(x) for x in node]
    if isinstance(node, dict):
        return {k: _resolve_node(v) for k, v in node.items()}
    return node


def _contains_unresolved_canon(node: Any) -> bool:
    if isinstance(node, str):
        return node.startswith("CANON.")
    if isinstance(node, list):
        return any(_contains_unresolved_canon(x) for x in node)
    if isinstance(node, dict):
        return any(_contains_unresolved_canon(v) for v in node.values())
    return False


def _dump_yaml_canonical(data: Any) -> bytes:
    text = yaml.safe_dump(
        data,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.encode("utf-8")


@dataclass(frozen=True)
class ResolvedConfig:
    data: dict[str, Any]
    resolved_bytes: bytes
    sha256: str


def load_and_resolve_pipeline(config_path: Path) -> ResolvedConfig:
    if not config_path.is_file():
        raise ConfigError(f"Config does not exist: {config_path}")

    root = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if root is None:
        root = {}
    if not isinstance(root, dict):
        raise ConfigError("Pipeline config must be a mapping at the top level")

    # Load optional companion config files that phases depend on.
    # This keeps --config stable while still making the pack runnable.
    base_dir = config_path.parent
    companion_files = [
        "models.yaml",
        "datasets.yaml",
        "gains.yaml",
        "determinism.yaml",
        "budgets.yaml",
        "thresholds.yaml",
        "checklists.yaml",
    ]
    for filename in companion_files:
        path = base_dir / filename
        if not path.is_file():
            continue
        comp = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(comp, dict):
            for k, v in comp.items():
                if k not in root:
                    root[k] = v

    resolved = _resolve_node(root)
    if not isinstance(resolved, dict):
        raise ConfigError("Resolved config must be a mapping")
    if _contains_unresolved_canon(resolved):
        raise ConfigError("Unresolved CANON reference remained after resolution")

    resolved_bytes = _dump_yaml_canonical(resolved)
    sha = hashlib.sha256(resolved_bytes).hexdigest()
    return ResolvedConfig(data=resolved, resolved_bytes=resolved_bytes, sha256=sha)


def apply_cli_overrides(*, resolved: ResolvedConfig, model_role: str | None) -> ResolvedConfig:
    """
    Apply CLI-level overrides that MUST affect config hashing and run_id generation.
    """
    if model_role is None:
        return resolved

    canon = get_canon()
    role_norm = str(model_role).strip().upper()
    roles = canon.get("ENUM", {}).get("MODEL_ROLE", {})
    role_base = str(roles.get("BASE", "BASE"))
    role_instruct = str(roles.get("INSTRUCT", "INSTRUCT"))
    if role_norm not in {role_base, role_instruct}:
        raise ConfigError(f"Unknown model_role: {model_role}")

    models = resolved.data.get("models", {})
    if not isinstance(models, dict):
        raise ConfigError("Resolved config models must be a mapping")
    role_key = "base" if role_norm == role_base else "instruct"
    model_cfg = models.get(role_key, {})
    if not isinstance(model_cfg, dict):
        raise ConfigError(f"Resolved config models.{role_key} must be a mapping")
    hf_id = model_cfg.get("hf_id")
    if not isinstance(hf_id, str) or not hf_id:
        raise ConfigError(f"Resolved config missing models.{role_key}.hf_id")

    data = dict(resolved.data)
    data["model_role"] = role_norm
    data["model_selected"] = {"role": role_norm, "hf_id": hf_id}
    resolved_bytes = _dump_yaml_canonical(data)
    sha = hashlib.sha256(resolved_bytes).hexdigest()
    return ResolvedConfig(data=data, resolved_bytes=resolved_bytes, sha256=sha)
