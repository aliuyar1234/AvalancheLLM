from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class LoadedModel:
    model: torch.nn.Module
    tokenizer: Any
    hf_id: str
    model_revision: str | None
    tokenizer_revision: str | None


def _local_tree_manifest_sha256(root: Path) -> str:
    """
    Best-effort content fingerprint for local checkpoints without a VCS / HF commit hash.
    Uses only relative paths and file sizes (fast and deterministic, without hashing large weights).
    """
    items: list[dict[str, Any]] = []
    for p in sorted(root.rglob("*"), key=lambda x: x.as_posix()):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        st = p.stat()
        items.append({"path": rel, "bytes": int(st.st_size)})
    data = json.dumps(items, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def load_model(hf_id: str, *, device: str, dtype: str) -> LoadedModel:
    path = Path(hf_id)
    is_local = path.exists()
    model_id = str(path) if is_local else hf_id

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(dtype, None)
    if torch_dtype is None:
        raise ValueError(f"Unknown dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        model = model.to("cuda")
    elif device == "cpu":
        model = model.to("cpu")
    else:
        raise ValueError(f"Unknown device: {device}")
    model.eval()
    expected_device = device
    for name, p in model.named_parameters():
        if p.device.type != expected_device:
            raise RuntimeError(f"Parameter not on {expected_device}: {name} is on {p.device}")
    for name, b in model.named_buffers():
        if b.device.type != expected_device:
            raise RuntimeError(f"Buffer not on {expected_device}: {name} is on {b.device}")

    model_rev = getattr(getattr(model, "config", None), "_commit_hash", None)
    tok_rev = getattr(getattr(tokenizer, "init_kwargs", None), "get", lambda _k, _d=None: None)("revision", None)
    local_manifest = _local_tree_manifest_sha256(path) if is_local and path.is_dir() else None
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        hf_id=model_id,
        model_revision=str(model_rev) if model_rev else (str(local_manifest) if local_manifest else None),
        tokenizer_revision=str(tok_rev) if tok_rev else (str(local_manifest) if local_manifest else None),
    )
