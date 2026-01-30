from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from datasets import load_dataset

from ..canon import get_canon


class DatasetError(RuntimeError):
    pass


@dataclass(frozen=True)
class TokenWindow:
    dataset_role: str
    chunk_index: int
    input_ids: list[int]


def _sha256_json(obj: Any) -> str:
    data = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _text_column(ds: Any) -> str:
    if "text" in ds.column_names:
        return "text"
    # Fail-closed: require an unambiguous text-like field.
    string_cols = []
    for name in ds.column_names:
        feat = ds.features.get(name)
        if getattr(feat, "dtype", None) == "string":
            string_cols.append(name)
    if len(string_cols) == 1:
        return string_cols[0]
    raise DatasetError(f"Could not determine text column; candidates={string_cols}")


def _token_windows(ds: Any, tokenizer: Any, *, seq_len: int, dataset_role: str) -> Iterable[TokenWindow]:
    col = _text_column(ds)
    buf: list[int] = []
    chunk_index = 0
    for ex in ds:
        text = ex.get(col)
        if text is None:
            continue
        if not isinstance(text, str):
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        buf.extend(int(i) for i in ids)
        while len(buf) >= seq_len:
            window = buf[:seq_len]
            del buf[:seq_len]
            yield TokenWindow(dataset_role=dataset_role, chunk_index=chunk_index, input_ids=window)
            chunk_index += 1


def _seed_from_run(*, run_id: str, dataset_role: str) -> int:
    canon = get_canon()
    base = int(canon["CONST"]["BOOTSTRAP_SEED"])
    h = hashlib.sha256(f"{base}|{run_id}|{dataset_role}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def select_token_windows(
    *,
    dataset_role: str,
    hf_id: str,
    hf_config: str | None,
    split: str,
    tokenizer: Any,
    seq_len: int,
    n_windows: int,
    run_id: str,
) -> tuple[list[TokenWindow], dict[str, Any]]:
    if n_windows <= 0:
        raise DatasetError("n_windows must be > 0")

    canon = get_canon()
    pool_factor = int(canon["CONST"]["WINDOW_POOL_FACTOR"])
    pool_cap = int(canon["CONST"]["BOOTSTRAP_REPS"])
    pool_size = max(n_windows, min(pool_cap, n_windows * pool_factor))

    ds_kwargs: dict[str, Any] = {"split": split}
    if hf_config:
        ds_kwargs["name"] = hf_config
    if dataset_role == str(canon["ENUM"]["DATASET_ROLE"]["B"]):
        ds_kwargs["streaming"] = True
    ds = load_dataset(hf_id, **ds_kwargs)

    windows = []
    for w in _token_windows(ds, tokenizer, seq_len=seq_len, dataset_role=dataset_role):
        windows.append(w)
        if len(windows) >= pool_size:
            break
    if len(windows) < n_windows:
        raise DatasetError(f"Not enough windows; have={len(windows)} need={n_windows}")

    rng = np.random.default_rng(_seed_from_run(run_id=run_id, dataset_role=dataset_role))
    idx = rng.choice(len(windows), size=n_windows, replace=False)
    idx_sorted = sorted(int(i) for i in idx.tolist())
    selected = [windows[i] for i in idx_sorted]

    slice_obj = [
        {"dataset_role": w.dataset_role, "chunk_index": w.chunk_index, "input_ids": w.input_ids} for w in selected
    ]
    slice_sha = _sha256_json(slice_obj)
    meta = {
        "dataset_role": dataset_role,
        "hf_id": hf_id,
        "hf_config": hf_config,
        "split": split,
        "seq_len": seq_len,
        "n_windows": n_windows,
        "pool_size": len(windows),
        "selected_pool_indices": idx_sorted,
        "dataset_slice_sha256": slice_sha,
        "dataset_fingerprint": getattr(ds, "_fingerprint", None),
    }
    return selected, meta
