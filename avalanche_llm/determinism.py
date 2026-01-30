from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

from .canon import get_canon


def set_determinism(det_cfg: dict[str, Any] | None) -> dict[str, Any]:
    det_cfg = det_cfg or {}
    seed = int(det_cfg.get("seed", 0))
    torch_deterministic = bool(det_cfg.get("torch_deterministic", False))
    disable_tf32 = bool(det_cfg.get("disable_tf32", False))

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)

    out: dict[str, Any] = {
        "seed": seed,
        "torch_deterministic": torch_deterministic,
        "disable_tf32": disable_tf32,
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
    }

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch_deterministic:
            # Required for CuBLAS determinism on CUDA >= 10.2.
            canon = get_canon()
            cublas_cfg = str(canon["CONST"]["CUBLAS_WORKSPACE_CONFIG"])
            if not cublas_cfg:
                raise RuntimeError("Missing CANON.CONST.CUBLAS_WORKSPACE_CONFIG")
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", cublas_cfg)
            torch.use_deterministic_algorithms(True)
        if disable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        out.update(
            {
                "torch_use_deterministic_algorithms": bool(
                    getattr(torch, "are_deterministic_algorithms_enabled", lambda: False)()
                ),
                "torch_allow_tf32_matmul": getattr(torch.backends.cuda.matmul, "allow_tf32", None),
                "torch_allow_tf32_cudnn": getattr(torch.backends.cudnn, "allow_tf32", None),
            }
        )
    except Exception as e:  # pragma: no cover
        out["torch_error"] = str(e)

    return out
