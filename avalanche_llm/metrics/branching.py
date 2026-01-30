from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Branching:
    b_time: float
    b_depth: float

    @property
    def b_tot(self) -> float:
        return float(self.b_time + self.b_depth)


def pooled_branching(x: np.ndarray) -> Branching:
    """
    x: occupancy array [L, T], pooled estimator per spec/06.
    """
    if x.ndim != 2:
        raise ValueError("x must be [L, T]")
    x01 = (x > 0).astype(np.uint8)
    L, T = x01.shape
    denom = float(np.sum(x01[:, :], dtype=np.int64))
    if denom == 0.0:
        return Branching(b_time=float("nan"), b_depth=float("nan"))
    num_time = float(np.sum(x01[:, :-1] * x01[:, 1:], dtype=np.int64))
    num_depth = float(np.sum(x01[:-1, :] * x01[1:, :], dtype=np.int64))
    return Branching(b_time=num_time / denom, b_depth=num_depth / denom)

