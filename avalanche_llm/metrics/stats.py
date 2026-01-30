from __future__ import annotations

import numpy as np


def bootstrap_ci(values: np.ndarray, *, reps: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(reps):
        idx = rng.integers(0, v.size, size=v.size)
        means.append(float(np.mean(v[idx])))
    means = np.array(means, dtype=np.float64)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi

