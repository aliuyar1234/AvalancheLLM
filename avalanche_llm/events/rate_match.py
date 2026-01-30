from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..canon import get_canon


@dataclass(frozen=True)
class TauResult:
    tau: float
    achieved_rate: float


def _round_down(x: float, step: float) -> float:
    return math.floor(x / step) * step


def _round_up(x: float, step: float) -> float:
    return math.ceil(x / step) * step


def select_tau_histogram(v_samples: np.ndarray, target_rate: float) -> TauResult:
    """
    Deterministic histogram tail-quantile estimator per spec/04.
    Returns tau such that P(v > tau) ~= target_rate.
    """
    canon = get_canon()
    bins = int(canon["CONST"]["RATE_MATCH_HIST_BINS"])
    edge_step = float(canon["CONST"]["RATE_MATCH_EDGE_ROUND_ABS"])

    v = np.asarray(v_samples, dtype=np.float64).ravel()
    if v.size == 0:
        raise ValueError("Empty v_samples")

    v_min = float(np.nanmin(v))
    v_max = float(np.nanmax(v))
    if not np.isfinite(v_min) or not np.isfinite(v_max):
        raise ValueError("Non-finite v_min/v_max")

    v_min_r = _round_down(v_min, edge_step)
    v_max_r = _round_up(v_max, edge_step)
    if v_min_r == v_max_r:
        raise ValueError("Degenerate histogram range")

    hist, edges = np.histogram(v, bins=bins, range=(v_min_r, v_max_r))
    cdf = np.cumsum(hist, dtype=np.int64) / float(v.size)

    # Tail prob at edge i is 1 - CDF(edge_i)
    tail = 1.0 - cdf
    # Choose the smallest threshold such that tail <= target_rate (i.e. we spike no more than target).
    idx = int(np.argmax(tail <= target_rate))
    tau = float(edges[min(idx + 1, len(edges) - 1)])
    achieved = float(np.mean(v > tau))
    return TauResult(tau=tau, achieved_rate=achieved)


def verify_rate_match(v_samples: np.ndarray, target_rate: float) -> TauResult:
    canon = get_canon()
    tol = float(canon["CONST"]["RATE_MATCH_TOL_ABS"])
    out = select_tau_histogram(v_samples=v_samples, target_rate=target_rate)
    if abs(out.achieved_rate - float(target_rate)) > tol:
        raise ValueError(
            f"Rate match failed: achieved={out.achieved_rate} target={target_rate} tol={tol}"
        )
    return out

