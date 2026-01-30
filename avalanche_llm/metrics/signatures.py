from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..canon import get_canon
from ..raster.cc import ComponentStats


def chi_susceptibility(a: np.ndarray) -> float:
    """
    Susceptibility proxy per spec/09:
    chi = Var(Y) / (Mean(Y) + CHI_EPS), where Y is total activity per sequence.

    a: count raster [N, L, T]
    """
    if a.ndim != 3:
        raise ValueError("a must be [N, L, T]")
    canon = get_canon()
    eps = float(canon["CONST"]["CHI_EPS"])
    y = np.sum(a, axis=(1, 2), dtype=np.float64)
    mean = float(np.mean(y))
    var = float(np.var(y))
    return var / (mean + eps)


@dataclass(frozen=True)
class CracklingFitDiagnostics:
    gamma: float
    ci_low: float
    ci_high: float
    ci_width: float
    n_duration_points: int
    n_avalanches_used: int
    r2: float
    passed: bool


def crackling_fit_diagnostics(comps: list[ComponentStats], *, seed: int) -> CracklingFitDiagnostics:
    """
    Crackling relation per spec/09 with explicit diagnostics:
    fit log E[S|D] ~ gamma log D over D in [CRACKLING_D_RANGE_MIN, CRACKLING_D_RANGE_MAX]
    with at least CRACKLING_MIN_POINTS duration points. Bootstrap CI uses BOOTSTRAP_REPS.
    """
    canon = get_canon()
    d_min = int(canon["CONST"]["CRACKLING_D_RANGE_MIN"])
    d_max = int(canon["CONST"]["CRACKLING_D_RANGE_MAX"])
    min_pts = int(canon["CONST"]["CRACKLING_MIN_POINTS"])
    reps = int(canon["CONST"]["BOOTSTRAP_REPS"])
    ci_width_max = float(canon["CONST"]["BOOTSTRAP_CI_WIDTH_MAX"])

    if not comps:
        return CracklingFitDiagnostics(
            gamma=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            ci_width=float("nan"),
            n_duration_points=0,
            n_avalanches_used=0,
            r2=float("nan"),
            passed=False,
        )

    ds = np.array([int(c.span_tokens) for c in comps], dtype=np.int64)
    ss = np.array([float(c.size) for c in comps], dtype=np.float64)
    keep = (ds >= d_min) & (ds <= d_max) & np.isfinite(ss) & (ss > 0.0)
    ds = ds[keep]
    ss = ss[keep]
    if ds.size == 0:
        return CracklingFitDiagnostics(
            gamma=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            ci_width=float("nan"),
            n_duration_points=0,
            n_avalanches_used=0,
            r2=float("nan"),
            passed=False,
        )

    def _fit_gamma(d: np.ndarray, s: np.ndarray) -> tuple[float, int, float] | None:
        uniq = np.unique(d)
        d_vals: list[float] = []
        mean_s: list[float] = []
        for dur in uniq.tolist():
            mask = d == dur
            m = float(np.mean(s[mask])) if np.any(mask) else float("nan")
            if not np.isfinite(m) or m <= 0.0:
                continue
            d_vals.append(float(dur))
            mean_s.append(m)
        n_pts = len(d_vals)
        if n_pts < min_pts:
            return None
        x = np.log(np.asarray(d_vals, dtype=np.float64))
        y = np.log(np.asarray(mean_s, dtype=np.float64))
        vx = float(np.var(x))
        if vx <= 0.0:
            return None
        mx = float(np.mean(x))
        my = float(np.mean(y))
        cov = float(np.mean((x - mx) * (y - my)))
        gamma = cov / vx
        intercept = my - gamma * mx
        y_hat = gamma * x + intercept
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - my) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
        return (float(gamma), n_pts, float(r2))

    fit = _fit_gamma(ds, ss)
    if fit is None:
        return CracklingFitDiagnostics(
            gamma=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            ci_width=float("nan"),
            n_duration_points=0,
            n_avalanches_used=int(ds.size),
            r2=float("nan"),
            passed=False,
        )

    gamma, n_pts, r2 = fit
    if not np.isfinite(gamma):
        return CracklingFitDiagnostics(
            gamma=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            ci_width=float("nan"),
            n_duration_points=n_pts,
            n_avalanches_used=int(ds.size),
            r2=float(r2),
            passed=False,
        )

    rng = np.random.default_rng(seed)
    gammas: list[float] = []
    for _ in range(reps):
        idx = rng.integers(0, ds.size, size=ds.size)
        gfit = _fit_gamma(ds[idx], ss[idx])
        if gfit is None:
            continue
        g, _, _ = gfit
        if not np.isfinite(g):
            continue
        gammas.append(float(g))

    if not gammas:
        lo = hi = float(gamma)
    else:
        gg = np.asarray(gammas, dtype=np.float64)
        lo = float(np.quantile(gg, 0.025))
        hi = float(np.quantile(gg, 0.975))

    ci_width = float(hi - lo) if np.isfinite(lo) and np.isfinite(hi) else float("nan")
    passed = bool(
        np.isfinite(gamma)
        and np.isfinite(lo)
        and np.isfinite(hi)
        and (n_pts >= min_pts)
        and np.isfinite(ci_width)
        and (ci_width <= ci_width_max)
    )
    return CracklingFitDiagnostics(
        gamma=float(gamma),
        ci_low=float(lo),
        ci_high=float(hi),
        ci_width=float(ci_width),
        n_duration_points=int(n_pts),
        n_avalanches_used=int(ds.size),
        r2=float(r2),
        passed=passed,
    )


def crackling_fit(comps: list[ComponentStats], *, seed: int) -> tuple[float, float, float]:
    """
    Crackling relation per spec/09:
    fit log E[S|D] ~ gamma log D over D in [CRACKLING_D_RANGE_MIN, CRACKLING_D_RANGE_MAX]
    with at least CRACKLING_MIN_POINTS duration points. Bootstrap CI uses BOOTSTRAP_REPS.

    Returns (gamma, ci_low, ci_high); values are NaN if not available.
    """
    d = crackling_fit_diagnostics(comps, seed=seed)
    return (float(d.gamma), float(d.ci_low), float(d.ci_high))
