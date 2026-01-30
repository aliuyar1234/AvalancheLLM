from __future__ import annotations

import numpy as np

from ..canon import get_canon


def standardize(u: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Standardize activations u using mu and sigma and CANON.CONST.EPS.
    """
    canon = get_canon()
    eps = float(canon["CONST"]["EPS"])
    sigma_safe = np.where(sigma < eps, eps, sigma)
    return (u - mu) / sigma_safe

