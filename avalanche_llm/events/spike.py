from __future__ import annotations

import numpy as np

from ..canon import get_canon

def v_from_z(z: np.ndarray, spike_def_id: str) -> np.ndarray:
    canon = get_canon()
    one_sided = str(canon["ENUM"]["SPIKE_DEF_ID"]["SPIKE_ONE_SIDED_POS"])
    two_sided = str(canon["ENUM"]["SPIKE_DEF_ID"]["SPIKE_TWO_SIDED_ABS"])
    if spike_def_id == one_sided:
        return z
    if spike_def_id == two_sided:
        return np.abs(z)
    raise ValueError(f"Unknown spike_def_id: {spike_def_id}")


def spikes(v: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """
    v: [..., units]
    tau: broadcastable thresholds
    """
    return (v > tau).astype(np.uint8)
