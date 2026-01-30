from __future__ import annotations

import numpy as np


def a_x_from_unit_spikes(unit_spikes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    unit_spikes: [..., units] uint8
    Returns:
      A: event counts [...], uint16
      X: occupancy [...], uint8
    """
    a = np.sum(unit_spikes.astype(np.uint16), axis=-1)
    x = (a > 0).astype(np.uint8)
    return a, x

