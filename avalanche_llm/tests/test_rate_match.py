from __future__ import annotations

import numpy as np

from avalanche_llm.events.rate_match import verify_rate_match


def test_rate_match_histogram_tolerance() -> None:
    # Large deterministic sample to make histogram quantile stable within tolerance.
    n = 200_000
    v = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float64)
    target_rate = 0.1
    out = verify_rate_match(v_samples=v, target_rate=target_rate)
    assert abs(out.achieved_rate - target_rate) <= 5e-4

