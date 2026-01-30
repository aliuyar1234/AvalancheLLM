from __future__ import annotations

import numpy as np

from avalanche_llm.canon import get_canon
from avalanche_llm.metrics.branching import pooled_branching
from avalanche_llm.raster.cc import connected_components


def test_toy_connected_components_and_branching() -> None:
    canon = get_canon()
    adj_4n = str(canon["ENUM"]["ADJACENCY_ID"]["ADJ_4N"])

    # Toy raster from spec/16: nodes at (t=2,l=2),(t=3,l=2),(t=3,l=3) in 1-based indexing.
    # Implementation uses zero-based indices and expects [L, T].
    L, T = 3, 4
    x = np.zeros((L, T), dtype=np.uint8)
    x[1, 1] = 1
    x[1, 2] = 1
    x[2, 2] = 1

    comps = connected_components(x, adjacency_id=adj_4n)
    assert len(comps) == 1
    c = comps[0]
    assert c.size == 3
    assert c.span_tokens == 2
    assert c.span_layers == 2

    b = pooled_branching(x)
    assert np.isclose(b.b_time, 1.0 / 3.0)
    assert np.isclose(b.b_depth, 1.0 / 3.0)

