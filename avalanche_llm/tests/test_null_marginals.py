from __future__ import annotations

import numpy as np

from avalanche_llm.raster.nulls import within_layer_time_permutation
from avalanche_llm.raster.nulls import within_layer_time_circular_shift


def test_within_layer_time_permutation_preserves_marginals() -> None:
    rng = np.random.default_rng(0)
    L, T = 5, 17
    a = rng.integers(0, 10, size=(L, T), dtype=np.uint16)
    out = within_layer_time_permutation(a, run_id="run_test", cond_id=0, seq_id=0)
    assert out.a_perm.shape == a.shape
    assert out.x_perm.shape == a.shape
    for l in range(L):
        assert np.array_equal(np.sort(a[l, :]), np.sort(out.a_perm[l, :]))
        assert np.array_equal((out.a_perm[l, :] > 0).astype(np.uint8), out.x_perm[l, :])


def test_within_layer_time_circular_shift_preserves_marginals() -> None:
    rng = np.random.default_rng(0)
    L, T = 5, 17
    a = rng.integers(0, 10, size=(L, T), dtype=np.uint16)
    out = within_layer_time_circular_shift(a, run_id="run_test", cond_id=0, seq_id=0)
    assert out.a_perm.shape == a.shape
    assert out.x_perm.shape == a.shape
    for l in range(L):
        assert np.array_equal(np.sort(a[l, :]), np.sort(out.a_perm[l, :]))
        assert np.array_equal((out.a_perm[l, :] > 0).astype(np.uint8), out.x_perm[l, :])
