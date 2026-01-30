from __future__ import annotations

import numpy as np

from ..raster.cc import ComponentStats, connected_components, connected_components_weighted


def avalanches_from_x(x: np.ndarray, *, adjacency_id: str) -> list[ComponentStats]:
    """
    x: occupancy array [L, T]
    """
    return connected_components(x, adjacency_id=adjacency_id)


def avalanches_from_a(a: np.ndarray, *, adjacency_id: str) -> list[ComponentStats]:
    """
    a: count array [L, T]; component size is sum of a over the component.
    """
    return connected_components_weighted(a, adjacency_id=adjacency_id)


def avalanche_size_stats(comps: list[ComponentStats]) -> dict[str, float]:
    if not comps:
        return {
            "n_avalanches": 0,
            "size_mean": 0.0,
            "size_median": 0.0,
            "span_tokens_mean": 0.0,
            "span_layers_mean": 0.0,
        }
    sizes = np.array([c.size for c in comps], dtype=np.float64)
    spans_t = np.array([c.span_tokens for c in comps], dtype=np.float64)
    spans_l = np.array([c.span_layers for c in comps], dtype=np.float64)
    return {
        "n_avalanches": int(len(comps)),
        "size_mean": float(np.mean(sizes)),
        "size_median": float(np.median(sizes)),
        "span_tokens_mean": float(np.mean(spans_t)),
        "span_layers_mean": float(np.mean(spans_l)),
    }
