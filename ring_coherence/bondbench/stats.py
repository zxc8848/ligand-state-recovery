from __future__ import annotations

import numpy as np


def bootstrap_binary_ci(
    values: list[bool], replicates: int = 1000, seed: int = 20260822
) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(replicates, len(array)))
    sampled = array[indices].mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(sampled, 0.025)),
        float(np.quantile(sampled, 0.975)),
    )


def bootstrap_mean_ci(
    values: list[float], replicates: int = 1000, seed: int = 20260822
) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(replicates, len(array)))
    sampled = array[indices].mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(sampled, 0.025)),
        float(np.quantile(sampled, 0.975)),
    )
