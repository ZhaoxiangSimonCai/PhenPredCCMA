from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence
import warnings

import numpy as np


@dataclass
class FeatureSelector:
    indices_by_block: Dict[str, np.ndarray]
    max_features: int
    min_per_modality: int
    strategy: str = "nan_variance_by_block"

    @property
    def total_selected(self) -> int:
        return int(sum(int(v.size) for v in self.indices_by_block.values()))

    def transform_blocks(
        self,
        blocks: Mapping[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        transformed: Dict[str, np.ndarray] = {}
        for block_name, x in blocks.items():
            idx = self.indices_by_block.get(block_name, np.zeros((0,), dtype=np.int64))
            transformed[block_name] = x[:, idx].astype(np.float32, copy=False)
        return transformed

    def selected_names(
        self,
        feature_names_by_block: Mapping[str, Sequence[str]],
    ) -> Dict[str, List[str]]:
        selected: Dict[str, List[str]] = {}
        for block_name, names in feature_names_by_block.items():
            idx = self.indices_by_block.get(block_name, np.zeros((0,), dtype=np.int64))
            selected[block_name] = [str(names[i]) for i in idx.tolist()]
        return selected


def _nan_variance_score(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.zeros((0,), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(invalid="ignore"):
            score = np.nanvar(x.astype(np.float64, copy=False), axis=0)
    return np.where(np.isfinite(score), score, -np.inf)


def _rank_top_k(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or scores.size == 0:
        return np.zeros((0,), dtype=np.int64)
    k = min(int(k), int(scores.shape[0]))
    if k <= 0:
        return np.zeros((0,), dtype=np.int64)
    idx = np.arange(scores.shape[0], dtype=np.int64)
    order = np.lexsort((idx, -scores))
    return order[:k].astype(np.int64)


def _allocate_quotas(
    max_features: int,
    n_features_by_block: Mapping[str, int],
    min_per_modality: int,
) -> Dict[str, int]:
    counts = {k: int(v) for k, v in n_features_by_block.items()}
    total = int(sum(counts.values()))
    if max_features >= total:
        return counts

    active = [name for name, n in counts.items() if n > 0]
    if not active:
        return {name: 0 for name in counts}

    quotas = {name: 0 for name in counts}
    min_floor = {name: min(int(min_per_modality), counts[name]) for name in active}

    floor_sum = sum(min_floor.values())
    if floor_sum > max_features:
        remaining = int(max_features)
        order = sorted(active, key=lambda name: counts[name], reverse=True)
        while remaining > 0:
            updated = False
            for name in order:
                if quotas[name] >= counts[name]:
                    continue
                quotas[name] += 1
                remaining -= 1
                updated = True
                if remaining == 0:
                    break
            if not updated:
                break
        return quotas

    for name, floor in min_floor.items():
        quotas[name] = floor

    remaining = int(max_features - floor_sum)
    if remaining <= 0:
        return quotas

    capacities = {name: counts[name] - quotas[name] for name in active}
    cap_sum = sum(max(0, cap) for cap in capacities.values())
    if cap_sum <= 0:
        return quotas

    for name in active:
        cap = capacities[name]
        if cap <= 0:
            continue
        share = int(round(remaining * cap / cap_sum))
        quotas[name] += min(share, cap)

    used = sum(quotas.values())
    if used > max_features:
        over = int(used - max_features)
        order = sorted(active, key=lambda name: quotas[name], reverse=True)
        i = 0
        while over > 0 and order:
            name = order[i % len(order)]
            if quotas[name] > min_floor[name]:
                quotas[name] -= 1
                over -= 1
            i += 1
    elif used < max_features:
        under = int(max_features - used)
        order = sorted(active, key=lambda name: capacities[name], reverse=True)
        i = 0
        while under > 0 and order:
            name = order[i % len(order)]
            if quotas[name] < counts[name]:
                quotas[name] += 1
                under -= 1
            i += 1

    return quotas


def fit_feature_selector(
    blocks: Mapping[str, np.ndarray],
    *,
    max_features: int = 2000,
    min_per_modality: int = 100,
) -> FeatureSelector:
    if max_features <= 0:
        raise ValueError(f"max_features must be > 0, got {max_features}")

    quotas = _allocate_quotas(
        max_features=max_features,
        n_features_by_block={name: int(x.shape[1]) for name, x in blocks.items()},
        min_per_modality=min_per_modality,
    )

    indices_by_block: Dict[str, np.ndarray] = {}
    for block_name, x in blocks.items():
        scores = _nan_variance_score(x)
        indices_by_block[block_name] = _rank_top_k(scores, quotas.get(block_name, 0))

    return FeatureSelector(
        indices_by_block=indices_by_block,
        max_features=max_features,
        min_per_modality=min_per_modality,
    )


def fit_variance_only_selector(
    blocks: Mapping[str, np.ndarray],
    *,
    variance_cutoff: float,
) -> FeatureSelector:
    """Per-block variance prefilter without quotas, matching cdsr_models' `vc` step.

    Keeps every column with NaN-aware variance > `variance_cutoff`; no top-K
    truncation and no modality balancing. Intended as the prefilter stage of the
    `per_target_corr` selection mode.
    """
    indices_by_block: Dict[str, np.ndarray] = {}
    for block_name, x in blocks.items():
        scores = _nan_variance_score(x)
        kept = np.flatnonzero(scores > float(variance_cutoff)).astype(np.int64)
        indices_by_block[block_name] = kept

    total = int(sum(int(v.size) for v in indices_by_block.values()))
    return FeatureSelector(
        indices_by_block=indices_by_block,
        max_features=total,
        min_per_modality=0,
        strategy=f"variance_only(vc={float(variance_cutoff):g})",
    )


def select_per_target_corr_indices(
    x_train: np.ndarray,
    y_train: np.ndarray,
    top_n: int,
) -> np.ndarray:
    """Top-N feature indices by |Pearson correlation| with y on training rows.

    Matches the per-fold step in `cdsr_models::random_forest`
    (`rank(-abs(cor(X_train, y))) <= n`). Inputs are assumed to be finite and
    aligned to the same rows; the caller filters NaNs upstream. Ties are broken
    by feature index (lexsort), so output is deterministic.
    """
    if x_train.ndim != 2:
        raise ValueError(f"x_train must be 2D, got shape {x_train.shape}")
    if y_train.ndim != 1:
        raise ValueError(f"y_train must be 1D, got shape {y_train.shape}")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"x_train rows ({x_train.shape[0]}) must match y_train length ({y_train.shape[0]})"
        )

    n_rows, n_cols = int(x_train.shape[0]), int(x_train.shape[1])
    k = max(0, min(int(top_n), n_cols))
    if k == 0 or n_rows < 2 or n_cols == 0:
        return np.zeros((0,), dtype=np.int64)

    x = x_train.astype(np.float64, copy=False)
    y = y_train.astype(np.float64, copy=False)

    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - float(y.mean())

    num = x_centered.T @ y_centered
    x_ss = np.einsum("ij,ij->j", x_centered, x_centered)
    y_ss = float(np.dot(y_centered, y_centered))
    denom = np.sqrt(x_ss * y_ss)

    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(denom > 1e-12, num / denom, 0.0)
    abs_r = np.where(np.isfinite(r), np.abs(r), -np.inf)

    idx = np.arange(abs_r.shape[0], dtype=np.int64)
    order = np.lexsort((idx, -abs_r))
    return order[:k].astype(np.int64)
