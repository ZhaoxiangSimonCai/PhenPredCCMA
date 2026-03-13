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
