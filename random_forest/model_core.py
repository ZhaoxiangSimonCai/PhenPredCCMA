#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def parse_rf_max_features(raw: str):
    value = str(raw).strip().lower()
    if value in {"sqrt", "log2"}:
        return value
    if value in {"all", "1.0", "1"}:
        return 1.0
    try:
        return float(value)
    except ValueError as exc:  # noqa: BLE001
        raise ValueError(
            f"Unsupported --rf-max-features value: {raw}. Use sqrt, log2, all, or a float."
        ) from exc


def make_random_forest_regressor(cfg: Any, seed: int) -> RandomForestRegressor:
    max_depth = None if int(cfg.rf_max_depth) <= 0 else int(cfg.rf_max_depth)
    max_features = parse_rf_max_features(cfg.rf_max_features)
    return RandomForestRegressor(
        n_estimators=int(cfg.rf_n_estimators),
        max_depth=max_depth,
        min_samples_leaf=int(cfg.rf_min_samples_leaf),
        max_features=max_features,
        n_jobs=int(cfg.rf_n_jobs),
        random_state=int(seed),
    )


def fit_predict_per_target_rf(
    cfg: Any,
    *,
    label: str,
    target_names: Sequence[str],
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_mask: np.ndarray,
    eval_x: np.ndarray,
    log_fn,
    corr_top_n: int | None = None,
    feature_names: Sequence[str] | None = None,
    feature_blocks: Sequence[str] | None = None,
) -> tuple[np.ndarray, List[Dict[str, object]], List[Dict[str, object]]]:
    n_eval = int(eval_x.shape[0])
    n_targets = int(train_y.shape[1])
    pred = np.full((n_eval, n_targets), np.nan, dtype=np.float32)
    rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []

    use_corr = corr_top_n is not None and int(corr_top_n) > 0
    if use_corr:
        from feature_selection import select_per_target_corr_indices  # type: ignore
        n_feat = int(train_x.shape[1])
        if feature_names is None:
            feature_names_arr = np.array([f"f{i}" for i in range(n_feat)], dtype=object)
        else:
            feature_names_arr = np.asarray(feature_names, dtype=object)
        if feature_blocks is None:
            feature_blocks_arr = np.array(["unknown"] * n_feat, dtype=object)
        else:
            feature_blocks_arr = np.asarray(feature_blocks, dtype=object)
        if feature_names_arr.shape[0] != n_feat or feature_blocks_arr.shape[0] != n_feat:
            raise AssertionError(
                "feature_names / feature_blocks length must match train_x.shape[1] "
                f"(names={feature_names_arr.shape[0]}, blocks={feature_blocks_arr.shape[0]}, "
                f"features={n_feat})"
            )

    for j in range(n_targets):
        valid = train_mask[:, j].astype(bool)
        x_train_j = train_x[valid]
        y_train_j = train_y[valid, j].astype(np.float32)

        finite = np.isfinite(y_train_j)
        x_train_j = x_train_j[finite]
        y_train_j = y_train_j[finite]

        if y_train_j.size == 0:
            rows.append(
                {
                    "target": target_names[j],
                    "status": "no_train_labels",
                    "n_train_total": 0,
                    "fallback_value": np.nan,
                    "n_features_used": 0,
                }
            )
            continue

        fallback_value = float(np.mean(y_train_j))
        if y_train_j.size < 2 or float(np.nanstd(y_train_j)) <= 1e-12:
            pred[:, j] = fallback_value
            status = "constant_fallback"
            n_features_used = 0
        else:
            status = "fit"
            if use_corr:
                idx_j = select_per_target_corr_indices(x_train_j, y_train_j, int(corr_top_n))
                if idx_j.size == 0:
                    pred[:, j] = fallback_value
                    status = "no_features_selected"
                    n_features_used = 0
                    rows.append(
                        {
                            "target": target_names[j],
                            "status": status,
                            "n_train_total": int(y_train_j.shape[0]),
                            "fallback_value": fallback_value,
                            "n_features_used": n_features_used,
                        }
                    )
                    if (j + 1) % cfg.log_every_targets == 0 or j + 1 == n_targets:
                        log_fn(f"[{label}] targets processed: {j + 1}/{n_targets}")
                    continue
                x_train_j = x_train_j[:, idx_j]
                eval_x_j = eval_x[:, idx_j]
                names_j = feature_names_arr[idx_j]
                blocks_j = feature_blocks_arr[idx_j]
                target_name = str(target_names[j])
                for rank, (name, block) in enumerate(zip(names_j, blocks_j), start=1):
                    selected_rows.append(
                        {
                            "target": target_name,
                            "rank": rank,
                            "block": str(block),
                            "feature": str(name),
                        }
                    )
                n_features_used = int(idx_j.shape[0])
            else:
                eval_x_j = eval_x
                n_features_used = int(x_train_j.shape[1])

            model_seed = int(cfg.seed + j * 17 + 1)
            model = make_random_forest_regressor(cfg=cfg, seed=model_seed)
            try:
                model.fit(x_train_j, y_train_j)
                pred_j = np.asarray(model.predict(eval_x_j), dtype=np.float32).reshape(-1)
                if pred_j.shape[0] != n_eval:
                    raise RuntimeError(
                        f"Predict output length mismatch for target {target_names[j]}: "
                        f"expected {n_eval}, got {pred_j.shape[0]}"
                    )
                pred_j = np.where(np.isfinite(pred_j), pred_j, fallback_value).astype(np.float32)
                pred[:, j] = pred_j
            except Exception as exc:  # noqa: BLE001
                pred[:, j] = fallback_value
                status = f"fit_error:{type(exc).__name__}"
                log_fn(
                    f"[{label}] target={target_names[j]} fallback due to {type(exc).__name__}: {exc}"
                )

        rows.append(
            {
                "target": target_names[j],
                "status": status,
                "n_train_total": int(y_train_j.shape[0]),
                "fallback_value": fallback_value,
                "n_features_used": int(n_features_used),
            }
        )

        if (j + 1) % cfg.log_every_targets == 0 or j + 1 == n_targets:
            log_fn(f"[{label}] targets processed: {j + 1}/{n_targets}")

    return pred, rows, selected_rows
