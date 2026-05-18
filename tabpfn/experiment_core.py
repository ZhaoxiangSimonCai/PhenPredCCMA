#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

import numpy as np
import pandas as pd
import torch


def import_installed_tabpfn_regressors():
    excluded = {SCRIPT_DIR.resolve(), ROOT.resolve()}
    original_sys_path = list(sys.path)
    try:
        sys.path = [
            entry
            for entry in original_sys_path
            if Path(entry or ".").resolve() not in excluded
        ]
        from tabpfn import TabPFNRegressor as installed_regressor
        from tabpfn.finetuning import FinetunedTabPFNRegressor as installed_finetuned_regressor
    finally:
        sys.path = original_sys_path
    return installed_regressor, installed_finetuned_regressor


TabPFNRegressor, FinetunedTabPFNRegressor = import_installed_tabpfn_regressors()

RAW_VIEW_FILE_STEMS = {
    "transcriptomics": "transcriptomics",
    "methylation": "methylation",
    "drugresponse": "drugresponse",
    "crisprcas9": "crisprcas9",
    "mutations": "mutations_binary",
    "copynumber": "copynumber",
    "clinical": "clinical",
}
RAW_ROWS_ARE_SAMPLES = {"crisprcas9", "copynumber", "clinical"}
CONTINUOUS_BLOCKS = {"transcriptomics", "methylation", "drugresponse", "crisprcas9"}
PREDICTOR_BLOCKS = {
    "crisprcas9":   ["transcriptomics", "methylation", "copynumber", "mutations", "clinical"],
    "drugresponse": ["transcriptomics", "methylation", "copynumber", "mutations", "clinical"],
}
NON_MOSA_IMPUTED_VIEWS = {"mutations", "clinical"}
SUMMARY_METRICS = ["test_r2", "test_pearsonr", "test_rmse"]


@dataclass
class PreparedRunData:
    target_family: str
    variant: str
    target_names: List[str]
    train_sample_ids: np.ndarray
    test_sample_ids: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    y_train_mask: np.ndarray
    y_test_mask: np.ndarray
    train_blocks: Dict[str, np.ndarray]
    test_blocks: Dict[str, np.ndarray]
    feature_names_by_block: Dict[str, List[str]]
    source_paths: Dict[str, str]
    metadata: Dict[str, Any]


@dataclass
class BlockPreprocessor:
    fill_values: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    standardize: bool


@dataclass
class MultiBlockPreprocessor:
    blocks: Dict[str, BlockPreprocessor]

    def transform_blocks(self, blocks: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        transformed: Dict[str, np.ndarray] = {}
        for block_name, x in blocks.items():
            params = self.blocks[block_name]
            x = x.astype(np.float32, copy=False)
            filled = np.where(np.isfinite(x), x, params.fill_values).astype(np.float32)
            if params.standardize:
                filled = ((filled - params.center) / params.scale).astype(np.float32)
            transformed[block_name] = filled
        return transformed


def default_model_path() -> str:
    reference_model = Path(
        "/home/scai/scratch/PredCRISPRCCMA/tabpfn/models/tabpfn-v2.5-regressor-v2.5_default.ckpt"
    )
    return str(reference_model) if reference_model.exists() else "auto"


def normalize_model_path_arg(raw_value: Any) -> str | List[str]:
    if isinstance(raw_value, (list, tuple)):
        values = list(raw_value)
    else:
        values = [raw_value]

    normalized: List[str] = []
    for value in values:
        if value is None:
            continue
        parts = [part.strip() for part in str(value).split(",") if part.strip()]
        normalized.extend(parts)

    if not normalized:
        return default_model_path()
    return normalized[0] if len(normalized) == 1 else normalized


def model_path_display_name(model_path: Any) -> str:
    paths = model_path if isinstance(model_path, list) else [model_path]
    return ",".join(Path(str(path)).name for path in paths)


def model_path_count(model_path: Any) -> int:
    return len(model_path) if isinstance(model_path, list) else 1


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_log(cfg: Any, text: str) -> None:
    if not getattr(cfg, "quiet", False):
        print(text, flush=True)


def resolve_device_spec(device: str, gpu_id: int) -> str:
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False.")
        return f"cuda:{gpu_id}"
    if torch.cuda.is_available():
        return f"cuda:{gpu_id}"
    return "cpu"


def read_csv_numeric(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df.apply(pd.to_numeric, errors="coerce")


def read_raw_view_matrix(path: Path, view_name: str) -> pd.DataFrame:
    df = read_csv_numeric(path)
    if view_name not in RAW_ROWS_ARE_SAMPLES:
        df = df.T
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def read_mosa_view_matrix(path: Path) -> pd.DataFrame:
    df = read_csv_numeric(path)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def load_mosa_target_names(
    mosa_dir: Path,
    mosa_timestamp: str,
    target_family: str,
) -> tuple[List[str], Path]:
    path = mosa_dir / f"{mosa_timestamp}_imputed_{target_family}_test_all.csv.gz"
    df = read_mosa_view_matrix(path)
    return df.columns.astype(str).tolist(), path


def apply_target_limit(df: pd.DataFrame, target_limit: int) -> pd.DataFrame:
    if target_limit <= 0 or target_limit >= df.shape[1]:
        return df
    return df.iloc[:, :target_limit].copy()


def ordered_intersection(reference_ids: Sequence[str], candidate_sets: Sequence[Sequence[str]]) -> List[str]:
    candidate_sets = [set(str(v) for v in seq) for seq in candidate_sets]
    ordered: List[str] = []
    for sample_id in reference_ids:
        sample_id = str(sample_id)
        if all(sample_id in values for values in candidate_sets):
            ordered.append(sample_id)
    return ordered


def align_feature_df(df: pd.DataFrame, sample_ids: Sequence[str]) -> pd.DataFrame:
    sample_ids = [str(v) for v in sample_ids]
    out = df.copy()
    out.index = out.index.astype(str)
    out.columns = out.columns.astype(str)
    return out.reindex(index=sample_ids)


def assert_feature_frame_is_sample_by_feature(df: pd.DataFrame, sample_ids: Sequence[str], view_name: str) -> None:
    if df.shape[0] != len(sample_ids):
        raise AssertionError(
            f"View '{view_name}' is not aligned as sample-by-feature. "
            f"Expected {len(sample_ids)} rows, got {df.shape[0]}."
        )


def build_prepared_run_data(
    *,
    target_family: str,
    variant: str,
    train_sample_ids: Sequence[str],
    test_sample_ids: Sequence[str],
    y_train_df: pd.DataFrame,
    y_test_df: pd.DataFrame,
    feature_train_df: Mapping[str, pd.DataFrame],
    feature_test_df: Mapping[str, pd.DataFrame],
    predictor_blocks: Sequence[str],
    source_paths: Dict[str, str],
    metadata: Dict[str, Any],
) -> PreparedRunData:
    train_sample_ids = [str(v) for v in train_sample_ids]
    test_sample_ids = [str(v) for v in test_sample_ids]

    y_train_df = y_train_df.reindex(index=train_sample_ids)
    y_test_df = y_test_df.reindex(index=test_sample_ids)

    aligned_train_blocks: Dict[str, np.ndarray] = {}
    aligned_test_blocks: Dict[str, np.ndarray] = {}
    feature_names_by_block: Dict[str, List[str]] = {}

    for view_name in predictor_blocks:
        aligned_train_df = align_feature_df(feature_train_df[view_name], train_sample_ids)
        aligned_test_df = align_feature_df(feature_test_df[view_name], test_sample_ids)
        assert_feature_frame_is_sample_by_feature(aligned_train_df, train_sample_ids, view_name)
        assert_feature_frame_is_sample_by_feature(aligned_test_df, test_sample_ids, view_name)
        feature_names_by_block[view_name] = aligned_train_df.columns.astype(str).tolist()
        if feature_names_by_block[view_name] != aligned_test_df.columns.astype(str).tolist():
            raise AssertionError(f"Feature mismatch between train/test for view '{view_name}'.")
        aligned_train_blocks[view_name] = aligned_train_df.to_numpy(dtype=np.float32)
        aligned_test_blocks[view_name] = aligned_test_df.to_numpy(dtype=np.float32)

    y_train = y_train_df.to_numpy(dtype=np.float32)
    y_test = y_test_df.to_numpy(dtype=np.float32)

    return PreparedRunData(
        target_family=target_family,
        variant=variant,
        target_names=y_train_df.columns.astype(str).tolist(),
        train_sample_ids=np.asarray(train_sample_ids, dtype=object),
        test_sample_ids=np.asarray(test_sample_ids, dtype=object),
        y_train=y_train,
        y_test=y_test,
        y_train_mask=np.isfinite(y_train),
        y_test_mask=np.isfinite(y_test),
        train_blocks=aligned_train_blocks,
        test_blocks=aligned_test_blocks,
        feature_names_by_block=feature_names_by_block,
        source_paths=source_paths,
        metadata=metadata,
    )


def fit_block_preprocessor(x_train: np.ndarray, *, standardize: bool) -> BlockPreprocessor:
    x_train = x_train.astype(np.float32, copy=False)
    with np.errstate(all="ignore"):
        fill_values = np.nanmedian(x_train, axis=0)
    fill_values = np.where(np.isfinite(fill_values), fill_values, 0.0).astype(np.float32)

    filled = np.where(np.isfinite(x_train), x_train, fill_values).astype(np.float32)
    if standardize:
        center = filled.mean(axis=0).astype(np.float32)
        scale = filled.std(axis=0).astype(np.float32)
        scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0).astype(np.float32)
    else:
        center = np.zeros((x_train.shape[1],), dtype=np.float32)
        scale = np.ones((x_train.shape[1],), dtype=np.float32)

    return BlockPreprocessor(
        fill_values=fill_values,
        center=center,
        scale=scale,
        standardize=bool(standardize),
    )


def fit_preprocessor(blocks: Mapping[str, np.ndarray]) -> MultiBlockPreprocessor:
    return MultiBlockPreprocessor(
        blocks={
            block_name: fit_block_preprocessor(
                x_train,
                standardize=block_name in CONTINUOUS_BLOCKS,
            )
            for block_name, x_train in blocks.items()
        }
    )


def concat_blocks(blocks: Mapping[str, np.ndarray], block_order: Sequence[str]) -> np.ndarray:
    return np.concatenate([blocks[name] for name in block_order], axis=1).astype(np.float32)


def masked_r2_per_target(y_true: np.ndarray, y_pred: np.ndarray, y_mask: np.ndarray) -> np.ndarray:
    scores = np.full((y_true.shape[1],), np.nan, dtype=np.float64)
    for j in range(y_true.shape[1]):
        mask = y_mask[:, j].astype(bool)
        if int(mask.sum()) < 2:
            continue
        y_true_j = y_true[mask, j].astype(np.float64)
        y_pred_j = y_pred[mask, j].astype(np.float64)
        finite = np.isfinite(y_true_j) & np.isfinite(y_pred_j)
        if int(finite.sum()) < 2:
            continue
        y_true_j = y_true_j[finite]
        y_pred_j = y_pred_j[finite]
        denom = float(np.square(y_true_j - np.mean(y_true_j)).sum())
        if denom <= 0:
            continue
        numer = float(np.square(y_true_j - y_pred_j).sum())
        scores[j] = 1.0 - (numer / denom)
    return scores


def masked_pearsonr_per_target(y_true: np.ndarray, y_pred: np.ndarray, y_mask: np.ndarray) -> np.ndarray:
    scores = np.full((y_true.shape[1],), np.nan, dtype=np.float64)
    for j in range(y_true.shape[1]):
        mask = y_mask[:, j].astype(bool)
        if int(mask.sum()) < 2:
            continue
        y_true_j = y_true[mask, j].astype(np.float64)
        y_pred_j = y_pred[mask, j].astype(np.float64)
        finite = np.isfinite(y_true_j) & np.isfinite(y_pred_j)
        if int(finite.sum()) < 2:
            continue
        y_true_j = y_true_j[finite]
        y_pred_j = y_pred_j[finite]
        if float(np.std(y_true_j)) <= 1e-12 or float(np.std(y_pred_j)) <= 1e-12:
            continue
        scores[j] = float(np.corrcoef(y_true_j, y_pred_j)[0, 1])
    return scores


def masked_rmse_per_target(y_true: np.ndarray, y_pred: np.ndarray, y_mask: np.ndarray) -> np.ndarray:
    scores = np.full((y_true.shape[1],), np.nan, dtype=np.float64)
    for j in range(y_true.shape[1]):
        mask = y_mask[:, j].astype(bool)
        if int(mask.sum()) < 1:
            continue
        y_true_j = y_true[mask, j].astype(np.float64)
        y_pred_j = y_pred[mask, j].astype(np.float64)
        finite = np.isfinite(y_true_j) & np.isfinite(y_pred_j)
        if int(finite.sum()) < 1:
            continue
        residual = y_pred_j[finite] - y_true_j[finite]
        scores[j] = float(np.sqrt(np.mean(np.square(residual))))
    return scores


def nanmean_or_nan(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    return float(np.nanmean(values))


def pooled_observed_arrays(y_true: np.ndarray, y_pred: np.ndarray, y_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = y_mask.astype(bool) & np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask].astype(np.float64), y_pred[mask].astype(np.float64)


def pooled_pearsonr(y_true: np.ndarray, y_pred: np.ndarray, y_mask: np.ndarray) -> float:
    y_true_obs, y_pred_obs = pooled_observed_arrays(y_true, y_pred, y_mask)
    if y_true_obs.size < 2:
        return float("nan")
    if float(np.std(y_true_obs)) <= 1e-12 or float(np.std(y_pred_obs)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(y_true_obs, y_pred_obs)[0, 1])


def pooled_rmse(y_true: np.ndarray, y_pred: np.ndarray, y_mask: np.ndarray) -> float:
    y_true_obs, y_pred_obs = pooled_observed_arrays(y_true, y_pred, y_mask)
    if y_true_obs.size < 1:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(y_pred_obs - y_true_obs))))


def compute_test_metrics(prepared: PreparedRunData, pred_test: np.ndarray) -> tuple[Dict[str, Any], pd.DataFrame]:
    test_r2_per_target = masked_r2_per_target(prepared.y_test, pred_test, prepared.y_test_mask)
    test_pearsonr_per_target = masked_pearsonr_per_target(prepared.y_test, pred_test, prepared.y_test_mask)
    test_rmse_per_target = masked_rmse_per_target(prepared.y_test, pred_test, prepared.y_test_mask)

    per_target_df = pd.DataFrame(
        {
            "target": prepared.target_names,
            "test_r2": test_r2_per_target.astype(np.float64),
            "test_pearsonr": test_pearsonr_per_target.astype(np.float64),
            "test_rmse": test_rmse_per_target.astype(np.float64),
            "valid_test_n": prepared.y_test_mask.astype(int).sum(axis=0).astype(int),
        }
    )

    summary_metrics = {
        "test_r2": nanmean_or_nan(test_r2_per_target),
        "test_pearsonr": nanmean_or_nan(test_pearsonr_per_target),
        "test_rmse": nanmean_or_nan(test_rmse_per_target),
        "pooled_test_pearsonr": pooled_pearsonr(prepared.y_test, pred_test, prepared.y_test_mask),
        "pooled_test_rmse": pooled_rmse(prepared.y_test, pred_test, prepared.y_test_mask),
        "test_observation_count": int(prepared.y_test_mask.astype(int).sum()),
        "metric_aggregation": "macro_mean_over_targets",
    }
    return summary_metrics, per_target_df


def build_test_prediction_records(
    prepared: PreparedRunData,
    pred_test: np.ndarray,
    metadata: Mapping[str, Any],
) -> pd.DataFrame:
    n_samples, n_targets = pred_test.shape
    sample_ids = np.repeat(prepared.test_sample_ids.astype(str), n_targets)
    targets = np.tile(np.asarray(prepared.target_names, dtype=object), n_samples)
    y_true = prepared.y_test.astype(np.float64).reshape(-1)
    y_pred = pred_test.astype(np.float64).reshape(-1)
    observed = prepared.y_test_mask.astype(bool).reshape(-1)
    residual = np.where(observed, y_pred - y_true, np.nan)

    records = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "target": targets,
            "y_true": y_true,
            "y_pred": y_pred,
            "is_observed": observed.astype(int),
            "residual": residual,
            "abs_error": np.where(observed, np.abs(residual), np.nan),
            "squared_error": np.where(observed, np.square(residual), np.nan),
        }
    )
    for key, value in metadata.items():
        records[key] = value
    return records


def write_test_outputs(
    out_dir: Path,
    prepared: PreparedRunData,
    pred_test: np.ndarray,
    summary_metadata: Dict[str, Any],
) -> tuple[Dict[str, Any], pd.DataFrame]:
    summary_metrics, per_target_df = compute_test_metrics(prepared, pred_test)
    summary_payload = dict(summary_metadata)
    summary_payload.update(summary_metrics)

    per_target_df.to_csv(out_dir / "test_metrics_per_target.csv", index=False)

    pred_test_df = pd.DataFrame(pred_test, index=prepared.test_sample_ids, columns=prepared.target_names)
    pred_test_df.round(6).to_csv(out_dir / "test_predictions_wide.csv.gz", compression="gzip")

    truth_test_df = pd.DataFrame(prepared.y_test, index=prepared.test_sample_ids, columns=prepared.target_names)
    truth_test_df.round(6).to_csv(out_dir / "test_truth_wide.csv.gz", compression="gzip")

    prediction_records = build_test_prediction_records(prepared, pred_test, summary_metadata)
    prediction_records.to_csv(out_dir / "test_prediction_records.csv.gz", index=False, compression="gzip")

    with (out_dir / "test_metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    return summary_payload, per_target_df


def make_tabpfn_regressor(cfg: Any, seed: int, device_spec: str) -> Any:
    model_path = normalize_model_path_arg(getattr(cfg, "tabpfn_model_path", default_model_path()))

    if getattr(cfg, "tabpfn_estimator_mode", "standard") == "finetune":
        time_limit = int(getattr(cfg, "tabpfn_finetune_time_limit", 0))
        extra_regressor_kwargs = {
            "n_estimators": int(getattr(cfg, "tabpfn_n_estimators", 8)),
            "model_path": model_path,
            "device": device_spec,
            "ignore_pretraining_limits": bool(
                getattr(cfg, "tabpfn_ignore_pretraining_limits", False)
            ),
            "inference_precision": getattr(cfg, "tabpfn_inference_precision", "auto"),
            "n_preprocessing_jobs": int(getattr(cfg, "tabpfn_n_preprocessing_jobs", 1)),
        }
        return FinetunedTabPFNRegressor(
            device=device_spec,
            epochs=int(getattr(cfg, "tabpfn_finetune_epochs", 30)),
            time_limit=time_limit if time_limit > 0 else None,
            learning_rate=float(getattr(cfg, "tabpfn_finetune_learning_rate", 1e-5)),
            validation_split_ratio=float(
                getattr(cfg, "tabpfn_finetune_validation_split_ratio", 0.1)
            ),
            early_stopping_patience=int(
                getattr(cfg, "tabpfn_finetune_early_stopping_patience", 8)
            ),
            n_estimators_finetune=int(getattr(cfg, "tabpfn_finetune_n_estimators", 2)),
            n_estimators_validation=int(
                getattr(cfg, "tabpfn_finetune_n_estimators_validation", 2)
            ),
            n_estimators_final_inference=int(
                getattr(cfg, "tabpfn_finetune_n_estimators_final_inference", 8)
            ),
            random_state=seed,
            save_checkpoint_interval=None,
            extra_regressor_kwargs=extra_regressor_kwargs,
        )

    return TabPFNRegressor(
        n_estimators=int(getattr(cfg, "tabpfn_n_estimators", 8)),
        model_path=model_path,
        device=device_spec,
        ignore_pretraining_limits=bool(getattr(cfg, "tabpfn_ignore_pretraining_limits", False)),
        inference_precision=getattr(cfg, "tabpfn_inference_precision", "auto"),
        fit_mode=getattr(cfg, "tabpfn_fit_mode", "fit_preprocessors"),
        random_state=seed,
        n_preprocessing_jobs=int(getattr(cfg, "tabpfn_n_preprocessing_jobs", 1)),
    )


def fit_predict_per_target(
    cfg: Any,
    *,
    label: str,
    target_names: Sequence[str],
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_mask: np.ndarray,
    eval_x: np.ndarray,
    device_spec: str,
    base_seed: int,
    corr_top_n: int | None = None,
    feature_names: Sequence[str] | None = None,
    feature_blocks: Sequence[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
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
                        maybe_log(cfg, f"[{label}] targets processed: {j + 1}/{n_targets}")
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

            model_seed = int(base_seed + j * 17 + 1)
            model = make_tabpfn_regressor(cfg=cfg, seed=model_seed, device_spec=device_spec)
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
                maybe_log(
                    cfg,
                    f"[{label}] target={target_names[j]} fallback due to {type(exc).__name__}: {exc}",
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
            maybe_log(cfg, f"[{label}] targets processed: {j + 1}/{n_targets}")

    selected_df = pd.DataFrame(
        selected_rows, columns=["target", "rank", "block", "feature"]
    )
    return pred, pd.DataFrame(rows), selected_df


def write_selected_feature_artifacts(
    out_dir: Path,
    prepared: PreparedRunData,
    selector: Any,
) -> None:
    selected = selector.selected_names(prepared.feature_names_by_block)
    counts = {}
    for block_name, names in prepared.feature_names_by_block.items():
        counts[f"{block_name}_total"] = len(names)
        counts[f"{block_name}_selected"] = len(selected[block_name])

    payload = {
        "selection_strategy": selector.strategy,
        "max_features": selector.max_features,
        "min_features_per_modality": selector.min_per_modality,
        "counts": counts,
        "selected_features": selected,
        "target_names": prepared.target_names,
    }
    with (out_dir / "selected_features.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_split_indices(out_dir: Path, prepared: PreparedRunData) -> None:
    payload: Dict[str, np.ndarray] = {
        "train_sample_ids": prepared.train_sample_ids.astype(object),
        "test_sample_ids": prepared.test_sample_ids.astype(object),
    }
    np.savez_compressed(out_dir / "split_indices.npz", **payload)
