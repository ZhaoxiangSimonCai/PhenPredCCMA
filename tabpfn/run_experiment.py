#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

import numpy as np
import pandas as pd
import torch


def import_installed_tabpfn_regressor():
    excluded = {SCRIPT_DIR.resolve(), ROOT.resolve()}
    original_sys_path = list(sys.path)
    try:
        sys.path = [
            entry
            for entry in original_sys_path
            if Path(entry or ".").resolve() not in excluded
        ]
        from tabpfn import TabPFNRegressor as installed_regressor
    finally:
        sys.path = original_sys_path
    return installed_regressor


TabPFNRegressor = import_installed_tabpfn_regressor()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from feature_selection import FeatureSelector, fit_feature_selector


RAW_VIEW_FILE_STEMS = {
    "transcriptomics": "transcriptomics",
    "methylation": "methylation",
    "drugresponse": "drugresponse",
    "crisprcas9": "crisprcas9",
    "mutations": "mutations_binary",
}
RAW_ROWS_ARE_SAMPLES = {"crisprcas9"}
CONTINUOUS_BLOCKS = {"transcriptomics", "methylation", "drugresponse", "crisprcas9"}
VARIANT_ORDER = ["original", "mosa_nan_only", "mosa_all"]
SAMPLE_FRAME_ORDER = ["overlap", "expanded"]
PREDICTOR_BLOCKS = {
    "crisprcas9": ["transcriptomics", "methylation", "drugresponse", "mutations"],
    "drugresponse": ["transcriptomics", "methylation", "crisprcas9", "mutations"],
}


@dataclass
class RunConfig:
    ccma_dir: str
    mosa_files_dir: str
    out_dir: str
    mosa_timestamp: str
    target_family: str
    sample_frame: str
    variant: str
    seed: int
    target_limit: int
    max_features: int
    min_features_per_modality: int
    device: str
    gpu_id: int
    tabpfn_n_estimators: int
    tabpfn_fit_mode: str
    tabpfn_inference_precision: str
    tabpfn_ignore_pretraining_limits: bool
    tabpfn_model_path: str
    tabpfn_n_preprocessing_jobs: int
    log_every_targets: int
    quiet: bool


@dataclass
class PreparedRunData:
    target_family: str
    sample_frame: str
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TabPFN CCMA-only comparisons for original and MOSA feature variants."
    )
    parser.add_argument(
        "--ccma-dir",
        type=str,
        default="data/clines/ccma_processed",
        help="Directory containing processed CCMA train/test matrices.",
    )
    parser.add_argument(
        "--mosa-files-dir",
        type=str,
        default="reports/vae/files",
        help="Directory containing MOSA train/test outputs.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="reports/tabpfn",
        help="Base output directory. Per-run outputs are nested under <out-dir>/<timestamp>/...",
    )
    parser.add_argument(
        "--mosa-timestamp",
        type=str,
        default="20260313_162348",
        help="Timestamp prefix for MOSA artifacts in reports/vae/files.",
    )
    parser.add_argument(
        "--target-family",
        type=str,
        required=True,
        choices=["crisprcas9", "drugresponse"],
    )
    parser.add_argument(
        "--sample-frame",
        type=str,
        default="both",
        choices=["overlap", "expanded", "both"],
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["original", "mosa_nan_only", "mosa_all", "all"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-limit", type=int, default=0, help="0 means use all targets.")
    parser.add_argument("--max-features", type=int, default=2000)
    parser.add_argument("--min-features-per-modality", type=int, default=100)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--tabpfn-n-estimators", type=int, default=8)
    parser.add_argument(
        "--tabpfn-fit-mode",
        type=str,
        default="fit_preprocessors",
        choices=["low_memory", "fit_preprocessors", "fit_with_cache", "batched"],
    )
    parser.add_argument("--tabpfn-inference-precision", type=str, default="auto")
    parser.add_argument(
        "--tabpfn-model-path",
        type=str,
        default=default_model_path(),
    )
    parser.add_argument("--tabpfn-ignore-pretraining-limits", action="store_true")
    parser.add_argument("--tabpfn-n-preprocessing-jobs", type=int, default=1)
    parser.add_argument("--log-every-targets", type=int, default=100)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def args_to_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        ccma_dir=args.ccma_dir,
        mosa_files_dir=args.mosa_files_dir,
        out_dir=args.out_dir,
        mosa_timestamp=args.mosa_timestamp,
        target_family=args.target_family,
        sample_frame=args.sample_frame,
        variant=args.variant,
        seed=int(args.seed),
        target_limit=int(args.target_limit),
        max_features=int(args.max_features),
        min_features_per_modality=int(args.min_features_per_modality),
        device=str(args.device),
        gpu_id=int(args.gpu_id),
        tabpfn_n_estimators=int(args.tabpfn_n_estimators),
        tabpfn_fit_mode=str(args.tabpfn_fit_mode),
        tabpfn_inference_precision=str(args.tabpfn_inference_precision),
        tabpfn_ignore_pretraining_limits=bool(args.tabpfn_ignore_pretraining_limits),
        tabpfn_model_path=str(args.tabpfn_model_path),
        tabpfn_n_preprocessing_jobs=int(args.tabpfn_n_preprocessing_jobs),
        log_every_targets=max(1, int(args.log_every_targets)),
        quiet=bool(args.quiet),
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_log(cfg: RunConfig, text: str) -> None:
    if not cfg.quiet:
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


def expand_sample_frames(raw: str) -> List[str]:
    if raw == "both":
        return SAMPLE_FRAME_ORDER
    return [raw]


def expand_variants(raw: str) -> List[str]:
    if raw == "all":
        return VARIANT_ORDER
    return [raw]


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


def load_target_labels(
    ccma_dir: Path,
    target_family: str,
    sample_frame: str,
    split: str,
) -> tuple[pd.DataFrame, Path]:
    frame_token = "overlap" if sample_frame == "overlap" else "mosa"
    path = ccma_dir / f"{target_family}_ccma_{frame_token}_{split}.csv"
    df = read_raw_view_matrix(path, target_family)
    return df, path


def load_predictor_view(
    *,
    ccma_dir: Path,
    mosa_dir: Path,
    mosa_timestamp: str,
    view_name: str,
    variant: str,
    sample_frame: str,
    split: str,
) -> tuple[pd.DataFrame, Path]:
    if view_name == "mutations" or variant == "original":
        frame_token = "overlap" if sample_frame == "overlap" else "mosa"
        file_stem = RAW_VIEW_FILE_STEMS[view_name]
        path = ccma_dir / f"{file_stem}_ccma_{frame_token}_{split}.csv"
        return read_raw_view_matrix(path, view_name if view_name != "mutations" else "mutations"), path

    suffix = "nans_only" if variant == "mosa_nan_only" else "all"
    path = mosa_dir / f"{mosa_timestamp}_imputed_{view_name}_{split}_{suffix}.csv.gz"
    return read_mosa_view_matrix(path), path


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


def ordered_intersection(reference_ids: Sequence[str], candidate_sets: Iterable[Sequence[str]]) -> List[str]:
    candidate_sets = [set(str(v) for v in seq) for seq in candidate_sets]
    ordered = []
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


def build_run_data(
    cfg: RunConfig,
    target_family: str,
    sample_frame: str,
    variant: str,
) -> PreparedRunData:
    ccma_dir = (ROOT / cfg.ccma_dir).resolve()
    mosa_dir = (ROOT / cfg.mosa_files_dir).resolve()

    predictor_blocks = PREDICTOR_BLOCKS[target_family]
    if target_family in predictor_blocks:
        raise AssertionError(f"Target modality '{target_family}' must not be used as a predictor.")

    y_train_df, y_train_path = load_target_labels(ccma_dir, target_family, sample_frame, "train")
    y_test_df, y_test_path = load_target_labels(ccma_dir, target_family, sample_frame, "test")
    mosa_target_names, mosa_target_path = load_mosa_target_names(
        mosa_dir,
        cfg.mosa_timestamp,
        target_family,
    )

    missing_train_targets = [name for name in mosa_target_names if name not in y_train_df.columns]
    missing_test_targets = [name for name in mosa_target_names if name not in y_test_df.columns]
    if missing_train_targets or missing_test_targets:
        raise AssertionError(
            f"MOSA target set for '{target_family}' does not match the raw CCMA labels. "
            f"Missing train={len(missing_train_targets)} test={len(missing_test_targets)}."
        )

    y_train_df = y_train_df.loc[:, mosa_target_names].copy()
    y_test_df = y_test_df.loc[:, mosa_target_names].copy()

    y_train_df = apply_target_limit(y_train_df, cfg.target_limit)
    y_test_df = y_test_df.loc[:, y_train_df.columns].copy()

    if not str(y_train_path).startswith(str(ccma_dir)) or not str(y_test_path).startswith(str(ccma_dir)):
        raise AssertionError("Target labels must come from raw processed CCMA files.")

    feature_train_df: Dict[str, pd.DataFrame] = {}
    feature_test_df: Dict[str, pd.DataFrame] = {}
    source_paths: Dict[str, str] = {
        "target_train": str(y_train_path),
        "target_test": str(y_test_path),
        "target_names_source": str(mosa_target_path),
    }

    non_mutation_views = [view for view in predictor_blocks if view != "mutations"]
    for view_name in predictor_blocks:
        train_df, train_path = load_predictor_view(
            ccma_dir=ccma_dir,
            mosa_dir=mosa_dir,
            mosa_timestamp=cfg.mosa_timestamp,
            view_name=view_name,
            variant=variant,
            sample_frame=sample_frame,
            split="train",
        )
        test_df, test_path = load_predictor_view(
            ccma_dir=ccma_dir,
            mosa_dir=mosa_dir,
            mosa_timestamp=cfg.mosa_timestamp,
            view_name=view_name,
            variant=variant,
            sample_frame=sample_frame,
            split="test",
        )
        feature_train_df[view_name] = train_df
        feature_test_df[view_name] = test_df
        source_paths[f"{view_name}_train"] = str(train_path)
        source_paths[f"{view_name}_test"] = str(test_path)

    target_train_ids = [str(v) for v in y_train_df.index.tolist()]
    target_test_ids = [str(v) for v in y_test_df.index.tolist()]

    if sample_frame == "overlap":
        train_sample_ids = target_train_ids
        test_sample_ids = target_test_ids
        for view_name in non_mutation_views:
            missing_train = sorted(set(train_sample_ids) - set(feature_train_df[view_name].index.astype(str)))
            missing_test = sorted(set(test_sample_ids) - set(feature_test_df[view_name].index.astype(str)))
            if missing_train or missing_test:
                raise AssertionError(
                    f"Overlap frame requires fixed sample IDs for variant '{variant}', "
                    f"but view '{view_name}' is missing train={len(missing_train)} test={len(missing_test)}."
                )
    elif variant == "original":
        train_sample_ids = target_train_ids
        test_sample_ids = target_test_ids
    else:
        train_sample_ids = ordered_intersection(
            target_train_ids,
            [feature_train_df[view_name].index.astype(str).tolist() for view_name in non_mutation_views],
        )
        test_sample_ids = ordered_intersection(
            target_test_ids,
            [feature_test_df[view_name].index.astype(str).tolist() for view_name in non_mutation_views],
        )

    if sample_frame == "overlap":
        if len(train_sample_ids) != 95 or len(test_sample_ids) != 24:
            raise AssertionError(
                f"Expected overlap frame size train/test = 95/24, got {len(train_sample_ids)}/{len(test_sample_ids)}."
            )
    elif len(test_sample_ids) != 24:
        raise AssertionError(
            f"Expected expanded frame test size 24, got {len(test_sample_ids)}."
        )

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

    return PreparedRunData(
        target_family=target_family,
        sample_frame=sample_frame,
        variant=variant,
        target_names=y_train_df.columns.astype(str).tolist(),
        train_sample_ids=np.asarray(train_sample_ids, dtype=object),
        test_sample_ids=np.asarray(test_sample_ids, dtype=object),
        y_train=y_train_df.to_numpy(dtype=np.float32),
        y_test=y_test_df.to_numpy(dtype=np.float32),
        y_train_mask=np.isfinite(y_train_df.to_numpy(dtype=np.float32)),
        y_test_mask=np.isfinite(y_test_df.to_numpy(dtype=np.float32)),
        train_blocks=aligned_train_blocks,
        test_blocks=aligned_test_blocks,
        feature_names_by_block=feature_names_by_block,
        source_paths=source_paths,
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


def subset_blocks(blocks: Mapping[str, np.ndarray], indices: np.ndarray) -> Dict[str, np.ndarray]:
    return {name: x[indices].astype(np.float32, copy=False) for name, x in blocks.items()}


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


def masked_r2_uniform_average(y_true: np.ndarray, y_pred: np.ndarray, y_mask: np.ndarray) -> float:
    scores = masked_r2_per_target(y_true, y_pred, y_mask)
    finite = np.isfinite(scores)
    if not finite.any():
        return float("nan")
    return float(np.nanmean(scores))


def make_tabpfn_regressor(cfg: RunConfig, seed: int, device_spec: str) -> TabPFNRegressor:
    return TabPFNRegressor(
        n_estimators=cfg.tabpfn_n_estimators,
        model_path=cfg.tabpfn_model_path,
        device=device_spec,
        ignore_pretraining_limits=cfg.tabpfn_ignore_pretraining_limits,
        inference_precision=cfg.tabpfn_inference_precision,
        fit_mode=cfg.tabpfn_fit_mode,
        random_state=seed,
        n_preprocessing_jobs=cfg.tabpfn_n_preprocessing_jobs,
    )


def fit_predict_per_target(
    cfg: RunConfig,
    *,
    label: str,
    target_names: Sequence[str],
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_mask: np.ndarray,
    eval_x: np.ndarray,
    device_spec: str,
    base_seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    n_eval = int(eval_x.shape[0])
    n_targets = int(train_y.shape[1])
    pred = np.full((n_eval, n_targets), np.nan, dtype=np.float32)
    rows: List[Dict[str, object]] = []

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
                }
            )
            continue

        fallback_value = float(np.mean(y_train_j))
        if y_train_j.size < 2 or float(np.nanstd(y_train_j)) <= 1e-12:
            pred[:, j] = fallback_value
            status = "constant_fallback"
        else:
            status = "fit"
            model_seed = int(base_seed + j * 17 + 1)
            model = make_tabpfn_regressor(cfg=cfg, seed=model_seed, device_spec=device_spec)
            try:
                model.fit(x_train_j, y_train_j)
                pred_j = np.asarray(model.predict(eval_x), dtype=np.float32).reshape(-1)
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
            }
        )

        if (j + 1) % cfg.log_every_targets == 0 or j + 1 == n_targets:
            maybe_log(cfg, f"[{label}] targets processed: {j + 1}/{n_targets}")

    return pred, pd.DataFrame(rows)


def write_selected_feature_artifacts(
    out_dir: Path,
    prepared: PreparedRunData,
    selector: FeatureSelector,
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


def save_split_indices(
    out_dir: Path,
    prepared: PreparedRunData,
) -> None:
    payload: Dict[str, np.ndarray] = {
        "train_sample_ids": prepared.train_sample_ids.astype(object),
        "test_sample_ids": prepared.test_sample_ids.astype(object),
    }
    np.savez_compressed(out_dir / "split_indices.npz", **payload)


def run_single_experiment(
    cfg: RunConfig,
    target_family: str,
    sample_frame: str,
    variant: str,
    device_spec: str,
) -> Path:
    prepared = build_run_data(cfg, target_family, sample_frame, variant)
    out_dir = (
        (ROOT / cfg.out_dir).resolve()
        / cfg.mosa_timestamp
        / target_family
        / sample_frame
        / variant
    )
    ensure_dir(out_dir)

    maybe_log(
        cfg,
        "[setup] "
        f"family={target_family} frame={sample_frame} variant={variant} "
        f"train_n={prepared.y_train.shape[0]} test_n={prepared.y_test.shape[0]} "
        f"targets={prepared.y_train.shape[1]}",
    )

    save_split_indices(out_dir, prepared)

    selector = fit_feature_selector(
        prepared.train_blocks,
        max_features=cfg.max_features,
        min_per_modality=cfg.min_features_per_modality,
    )
    preprocessor = fit_preprocessor(prepared.train_blocks)
    train_blocks = selector.transform_blocks(preprocessor.transform_blocks(prepared.train_blocks))
    test_blocks = selector.transform_blocks(preprocessor.transform_blocks(prepared.test_blocks))
    x_train = concat_blocks(train_blocks, PREDICTOR_BLOCKS[target_family])
    x_test = concat_blocks(test_blocks, PREDICTOR_BLOCKS[target_family])

    maybe_log(
        cfg,
        f"[final][{target_family}/{sample_frame}/{variant}] "
        f"train_n={x_train.shape[0]} test_n={x_test.shape[0]} "
        f"selected_features={selector.total_selected}",
    )

    pred_test, fit_diag = fit_predict_per_target(
        cfg,
        label=f"final_{target_family}_{sample_frame}_{variant}",
        target_names=prepared.target_names,
        train_x=x_train,
        train_y=prepared.y_train,
        train_mask=prepared.y_train_mask,
        eval_x=x_test,
        device_spec=device_spec,
        base_seed=cfg.seed + 30_000,
    )
    fit_diag.to_csv(out_dir / "target_fit_diagnostics.csv", index=False)

    test_r2_per_target = masked_r2_per_target(prepared.y_test, pred_test, prepared.y_test_mask)
    metrics_test_per_target = pd.DataFrame(
        {
            "target": prepared.target_names,
            "test_r2": test_r2_per_target.astype(np.float64),
            "valid_test_n": prepared.y_test_mask.astype(int).sum(axis=0).astype(int),
        }
    )
    metrics_test_per_target.to_csv(out_dir / "metrics_test_per_target.csv", index=False)

    pred_test_df = pd.DataFrame(pred_test, index=prepared.test_sample_ids, columns=prepared.target_names)
    pred_test_df.round(6).to_csv(out_dir / "pred_test.csv.gz", compression="gzip")

    test_r2 = masked_r2_uniform_average(prepared.y_test, pred_test, prepared.y_test_mask)
    metrics_payload = {
        "target_family": target_family,
        "sample_frame": sample_frame,
        "variant": variant,
        "mosa_timestamp": cfg.mosa_timestamp,
        "train_n": int(prepared.y_train.shape[0]),
        "test_n": int(prepared.y_test.shape[0]),
        "target_count": int(prepared.y_train.shape[1]),
        "selected_feature_count": int(selector.total_selected),
        "test_r2": float(test_r2),
    }
    with (out_dir / "metrics_test.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    config_payload = asdict(cfg)
    config_payload.update(
        {
            "resolved_ccma_dir": str((ROOT / cfg.ccma_dir).resolve()),
            "resolved_mosa_files_dir": str((ROOT / cfg.mosa_files_dir).resolve()),
            "resolved_out_dir": str(out_dir.resolve()),
            "predictor_blocks": PREDICTOR_BLOCKS[target_family],
            "source_paths": prepared.source_paths,
            "train_sample_ids": prepared.train_sample_ids.tolist(),
            "test_sample_ids": prepared.test_sample_ids.tolist(),
        }
    )
    with (out_dir / "config_used.json").open("w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)

    write_selected_feature_artifacts(out_dir, prepared, selector)
    maybe_log(
        cfg,
        f"[done][{target_family}/{sample_frame}/{variant}] "
        f"test_r2={metrics_payload['test_r2']:.6f}",
    )
    return out_dir


def main() -> None:
    args = parse_args()
    cfg = args_to_config(args)
    device_spec = resolve_device_spec(cfg.device, cfg.gpu_id)

    sample_frames = expand_sample_frames(cfg.sample_frame)
    variants = expand_variants(cfg.variant)
    maybe_log(
        cfg,
        f"[setup] device={device_spec} target_family={cfg.target_family} "
        f"sample_frames={sample_frames} variants={variants}",
    )

    for sample_frame in sample_frames:
        for variant in variants:
            run_single_experiment(
                cfg=cfg,
                target_family=cfg.target_family,
                sample_frame=sample_frame,
                variant=variant,
                device_spec=device_spec,
            )


if __name__ == "__main__":
    main()
