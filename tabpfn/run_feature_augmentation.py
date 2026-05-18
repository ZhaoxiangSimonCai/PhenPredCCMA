#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_core import (
    NON_MOSA_IMPUTED_VIEWS,
    PREDICTOR_BLOCKS,
    RAW_VIEW_FILE_STEMS,
    ROOT,
    PreparedRunData,
    apply_target_limit,
    build_prepared_run_data,
    concat_blocks,
    default_model_path,
    ensure_dir,
    fit_predict_per_target,
    fit_preprocessor,
    load_mosa_target_names,
    maybe_log,
    model_path_count,
    model_path_display_name,
    normalize_model_path_arg,
    ordered_intersection,
    read_mosa_view_matrix,
    read_raw_view_matrix,
    resolve_device_spec,
    save_split_indices,
    write_selected_feature_artifacts,
    write_test_outputs,
)
from feature_selection import fit_feature_selector, fit_variance_only_selector

import numpy as np


EXPERIMENT_NAME = "feature_augmentation"
VARIANT_ORDER = ["original", "mosa_nan_only", "mosa_all"]
SAMPLE_FRAME_ORDER = ["overlap", "expanded"]


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
    feature_selection_mode: str
    corr_top_n: int
    variance_cutoff: float
    device: str
    gpu_id: int
    tabpfn_n_estimators: int
    tabpfn_estimator_mode: str
    tabpfn_fit_mode: str
    tabpfn_inference_precision: str
    tabpfn_ignore_pretraining_limits: bool
    tabpfn_model_path: str | List[str]
    tabpfn_n_preprocessing_jobs: int
    tabpfn_finetune_epochs: int
    tabpfn_finetune_time_limit: int
    tabpfn_finetune_learning_rate: float
    tabpfn_finetune_validation_split_ratio: float
    tabpfn_finetune_early_stopping_patience: int
    tabpfn_finetune_n_estimators: int
    tabpfn_finetune_n_estimators_validation: int
    tabpfn_finetune_n_estimators_final_inference: int
    log_every_targets: int
    quiet: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TabPFN feature-augmentation comparisons for original and MOSA feature variants."
    )
    parser.add_argument("--ccma-dir", type=str, default="data/clines/ccma_processed")
    parser.add_argument("--mosa-files-dir", type=str, default="reports/vae/files")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="reports/tabpfn/feature_augmentation",
        help="Per-run outputs are written under <out-dir>/<timestamp>/<family>/<frame>/<variant>/",
    )
    parser.add_argument("--mosa-timestamp", type=str, default="20260313_162348")
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
        "--feature-selection-mode",
        type=str,
        default="block_variance",
        choices=["block_variance", "per_target_corr"],
        help=(
            "block_variance (default) keeps the existing per-block variance prefilter with "
            "per-modality quotas. per_target_corr matches cdsr_models::random_forest: "
            "variance cutoff per block, then top-N |corr(y)| features chosen per target "
            "inside the fit loop with no modality quotas."
        ),
    )
    parser.add_argument(
        "--corr-top-n",
        type=int,
        default=500,
        help="Top-N features per target by |Pearson r| with y (per_target_corr only).",
    )
    parser.add_argument(
        "--variance-cutoff",
        type=float,
        default=0.01,
        help="Variance prefilter cutoff applied per block (per_target_corr only).",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--tabpfn-n-estimators", type=int, default=8)
    parser.add_argument(
        "--tabpfn-estimator-mode",
        type=str,
        default="standard",
        choices=["standard", "finetune"],
        help="Use the standard TabPFNRegressor or the separate FinetunedTabPFNRegressor.",
    )
    parser.add_argument(
        "--tabpfn-fit-mode",
        type=str,
        default="fit_preprocessors",
        choices=["low_memory", "fit_preprocessors", "fit_with_cache", "batched"],
        help="Inference fit mode for standard TabPFN runs. Ignored in finetune mode.",
    )
    parser.add_argument("--tabpfn-inference-precision", type=str, default="auto")
    parser.add_argument(
        "--tabpfn-model-path",
        type=str,
        nargs="+",
        default=[default_model_path()],
        help="One or more TabPFN regressor checkpoints. Multiple values are passed through as a checkpoint ensemble.",
    )
    parser.add_argument("--tabpfn-ignore-pretraining-limits", action="store_true")
    parser.add_argument("--tabpfn-n-preprocessing-jobs", type=int, default=1)
    parser.add_argument("--tabpfn-finetune-epochs", type=int, default=30)
    parser.add_argument("--tabpfn-finetune-time-limit", type=int, default=0, help="0 disables the finetune time limit.")
    parser.add_argument("--tabpfn-finetune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--tabpfn-finetune-validation-split-ratio", type=float, default=0.1)
    parser.add_argument("--tabpfn-finetune-early-stopping-patience", type=int, default=8)
    parser.add_argument("--tabpfn-finetune-n-estimators", type=int, default=2)
    parser.add_argument("--tabpfn-finetune-n-estimators-validation", type=int, default=2)
    parser.add_argument("--tabpfn-finetune-n-estimators-final-inference", type=int, default=8)
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
        feature_selection_mode=str(args.feature_selection_mode),
        corr_top_n=int(args.corr_top_n),
        variance_cutoff=float(args.variance_cutoff),
        device=str(args.device),
        gpu_id=int(args.gpu_id),
        tabpfn_n_estimators=int(args.tabpfn_n_estimators),
        tabpfn_estimator_mode=str(args.tabpfn_estimator_mode),
        tabpfn_fit_mode=str(args.tabpfn_fit_mode),
        tabpfn_inference_precision=str(args.tabpfn_inference_precision),
        tabpfn_ignore_pretraining_limits=bool(args.tabpfn_ignore_pretraining_limits),
        tabpfn_model_path=normalize_model_path_arg(args.tabpfn_model_path),
        tabpfn_n_preprocessing_jobs=int(args.tabpfn_n_preprocessing_jobs),
        tabpfn_finetune_epochs=int(args.tabpfn_finetune_epochs),
        tabpfn_finetune_time_limit=int(args.tabpfn_finetune_time_limit),
        tabpfn_finetune_learning_rate=float(args.tabpfn_finetune_learning_rate),
        tabpfn_finetune_validation_split_ratio=float(args.tabpfn_finetune_validation_split_ratio),
        tabpfn_finetune_early_stopping_patience=int(args.tabpfn_finetune_early_stopping_patience),
        tabpfn_finetune_n_estimators=int(args.tabpfn_finetune_n_estimators),
        tabpfn_finetune_n_estimators_validation=int(args.tabpfn_finetune_n_estimators_validation),
        tabpfn_finetune_n_estimators_final_inference=int(args.tabpfn_finetune_n_estimators_final_inference),
        log_every_targets=max(1, int(args.log_every_targets)),
        quiet=bool(args.quiet),
    )


def expand_sample_frames(raw: str) -> List[str]:
    if raw == "both":
        return SAMPLE_FRAME_ORDER
    return [raw]


def expand_variants(raw: str) -> List[str]:
    if raw == "all":
        return VARIANT_ORDER
    return [raw]


def load_target_labels(
    ccma_dir: Path,
    target_family: str,
    sample_frame: str,
    split: str,
):
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
):
    if view_name in NON_MOSA_IMPUTED_VIEWS or variant == "original":
        frame_token = "overlap" if sample_frame == "overlap" else "mosa"
        file_stem = RAW_VIEW_FILE_STEMS[view_name]
        path = ccma_dir / f"{file_stem}_ccma_{frame_token}_{split}.csv"
        return read_raw_view_matrix(path, view_name), path

    suffix = "nans_only" if variant == "mosa_nan_only" else "all"
    path = mosa_dir / f"{mosa_timestamp}_imputed_{view_name}_{split}_{suffix}.csv.gz"
    return read_mosa_view_matrix(path), path


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

    feature_train_df: Dict[str, object] = {}
    feature_test_df: Dict[str, object] = {}
    source_paths: Dict[str, str] = {
        "target_train": str(y_train_path),
        "target_test": str(y_test_path),
        "target_names_source": str(mosa_target_path),
    }

    mosa_imputed_views = [view for view in predictor_blocks if view not in NON_MOSA_IMPUTED_VIEWS]
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
        for view_name in mosa_imputed_views:
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
            [feature_train_df[view_name].index.astype(str).tolist() for view_name in mosa_imputed_views],
        )
        test_sample_ids = ordered_intersection(
            target_test_ids,
            [feature_test_df[view_name].index.astype(str).tolist() for view_name in mosa_imputed_views],
        )

    if sample_frame == "overlap":
        if len(train_sample_ids) != 95 or len(test_sample_ids) != 24:
            raise AssertionError(
                f"Expected overlap frame size train/test = 95/24, got {len(train_sample_ids)}/{len(test_sample_ids)}."
            )
    elif len(test_sample_ids) != 24:
        raise AssertionError(f"Expected expanded frame test size 24, got {len(test_sample_ids)}.")

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "sample_frame": sample_frame,
        "variant": variant,
    }
    return build_prepared_run_data(
        target_family=target_family,
        variant=variant,
        train_sample_ids=train_sample_ids,
        test_sample_ids=test_sample_ids,
        y_train_df=y_train_df,
        y_test_df=y_test_df,
        feature_train_df=feature_train_df,
        feature_test_df=feature_test_df,
        predictor_blocks=predictor_blocks,
        source_paths=source_paths,
        metadata=metadata,
    )


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

    if cfg.feature_selection_mode == "per_target_corr":
        selector = fit_variance_only_selector(
            prepared.train_blocks,
            variance_cutoff=cfg.variance_cutoff,
        )
    else:
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

    feature_names_concat: List[str] = []
    feature_blocks_concat: List[str] = []
    for block_name in PREDICTOR_BLOCKS[target_family]:
        all_names = prepared.feature_names_by_block[block_name]
        idx = selector.indices_by_block.get(block_name, np.zeros((0,), dtype=np.int64))
        for i in idx.tolist():
            feature_names_concat.append(str(all_names[i]))
            feature_blocks_concat.append(block_name)
    if len(feature_names_concat) != x_train.shape[1]:
        raise AssertionError(
            f"Feature metadata length {len(feature_names_concat)} does not match "
            f"x_train.shape[1] {x_train.shape[1]} for "
            f"{target_family}/{sample_frame}/{variant}."
        )

    maybe_log(
        cfg,
        f"[final][{target_family}/{sample_frame}/{variant}] "
        f"train_n={x_train.shape[0]} test_n={x_test.shape[0]} "
        f"selected_features={selector.total_selected} "
        f"mode={cfg.feature_selection_mode}",
    )

    if cfg.feature_selection_mode == "per_target_corr":
        variance_summary = {
            "feature_selection_mode": "per_target_corr",
            "variance_cutoff": float(cfg.variance_cutoff),
            "corr_top_n": int(cfg.corr_top_n),
            "blocks": {
                block_name: {
                    "n_total": int(len(prepared.feature_names_by_block[block_name])),
                    "n_after_variance": int(
                        selector.indices_by_block.get(block_name, np.zeros((0,), dtype=np.int64)).size
                    ),
                }
                for block_name in PREDICTOR_BLOCKS[target_family]
            },
            "total_after_variance": int(selector.total_selected),
        }
        with (out_dir / "variance_filter_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(variance_summary, handle, indent=2)

    pred_test, fit_diag, selected_features_df = fit_predict_per_target(
        cfg,
        label=f"final_{target_family}_{sample_frame}_{variant}",
        target_names=prepared.target_names,
        train_x=x_train,
        train_y=prepared.y_train,
        train_mask=prepared.y_train_mask,
        eval_x=x_test,
        device_spec=device_spec,
        base_seed=cfg.seed + 30_000,
        corr_top_n=(cfg.corr_top_n if cfg.feature_selection_mode == "per_target_corr" else None),
        feature_names=feature_names_concat,
        feature_blocks=feature_blocks_concat,
    )
    fit_diag.to_csv(out_dir / "target_fit_diagnostics.csv", index=False)
    if cfg.feature_selection_mode == "per_target_corr":
        selected_features_df.to_csv(
            out_dir / "selected_features_per_target.csv.gz",
            index=False,
            compression="gzip",
        )

    summary_payload, _ = write_test_outputs(
        out_dir,
        prepared,
        pred_test,
        summary_metadata={
            "experiment_name": EXPERIMENT_NAME,
            "target_family": target_family,
            "sample_frame": sample_frame,
            "variant": variant,
            "mosa_timestamp": cfg.mosa_timestamp,
            "tabpfn_estimator_mode": cfg.tabpfn_estimator_mode,
            "tabpfn_model_label": model_path_display_name(cfg.tabpfn_model_path),
            "tabpfn_model_count": model_path_count(cfg.tabpfn_model_path),
            "train_n": int(prepared.y_train.shape[0]),
            "test_n": int(prepared.y_test.shape[0]),
            "target_count": int(prepared.y_train.shape[1]),
            "selected_feature_count": int(selector.total_selected),
        },
    )

    config_payload = asdict(cfg)
    config_payload.update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "resolved_ccma_dir": str((ROOT / cfg.ccma_dir).resolve()),
            "resolved_mosa_files_dir": str((ROOT / cfg.mosa_files_dir).resolve()),
            "resolved_out_dir": str(out_dir.resolve()),
            "predictor_blocks": PREDICTOR_BLOCKS[target_family],
            "source_paths": prepared.source_paths,
            "train_sample_ids": prepared.train_sample_ids.tolist(),
            "test_sample_ids": prepared.test_sample_ids.tolist(),
            "metadata": prepared.metadata,
        }
    )
    with (out_dir / "config_used.json").open("w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)

    write_selected_feature_artifacts(out_dir, prepared, selector)
    maybe_log(
        cfg,
        f"[done][{target_family}/{sample_frame}/{variant}] "
        f"test_r2={summary_payload['test_r2']:.6f} "
        f"test_pearsonr={summary_payload['test_pearsonr']:.6f} "
        f"test_rmse={summary_payload['test_rmse']:.6f}",
    )
    return out_dir


def main() -> None:
    args = parse_args()
    cfg = args_to_config(args)
    device_spec = resolve_device_spec(cfg.device, cfg.gpu_id)

    sample_frames = expand_sample_frames(cfg.sample_frame)
    variants = expand_variants(cfg.variant)
    if cfg.tabpfn_estimator_mode == "finetune":
        maybe_log(
            cfg,
            "[setup] finetune mode selected; --tabpfn-fit-mode is ignored because the separate "
            "FinetunedTabPFNRegressor manages its own training loop.",
        )
    maybe_log(
        cfg,
        f"[setup] device={device_spec} target_family={cfg.target_family} "
        f"estimator={cfg.tabpfn_estimator_mode} model={model_path_display_name(cfg.tabpfn_model_path)} "
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
