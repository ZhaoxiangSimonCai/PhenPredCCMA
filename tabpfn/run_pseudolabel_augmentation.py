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
    read_mosa_view_matrix,
    read_raw_view_matrix,
    resolve_device_spec,
    save_split_indices,
    write_selected_feature_artifacts,
    write_test_outputs,
)
from feature_selection import fit_feature_selector


EXPERIMENT_NAME = "pseudolabel_augmentation"
VARIANT_ORDER = ["real_overlap", "real_expanded", "pseudolabel_nan_only", "pseudolabel_all"]


@dataclass
class RunConfig:
    ccma_dir: str
    mosa_files_dir: str
    out_dir: str
    mosa_timestamp: str
    target_family: str
    variant: str
    seed: int
    target_limit: int
    max_features: int
    min_features_per_modality: int
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
        description="Run TabPFN pseudo-label augmentation comparisons using MOSA target matrices as training labels."
    )
    parser.add_argument("--ccma-dir", type=str, default="data/clines/ccma_processed")
    parser.add_argument("--mosa-files-dir", type=str, default="reports/vae/files")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="reports/tabpfn/pseudolabel_augmentation",
        help="Per-run outputs are written under <out-dir>/<timestamp>/<family>/<variant>/",
    )
    parser.add_argument("--mosa-timestamp", type=str, default="20260313_162348")
    parser.add_argument(
        "--target-family",
        type=str,
        required=True,
        choices=["crisprcas9", "drugresponse"],
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["real_overlap", "real_expanded", "pseudolabel_nan_only", "pseudolabel_all", "all"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-limit", type=int, default=0, help="0 means use all targets.")
    parser.add_argument("--max-features", type=int, default=2000)
    parser.add_argument("--min-features-per-modality", type=int, default=100)
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
        variant=args.variant,
        seed=int(args.seed),
        target_limit=int(args.target_limit),
        max_features=int(args.max_features),
        min_features_per_modality=int(args.min_features_per_modality),
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


def expand_variants(raw: str) -> List[str]:
    if raw == "all":
        return VARIANT_ORDER
    return [raw]


def load_real_target_labels(
    ccma_dir: Path,
    target_family: str,
    frame_token: str,
    split: str,
):
    path = ccma_dir / f"{target_family}_ccma_{frame_token}_{split}.csv"
    df = read_raw_view_matrix(path, target_family)
    return df, path


def load_pseudolabel_train_labels(
    mosa_dir: Path,
    mosa_timestamp: str,
    target_family: str,
    variant: str,
):
    suffix = "nans_only" if variant == "pseudolabel_nan_only" else "all"
    path = mosa_dir / f"{mosa_timestamp}_imputed_{target_family}_train_{suffix}.csv.gz"
    df = read_mosa_view_matrix(path)
    return df, path


def load_raw_predictor_view(
    ccma_dir: Path,
    view_name: str,
    split: str,
):
    file_stem = RAW_VIEW_FILE_STEMS[view_name]
    path = ccma_dir / f"{file_stem}_ccma_mosa_{split}.csv"
    return read_raw_view_matrix(path, view_name if view_name != "mutations" else "mutations"), path


def build_run_data(
    cfg: RunConfig,
    target_family: str,
    variant: str,
) -> PreparedRunData:
    ccma_dir = (ROOT / cfg.ccma_dir).resolve()
    mosa_dir = (ROOT / cfg.mosa_files_dir).resolve()

    predictor_blocks = PREDICTOR_BLOCKS[target_family]
    if target_family in predictor_blocks:
        raise AssertionError(f"Target modality '{target_family}' must not be used as a predictor.")

    y_test_df, y_test_path = load_real_target_labels(ccma_dir, target_family, "mosa", "test")
    mosa_target_names, mosa_target_path = load_mosa_target_names(
        mosa_dir,
        cfg.mosa_timestamp,
        target_family,
    )

    if variant == "real_overlap":
        y_train_df, y_train_path = load_real_target_labels(ccma_dir, target_family, "overlap", "train")
        label_source_kind = "real_overlap"
    elif variant == "real_expanded":
        y_train_df, y_train_path = load_real_target_labels(ccma_dir, target_family, "mosa", "train")
        label_source_kind = "real_expanded"
    else:
        y_train_df, y_train_path = load_pseudolabel_train_labels(mosa_dir, cfg.mosa_timestamp, target_family, variant)
        label_source_kind = variant

    missing_train_targets = [name for name in mosa_target_names if name not in y_train_df.columns]
    missing_test_targets = [name for name in mosa_target_names if name not in y_test_df.columns]
    if missing_train_targets or missing_test_targets:
        raise AssertionError(
            f"MOSA target set for '{target_family}' does not match the train/test labels. "
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

    for view_name in predictor_blocks:
        train_df, train_path = load_raw_predictor_view(ccma_dir, view_name, "train")
        test_df, test_path = load_raw_predictor_view(ccma_dir, view_name, "test")
        feature_train_df[view_name] = train_df
        feature_test_df[view_name] = test_df
        source_paths[f"{view_name}_train"] = str(train_path)
        source_paths[f"{view_name}_test"] = str(test_path)

    train_sample_ids = [str(v) for v in y_train_df.index.tolist()]
    test_sample_ids = [str(v) for v in y_test_df.index.tolist()]

    if variant == "real_overlap" and len(train_sample_ids) != 95:
        raise AssertionError(f"Expected 95 overlap training samples, got {len(train_sample_ids)}.")
    if len(test_sample_ids) != 24:
        raise AssertionError(f"Expected 24 held-out test samples, got {len(test_sample_ids)}.")

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "variant": variant,
        "label_source_kind": label_source_kind,
        "predictor_source_kind": "raw_mosa_predictors",
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
    variant: str,
    device_spec: str,
) -> Path:
    prepared = build_run_data(cfg, target_family, variant)
    out_dir = (
        (ROOT / cfg.out_dir).resolve()
        / cfg.mosa_timestamp
        / target_family
        / variant
    )
    ensure_dir(out_dir)

    maybe_log(
        cfg,
        "[setup] "
        f"family={target_family} variant={variant} "
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
        f"[final][{target_family}/{variant}] "
        f"train_n={x_train.shape[0]} test_n={x_test.shape[0]} "
        f"selected_features={selector.total_selected}",
    )

    pred_test, fit_diag, _ = fit_predict_per_target(
        cfg,
        label=f"final_{target_family}_{variant}",
        target_names=prepared.target_names,
        train_x=x_train,
        train_y=prepared.y_train,
        train_mask=prepared.y_train_mask,
        eval_x=x_test,
        device_spec=device_spec,
        base_seed=cfg.seed + 30_000,
    )
    fit_diag.to_csv(out_dir / "target_fit_diagnostics.csv", index=False)

    summary_payload, _ = write_test_outputs(
        out_dir,
        prepared,
        pred_test,
        summary_metadata={
            "experiment_name": EXPERIMENT_NAME,
            "target_family": target_family,
            "variant": variant,
            "mosa_timestamp": cfg.mosa_timestamp,
            "tabpfn_estimator_mode": cfg.tabpfn_estimator_mode,
            "tabpfn_model_label": model_path_display_name(cfg.tabpfn_model_path),
            "tabpfn_model_count": model_path_count(cfg.tabpfn_model_path),
            "label_source_kind": prepared.metadata["label_source_kind"],
            "predictor_source_kind": prepared.metadata["predictor_source_kind"],
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
        f"[done][{target_family}/{variant}] "
        f"test_r2={summary_payload['test_r2']:.6f} "
        f"test_pearsonr={summary_payload['test_pearsonr']:.6f} "
        f"test_rmse={summary_payload['test_rmse']:.6f}",
    )
    return out_dir


def main() -> None:
    args = parse_args()
    cfg = args_to_config(args)
    device_spec = resolve_device_spec(cfg.device, cfg.gpu_id)

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
        f"variants={variants}",
    )

    for variant in variants:
        run_single_experiment(
            cfg=cfg,
            target_family=cfg.target_family,
            variant=variant,
            device_spec=device_spec,
        )


if __name__ == "__main__":
    main()
