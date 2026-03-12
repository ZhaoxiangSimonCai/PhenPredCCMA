import time
import argparse
import numpy as np
import torch
import pandas as pd

from PhenPred.vae import shap_folder
from PhenPred.vae.ArtifactPaths import runtime_artifact_path
from PhenPred.vae.Hypers import Hypers
from PhenPred.vae.Train import CLinesTrain
from PhenPred.vae.DatasetCCMA import CLinesDatasetCCMA
from PhenPred.vae.DatasetDepMap23Q2 import CLinesDatasetDepMap23Q2
from PhenPred.vae.DatasetDepMap24Q4 import CLinesDatasetDepMap24Q4


def build_dataset(hyperparameters):
    dataset_kwargs = dict(
        datasets=hyperparameters["datasets"],
        labels_names=hyperparameters["labels"],
        standardize=hyperparameters["standardize"],
        filter_features=hyperparameters["filter_features"],
        filtered_encoder_only=hyperparameters["filtered_encoder_only"],
        feature_miss_rate_thres=hyperparameters["feature_miss_rate_thres"],
    )

    dataset_class = str(hyperparameters.get("dataset_class", "depmap24q4")).lower()

    if dataset_class in {"depmap24q4", "depmap24", "24q4"}:
        return CLinesDatasetDepMap24Q4(**dataset_kwargs)

    if dataset_class in {"depmap23q2", "depmap23", "23q2"}:
        return CLinesDatasetDepMap23Q2(**dataset_kwargs)

    if dataset_class == "ccma":
        dataset_kwargs["min_views_per_sample"] = hyperparameters.get(
            "min_views_per_sample", 2
        )
        dataset_kwargs["align_to_reference_features"] = hyperparameters.get(
            "align_to_reference_features", False
        )
        dataset_kwargs["reference_hypers_json"] = hyperparameters.get(
            "reference_hypers_json"
        ) or hyperparameters.get("transfer_hypers_json")
        dataset_kwargs["reference_feature_views"] = hyperparameters.get(
            "reference_feature_views"
        )
        dataset_kwargs["labels_mutations_file"] = hyperparameters.get(
            "labels_mutations_file"
        )
        return CLinesDatasetCCMA(**dataset_kwargs)

    raise ValueError(f"Unsupported dataset_class='{dataset_class}'")


def safe_stratify_by_tissue(data, hyperparameters):
    if "tissue" not in hyperparameters.get("labels", []):
        return None

    try:
        strat = data.samples_by_tissue("Haematopoietic and Lymphoid")
    except Exception:
        return None

    return strat if strat.nunique() > 1 else None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SHAP for a trained MOSA/CCMA run timestamp."
    )
    parser.add_argument(
        "--timestamp",
        required=True,
        help="Run timestamp used for model/hyperparameter files, e.g. 20260225_042500.",
    )
    parser.add_argument(
        "--hypers-json",
        default=None,
        help=(
            "Optional explicit hyperparameter json path. "
            "If omitted, uses reports/vae/configs/history/{timestamp}_hyperparameters.json "
            "with fallback to reports/vae/files/{timestamp}_hyperparameters.json."
        ),
    )
    parser.add_argument(
        "--explain-target",
        default=None,
        help="Explain target view, e.g. crisprcas9. Defaults to shap_target_view or latent.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="SHAP nsamples argument (default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for SHAP run (default: 42).",
    )
    parser.add_argument(
        "--skip-top200",
        action="store_true",
        help="Skip writing the top-200 feature feather output.",
    )
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Run SHAP on all available samples (overrides target-specific mini-batch size).",
    )
    parser.add_argument(
        "--shap-batch-size",
        type=int,
        default=None,
        help="Optional explicit SHAP batch size; ignored when --all-samples is set.",
    )
    parser.add_argument(
        "--multi-gpu-shap",
        action="store_true",
        help=(
            "Force DataParallel model forward for SHAP. "
            "When omitted, multi-GPU SHAP is still enabled automatically if >1 CUDA GPU is visible."
        ),
    )
    parser.add_argument(
        "--target-chunk-size",
        type=int,
        default=None,
        help=(
            "Number of output targets per SHAP chunk. "
            "Defaults to hypers['shap_target_chunk_size'] or 1."
        ),
    )
    parser.add_argument(
        "--shap-grad-batch-size",
        type=int,
        default=None,
        help=(
            "GradientExplainer internal gradient batch size. "
            "Larger values improve GPU utilization but increase memory. "
            "Set <= 0 (or omit) to use SHAP default (50)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.hypers_json is None:
        hyperparameters = Hypers.read_hyperparameters(timestamp=args.timestamp)
    else:
        hyperparameters = Hypers.read_hyperparameters(hypers_json=args.hypers_json)

    explain_target = (
        args.explain_target
        if args.explain_target is not None
        else hyperparameters.get("shap_target_view") or "latent"
    )

    clines_db = build_dataset(hyperparameters)
    train = CLinesTrain(
        clines_db,
        hyperparameters,
        verbose=hyperparameters.get("verbose", 0),
        stratify_cv_by=safe_stratify_by_tissue(clines_db, hyperparameters),
        timestamp=args.timestamp,
    )

    start_time = time.time()
    train.load_model()
    auto_multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1
    use_multi_gpu_shap = args.multi_gpu_shap or auto_multi_gpu
    if use_multi_gpu_shap:
        print(f"Multi-GPU SHAP enabled across {torch.cuda.device_count()} GPU(s).")

    shap_result = train.run_shap(
        n_samples=args.n_samples,
        seed=args.seed,
        explain_target=explain_target,
        use_all_samples=args.all_samples,
        shap_batch_size=args.shap_batch_size,
        shap_grad_batch_size=args.shap_grad_batch_size,
        use_data_parallel=use_multi_gpu_shap,
        target_chunk_size=args.target_chunk_size,
        show_progress=True,
        aggregate_abs_mean=True,
    )

    if not isinstance(shap_result, pd.DataFrame):
        raise RuntimeError(
            "Expected aggregated SHAP dataframe output from run_shap, "
            f"got type={type(shap_result)}."
        )

    if not args.skip_top200:
        train.save_shap_top200_features(shap_result, explain_target=explain_target)
    shap_df = train.save_shap(shap_result, explain_target=explain_target)

    runtime = time.time() - start_time
    suffix = "_mean_abs"
    shap_values_path = runtime_artifact_path(
        f"{train.timestamp}_shap_values_{explain_target}{suffix}.csv.gz"
    )
    feature_rank_path = runtime_artifact_path(
        f"{train.timestamp}_shap_feature_ranking_{explain_target}{suffix}.csv"
    )
    omic_rank_path = runtime_artifact_path(
        f"{train.timestamp}_shap_omic_ranking_{explain_target}{suffix}.csv"
    )

    print(f"SHAP explain_target={explain_target}")
    print("Saved table contains mean absolute SHAP values only (no per-sample values).")
    print(f"SHAP table: {shap_values_path}")
    print(f"Feature ranking: {feature_rank_path}")
    print(f"Omic ranking: {omic_rank_path}")
    if not args.skip_top200:
        print(
            "Top-200 table: "
            + runtime_artifact_path(
                f"{train.timestamp}_shap_values_top_features_{explain_target}{suffix}.feather"
            )
        )
    print(f"Rows in saved SHAP table: {shap_df.shape[0]:,}")
    print(
        "Runtime: {:02d}:{:02d}:{:02d}".format(
            int(runtime // 3600), int((runtime % 3600) // 60), int(runtime % 60)
        )
    )
