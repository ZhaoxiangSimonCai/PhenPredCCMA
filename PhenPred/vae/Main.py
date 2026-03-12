# %load_ext autoreload
# %autoreload 2

import os
import sys
import time
import json
import argparse
import torch
import PhenPred
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PhenPred.Utils import two_vars_correlation
from PhenPred.vae import plot_folder, data_folder
from PhenPred.vae.ArtifactPaths import (
    ensure_vae_artifact_dirs,
    runtime_artifact_path,
    timestamped_hyperparameters_output_path,
)
from PhenPred.vae.Hypers import Hypers
from PhenPred.vae.Train import CLinesTrain
from PhenPred.vae.DatasetMOFA import CLinesDatasetMOFA
from PhenPred.vae.DatasetMOVE import CLinesDatasetMOVE
from PhenPred.vae.DatasetCCMA import CLinesDatasetCCMA
from PhenPred.vae.DatasetMixOmics import CLinesDatasetMixOmics
from PhenPred.vae.BenchmarkCRISPR import CRISPRBenchmark
from PhenPred.vae.BenchmarkDrug import DrugResponseBenchmark
from PhenPred.vae.BenchmarkMismatch import MismatchBenchmark
from PhenPred.vae.BenchmarkInternal import InternalBenchmark
from PhenPred.vae.BenchmarkProteomics import ProteomicsBenchmark
from PhenPred.vae.BenchmarkLatentSpace import LatentSpaceBenchmark
from PhenPred.vae.DatasetDepMap23Q2 import CLinesDatasetDepMap23Q2
from PhenPred.vae.DatasetDepMap24Q4 import CLinesDatasetDepMap24Q4

torch.manual_seed(0)
np.random.seed(0)


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


def save_hyperparameters(hyperparameters, timestamp):
    ensure_vae_artifact_dirs()
    json.dump(
        hyperparameters,
        open(timestamped_hyperparameters_output_path(timestamp), "w"),
        indent=4,
        default=lambda o: "<not serializable>",
    )


def get_cvtest_datasets(hyperparameters, train):
    if hyperparameters["skip_cv"]:
        return {}

    cvtest_datasets = {}
    missing_views = []

    for k in hyperparameters["datasets"]:
        dpath = runtime_artifact_path(f"{train.timestamp}_imputed_{k}_cvtest.csv.gz")
        if os.path.isfile(dpath):
            cvtest_datasets[k] = pd.read_csv(dpath, index_col=0)
        else:
            missing_views.append(k)

    if len(missing_views) == 0:
        print(
            "Reusing saved CV test predictions from primary training "
            f"({len(cvtest_datasets)} views)."
        )
        return cvtest_datasets

    print(
        "Missing saved CV test predictions for views: "
        f"{', '.join(missing_views)}. Re-running CV with configured strategy."
    )
    _, cvtest_datasets = train.training(drop_last=True, skip_cv_save=False)
    return cvtest_datasets


def run_internal_benchmark(hyperparameters, train, clines_db, vae_predicted):
    cvtest_datasets = get_cvtest_datasets(hyperparameters, train)

    internal_benchmark = InternalBenchmark(
        train.timestamp, clines_db, vae_predicted, cvtest_datasets=cvtest_datasets
    )
    internal_benchmark.run()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MOSA/GMVAE and run configured benchmarks."
    )
    parser.add_argument(
        "--hypers-json",
        default=None,
        help=(
            "Optional path to a hyperparameters JSON file. "
            "If omitted, defaults to reports/vae/configs/hyperparameters.json."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_time = time.time()
    hyperparameters = Hypers.read_hyperparameters(hypers_json=args.hypers_json)

    clines_db = build_dataset(hyperparameters)
    train = CLinesTrain(
        clines_db,
        hyperparameters,
        verbose=hyperparameters["verbose"],
        stratify_cv_by=safe_stratify_by_tissue(clines_db, hyperparameters),
    )
    train.run()

    benchmark_mode = str(hyperparameters.get("benchmark_mode", "full")).lower()
    skip_benchmarks = bool(hyperparameters.get("skip_benchmarks", False))

    if skip_benchmarks or benchmark_mode == "none":
        save_hyperparameters(hyperparameters, train.timestamp)
        total_time = time.time() - start_time
        print(f"Total time: {int(total_time // 3600):02d}:{int((total_time % 3600) // 60):02d}")
        sys.exit(0)

    vae_imputed, vae_latent = train.load_vae_reconstructions()
    vae_predicted, _ = train.load_vae_reconstructions(mode="all")

    if benchmark_mode == "internal":
        run_internal_benchmark(hyperparameters, train, clines_db, vae_predicted)
        save_hyperparameters(hyperparameters, train.timestamp)
        total_time = time.time() - start_time
        print(f"Total time: {int(total_time // 3600):02d}:{int((total_time % 3600) // 60):02d}")
        sys.exit(0)

    if benchmark_mode != "full":
        raise ValueError(f"Unsupported benchmark_mode='{benchmark_mode}'")

    mofa_imputed, mofa_latent = CLinesDatasetMOFA.load_reconstructions(clines_db)
    move_diabetes_imputed, move_diabetes_latent = (
        CLinesDatasetMOVE.load_reconstructions(clines_db)
    )
    _, mixOmics_latent = CLinesDatasetMixOmics.load_reconstructions(clines_db)

    samples_mgexp = ~clines_db.dfs["transcriptomics"].isnull().all(axis=1)
    gexp_gdsc = pd.read_csv(f"{data_folder}/transcriptomics.csv", index_col=0).T
    gexp_move = vae_imputed["transcriptomics"]
    samples = set(gexp_gdsc.index).intersection(gexp_move.index)
    genes = list(set(gexp_gdsc.columns).intersection(gexp_move.columns))

    gexp_corr = pd.DataFrame(
        [
            two_vars_correlation(
                gexp_gdsc.loc[s, genes],
                gexp_move.loc[s, genes],
                method="pearson",
                extra_fields=dict(sample=s, with_gexp=samples_mgexp.loc[s]),
            )
            for s in samples
        ]
    )

    _, ax = plt.subplots(1, 1, figsize=(0.5, 2), dpi=600)
    sns.boxplot(
        data=gexp_corr,
        x="with_gexp",
        y="corr",
        palette="tab20c",
        linewidth=0.3,
        fliersize=1,
        notch=True,
        saturation=1.0,
        showcaps=False,
        boxprops=dict(linewidth=0.5, edgecolor="black"),
        whiskerprops=dict(linewidth=0.5, color="black"),
        flierprops=dict(
            marker="o",
            markerfacecolor="black",
            markersize=1.0,
            linestyle="none",
            markeredgecolor="none",
            alpha=0.6,
        ),
        medianprops=dict(linestyle="-", linewidth=0.5),
        ax=ax,
    )

    ax.set(
        title="",
        ylabel="Correlation between reconstructed\nand GDSC transcriptomics (Pearson's r)",
        xlabel="Sample with transcriptomics\nduring MOSA training",
    )
    PhenPred.save_figure(
        f"{plot_folder}/{train.timestamp}_reconstructed_gexp_correlation_boxplot"
    )

    latent_benchmark = LatentSpaceBenchmark(
        train.timestamp,
        clines_db,
        vae_latent,
        mofa_latent,
        move_diabetes_latent,
        mixOmics_latent,
    )
    latent_benchmark.plot_method_correlations()
    latent_benchmark.plot_latent_spaces(
        markers=clines_db.get_features(
            dict(
                metabolomics=["1-methylnicotinamide"],
                transcriptomics=["VIM"],
            )
        ),
    )

    plot_df = clines_db.get_features(
        dict(
            metabolomics=["1-methylnicotinamide"],
            transcriptomics=["VIM", "CDH1", "NNMT"],
            proteomics=["VIM", "CDH1"],
        )
    )
    g = sns.clustermap(
        plot_df.corr(),
        cmap="RdYlGn",
        center=0,
        xticklabels=False,
        vmin=-1,
        vmax=1,
        annot=True,
        annot_kws={"fontsize": 5},
        fmt=".2f",
        lw=0.0,
        cbar_kws={"shrink": 0.5},
        figsize=(3.0, 1.5),
    )
    if g.ax_cbar:
        g.ax_cbar.set_ylabel("Pearson\ncorrelation")
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    PhenPred.save_figure(f"{plot_folder}/selected_features_clustermap")

    print("Running drug benchmark")
    dres_benchmark = DrugResponseBenchmark(
        train.timestamp, clines_db, vae_imputed, mofa_imputed, move_diabetes_imputed
    )
    dres_benchmark.run()

    print("Running proteomics benchmark")
    proteomics_benchmark = ProteomicsBenchmark(
        train.timestamp, clines_db, vae_imputed, mofa_imputed, move_diabetes_imputed
    )
    proteomics_benchmark.run()
    proteomics_benchmark.copy_number(proteomics_only=True)

    print("Running CRISPR benchmark")
    crispr_benchmark = CRISPRBenchmark(
        train.timestamp,
        clines_db,
        vae_imputed,
        mofa_imputed,
        skew_threshold=-0.5,
        vae_latent=vae_latent,
    )
    crispr_benchmark.run()
    crispr_benchmark.gene_skew_correlation()
    crispr_benchmark.plot_associations(
        [
            ("BRAF", "MAPK1", "BRAF_mut"),
            ("FLI1", "TRIM8", "FLI1_EWSR1_fusion"),
            ("KRAS", "RAF1", "KRAS_mut"),
            ("NRAS", "SHOC2", "NRAS_mut"),
        ]
    )

    if not hyperparameters["skip_cv"]:
        print("Running mismatch benchmark with CV")
        cvtest_datasets = get_cvtest_datasets(hyperparameters, train)
        mismatch_benchmark = MismatchBenchmark(
            train.timestamp, clines_db, vae_predicted, cvtest_datasets
        )
        mismatch_benchmark.run()

    save_hyperparameters(hyperparameters, train.timestamp)

    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    print(f"Total time: {hours:02d}:{minutes:02d}")
