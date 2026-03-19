#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


FEATURE_VARIANT_ORDER = ["original", "mosa_nan_only", "mosa_all"]
FEATURE_FRAME_ORDER = ["overlap", "expanded"]
PSEUDOLABEL_VARIANT_ORDER = ["real_overlap", "real_expanded", "pseudolabel_nan_only", "pseudolabel_all"]
FAMILY_ORDER = ["crisprcas9", "drugresponse"]
MODEL_ORDER = ["random_forest", "tabpfn"]
METRIC_SPECS: List[Tuple[str, str]] = [
    ("test_r2", "Macro Test R2"),
    ("test_pearsonr", "Macro Test Pearson r"),
    ("test_rmse", "Macro Test RMSE"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare TabPFN and random-forest experiment outputs.")
    parser.add_argument("--experiment-name", type=str, required=True, choices=["feature_augmentation", "pseudolabel_augmentation"])
    parser.add_argument("--mosa-timestamp", type=str, required=True)
    parser.add_argument("--tabpfn-root", type=str, default="reports/tabpfn")
    parser.add_argument("--random-forest-root", type=str, default="reports/random_forest")
    parser.add_argument("--out-dir", type=str, default="reports/model_comparison")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collect_model_results(model_root: Path, experiment_name: str, mosa_timestamp: str, model_name: str):
    run_root = (model_root / experiment_name / mosa_timestamp).resolve()
    summary_rows: List[Dict[str, object]] = []
    per_target_rows: List[pd.DataFrame] = []

    for metrics_path in sorted(run_root.glob("**/test_metrics_summary.json")):
        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["model_name"] = model_name
        summary_rows.append(payload)

        per_target_path = metrics_path.parent / "test_metrics_per_target.csv"
        if per_target_path.exists():
            df = pd.read_csv(per_target_path)
            for key in ["experiment_name", "target_family", "sample_frame", "variant", "mosa_timestamp"]:
                if key in payload:
                    df[key] = payload[key]
            df["model_name"] = model_name
            per_target_rows.append(df)

    summary_df = pd.DataFrame(summary_rows)
    per_target_df = pd.concat(per_target_rows, ignore_index=True) if per_target_rows else pd.DataFrame()
    return summary_df, per_target_df


def add_run_label(df: pd.DataFrame, experiment_name: str) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    if experiment_name == "feature_augmentation":
        out["run_label"] = out["sample_frame"].astype(str) + " | " + out["variant"].astype(str)
    else:
        out["run_label"] = out["variant"].astype(str)
    return out


def run_label_order(experiment_name: str) -> List[str]:
    if experiment_name == "feature_augmentation":
        return [f"{frame} | {variant}" for frame in FEATURE_FRAME_ORDER for variant in FEATURE_VARIANT_ORDER]
    return PSEUDOLABEL_VARIANT_ORDER


def plot_aggregate(summary_df: pd.DataFrame, experiment_name: str, out_dir: Path) -> None:
    if summary_df.empty:
        return
    summary_df = add_run_label(summary_df, experiment_name)
    order = run_label_order(experiment_name)

    fig, axes = plt.subplots(len(METRIC_SPECS), len(FAMILY_ORDER), figsize=(14, 8), dpi=300)
    axes = np.atleast_2d(axes)

    for row_idx, (metric_col, metric_label) in enumerate(METRIC_SPECS):
        for col_idx, family in enumerate(FAMILY_ORDER):
            ax = axes[row_idx, col_idx]
            sub_df = summary_df[summary_df["target_family"] == family].copy()
            if sub_df.empty:
                ax.set_visible(False)
                continue
            sns.barplot(
                data=sub_df,
                x="run_label",
                y=metric_col,
                hue="model_name",
                order=order,
                hue_order=MODEL_ORDER,
                ax=ax,
            )
            ax.set(
                title=f"{family} | {metric_label}",
                xlabel="Run",
                ylabel=metric_label,
            )
            ax.tick_params(axis="x", rotation=35)
            legend = ax.get_legend()
            if legend is not None and not (row_idx == 0 and col_idx == 0):
                legend.remove()
            elif legend is not None:
                legend.set_title("")

    plt.tight_layout()
    fig.savefig(out_dir / "aggregate_model_comparison.png", bbox_inches="tight")
    plt.close(fig)


def build_summary_pairwise(summary_df: pd.DataFrame, experiment_name: str) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    merge_keys = ["target_family", "variant"]
    if experiment_name == "feature_augmentation":
        merge_keys.append("sample_frame")

    tabpfn_df = summary_df[summary_df["model_name"] == "tabpfn"].copy()
    rf_df = summary_df[summary_df["model_name"] == "random_forest"].copy()
    merged = tabpfn_df.merge(rf_df, on=merge_keys, suffixes=("_tabpfn", "_random_forest"))
    if merged.empty:
        return merged

    for metric_col, _ in METRIC_SPECS:
        merged[f"delta_{metric_col}_tabpfn_minus_random_forest"] = (
            merged[f"{metric_col}_tabpfn"] - merged[f"{metric_col}_random_forest"]
        )
    return merged


def build_per_target_pairwise(per_target_df: pd.DataFrame, experiment_name: str) -> pd.DataFrame:
    if per_target_df.empty:
        return pd.DataFrame()

    merge_keys = ["target_family", "variant", "target"]
    if experiment_name == "feature_augmentation":
        merge_keys.append("sample_frame")

    tabpfn_df = per_target_df[per_target_df["model_name"] == "tabpfn"].copy()
    rf_df = per_target_df[per_target_df["model_name"] == "random_forest"].copy()
    merged = tabpfn_df.merge(rf_df, on=merge_keys, suffixes=("_tabpfn", "_random_forest"))
    if merged.empty:
        return merged

    for metric_col, _ in METRIC_SPECS:
        merged[f"delta_{metric_col}_tabpfn_minus_random_forest"] = (
            merged[f"{metric_col}_tabpfn"] - merged[f"{metric_col}_random_forest"]
        )
    return merged


def plot_delta_distributions(pairwise_df: pd.DataFrame, experiment_name: str, out_dir: Path) -> None:
    if pairwise_df.empty:
        return

    pairwise_df = add_run_label(pairwise_df, experiment_name)
    order = run_label_order(experiment_name)

    for metric_col, metric_label in METRIC_SPECS:
        delta_col = f"delta_{metric_col}_tabpfn_minus_random_forest"
        fig, axes = plt.subplots(1, len(FAMILY_ORDER), figsize=(13, 4), dpi=300, sharey=True)
        axes = np.atleast_1d(axes)
        for idx, family in enumerate(FAMILY_ORDER):
            ax = axes[idx]
            sub_df = pairwise_df[pairwise_df["target_family"] == family].copy()
            if sub_df.empty:
                ax.set_visible(False)
                continue
            sns.boxplot(data=sub_df, x="run_label", y=delta_col, order=order, ax=ax)
            ax.axhline(0.0, color="black", lw=0.8, linestyle="--")
            ax.set(
                title=f"{family} | TabPFN - RF | {metric_label}",
                xlabel="Run",
                ylabel=f"Delta {metric_label}",
            )
            ax.tick_params(axis="x", rotation=35)
        plt.tight_layout()
        fig.savefig(out_dir / f"delta_{metric_col}.png", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = (Path(args.out_dir).resolve() / args.experiment_name / args.mosa_timestamp).resolve()
    ensure_dir(out_dir)

    tabpfn_summary, tabpfn_per_target = collect_model_results(
        Path(args.tabpfn_root).resolve(),
        args.experiment_name,
        args.mosa_timestamp,
        "tabpfn",
    )
    rf_summary, rf_per_target = collect_model_results(
        Path(args.random_forest_root).resolve(),
        args.experiment_name,
        args.mosa_timestamp,
        "random_forest",
    )

    summary_df = pd.concat([tabpfn_summary, rf_summary], ignore_index=True)
    per_target_df = pd.concat([tabpfn_per_target, rf_per_target], ignore_index=True)
    if summary_df.empty:
        raise FileNotFoundError("No model result files found for comparison.")

    summary_df.to_csv(out_dir / "combined_summary.csv", index=False)
    per_target_df.to_csv(out_dir / "combined_per_target.csv", index=False)

    summary_pairwise = build_summary_pairwise(summary_df, args.experiment_name)
    per_target_pairwise = build_per_target_pairwise(per_target_df, args.experiment_name)
    summary_pairwise.to_csv(out_dir / "summary_model_comparison.csv", index=False)
    per_target_pairwise.to_csv(out_dir / "per_target_model_comparison.csv", index=False)

    plot_aggregate(summary_df, args.experiment_name, out_dir)
    plot_delta_distributions(per_target_pairwise, args.experiment_name, out_dir)


if __name__ == "__main__":
    main()
