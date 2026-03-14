#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
METRIC_SPECS: List[Tuple[str, str]] = [
    ("test_r2", "Macro Test R2"),
    ("test_pearsonr", "Macro Test Pearson r"),
    ("test_rmse", "Macro Test RMSE"),
]


def parse_args(default_experiment_name: Optional[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and plot TabPFN experiment outputs."
    )
    parser.add_argument("--reports-root", type=str, default="reports/tabpfn")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=default_experiment_name,
        required=default_experiment_name is None,
        choices=["feature_augmentation", "pseudolabel_augmentation"],
    )
    parser.add_argument("--mosa-timestamp", type=str, required=True)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collect_results(run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, object]] = []
    per_target_rows: List[pd.DataFrame] = []

    for metrics_path in sorted(run_root.glob("**/test_metrics_summary.json")):
        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        summary_rows.append(payload)

        per_target_path = metrics_path.parent / "test_metrics_per_target.csv"
        if per_target_path.exists():
            df = pd.read_csv(per_target_path)
            for key in ["experiment_name", "target_family", "sample_frame", "variant", "mosa_timestamp"]:
                if key in payload:
                    df[key] = payload[key]
            per_target_rows.append(df)

    summary_df = pd.DataFrame(summary_rows)
    per_target_df = pd.concat(per_target_rows, ignore_index=True) if per_target_rows else pd.DataFrame()
    return summary_df, per_target_df


def finalize_axis_legend(ax: plt.Axes, show_legend: bool) -> None:
    legend = ax.get_legend()
    if legend is None:
        return
    if show_legend:
        legend.set_title("")
    else:
        legend.remove()


def plot_feature_aggregate(summary_df: pd.DataFrame, out_dir: Path) -> None:
    if summary_df.empty:
        return

    fig, axes = plt.subplots(len(METRIC_SPECS), len(FAMILY_ORDER), figsize=(11, 8), dpi=300)
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
                x="sample_frame",
                y=metric_col,
                hue="variant",
                order=FEATURE_FRAME_ORDER,
                hue_order=FEATURE_VARIANT_ORDER,
                ax=ax,
            )
            ax.set(
                title=f"{family} | {metric_label}",
                xlabel="Sample frame",
                ylabel=metric_label,
            )
            ax.tick_params(axis="x", rotation=20)
            finalize_axis_legend(ax, show_legend=(row_idx == 0 and col_idx == 0))

    plt.tight_layout()
    fig.savefig(out_dir / "aggregate_test_metrics.png", bbox_inches="tight")
    plt.close(fig)


def plot_feature_per_target(per_target_df: pd.DataFrame, out_dir: Path) -> None:
    if per_target_df.empty:
        return

    for metric_col, metric_label in METRIC_SPECS:
        fig, axes = plt.subplots(
            len(FAMILY_ORDER),
            len(FEATURE_FRAME_ORDER),
            figsize=(12, 6),
            dpi=300,
            sharey=True,
        )
        axes = np.atleast_2d(axes)

        for row_idx, family in enumerate(FAMILY_ORDER):
            for col_idx, frame in enumerate(FEATURE_FRAME_ORDER):
                ax = axes[row_idx, col_idx]
                sub_df = per_target_df[
                    (per_target_df["target_family"] == family)
                    & (per_target_df["sample_frame"] == frame)
                ].copy()
                if sub_df.empty:
                    ax.set_visible(False)
                    continue
                sns.boxplot(
                    data=sub_df,
                    x="variant",
                    y=metric_col,
                    order=FEATURE_VARIANT_ORDER,
                    ax=ax,
                )
                ax.set(
                    title=f"{family} | {frame}",
                    xlabel="Variant",
                    ylabel=metric_label,
                )
                ax.tick_params(axis="x", rotation=20)

        plt.tight_layout()
        fig.savefig(out_dir / f"per_target_distribution_{metric_col}.png", bbox_inches="tight")
        plt.close(fig)


def build_feature_pairwise(per_target_df: pd.DataFrame) -> pd.DataFrame:
    if per_target_df.empty:
        return pd.DataFrame()

    original = per_target_df[per_target_df["variant"] == "original"].copy()
    rows: List[pd.DataFrame] = []
    for compare_variant in ["mosa_nan_only", "mosa_all"]:
        compare_df = per_target_df[per_target_df["variant"] == compare_variant].copy()
        merged = original.merge(
            compare_df,
            on=["target_family", "sample_frame", "target"],
            suffixes=("_original", f"_{compare_variant}"),
        )
        if merged.empty:
            continue
        merged["compare_variant"] = compare_variant
        for metric_col, _ in METRIC_SPECS:
            merged[f"delta_{metric_col}"] = merged[f"{metric_col}_{compare_variant}"] - merged[f"{metric_col}_original"]
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_pseudolabel_aggregate(summary_df: pd.DataFrame, out_dir: Path) -> None:
    if summary_df.empty:
        return

    fig, axes = plt.subplots(len(METRIC_SPECS), len(FAMILY_ORDER), figsize=(11, 8), dpi=300)
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
                x="variant",
                y=metric_col,
                order=PSEUDOLABEL_VARIANT_ORDER,
                ax=ax,
            )
            ax.set(
                title=f"{family} | {metric_label}",
                xlabel="Training-label variant",
                ylabel=metric_label,
            )
            ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()
    fig.savefig(out_dir / "aggregate_test_metrics.png", bbox_inches="tight")
    plt.close(fig)


def plot_pseudolabel_per_target(per_target_df: pd.DataFrame, out_dir: Path) -> None:
    if per_target_df.empty:
        return

    for metric_col, metric_label in METRIC_SPECS:
        fig, axes = plt.subplots(1, len(FAMILY_ORDER), figsize=(10, 4), dpi=300, sharey=True)
        axes = np.atleast_1d(axes)

        for col_idx, family in enumerate(FAMILY_ORDER):
            ax = axes[col_idx]
            sub_df = per_target_df[per_target_df["target_family"] == family].copy()
            if sub_df.empty:
                ax.set_visible(False)
                continue
            sns.boxplot(
                data=sub_df,
                x="variant",
                y=metric_col,
                order=PSEUDOLABEL_VARIANT_ORDER,
                ax=ax,
            )
            ax.set(
                title=family,
                xlabel="Training-label variant",
                ylabel=metric_label,
            )
            ax.tick_params(axis="x", rotation=25)

        plt.tight_layout()
        fig.savefig(out_dir / f"per_target_distribution_{metric_col}.png", bbox_inches="tight")
        plt.close(fig)


def build_pseudolabel_pairwise(per_target_df: pd.DataFrame) -> pd.DataFrame:
    if per_target_df.empty:
        return pd.DataFrame()

    pairs = [
        ("real_overlap", "pseudolabel_nan_only"),
        ("real_overlap", "pseudolabel_all"),
        ("real_expanded", "pseudolabel_nan_only"),
        ("real_expanded", "pseudolabel_all"),
    ]
    rows: List[pd.DataFrame] = []
    for base_variant, compare_variant in pairs:
        base_df = per_target_df[per_target_df["variant"] == base_variant].copy()
        compare_df = per_target_df[per_target_df["variant"] == compare_variant].copy()
        merged = base_df.merge(
            compare_df,
            on=["target_family", "target"],
            suffixes=(f"_{base_variant}", f"_{compare_variant}"),
        )
        if merged.empty:
            continue
        merged["base_variant"] = base_variant
        merged["compare_variant"] = compare_variant
        for metric_col, _ in METRIC_SPECS:
            merged[f"delta_{metric_col}"] = merged[f"{metric_col}_{compare_variant}"] - merged[f"{metric_col}_{base_variant}"]
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main(default_experiment_name: Optional[str] = None) -> None:
    args = parse_args(default_experiment_name=default_experiment_name)
    run_root = (Path(args.reports_root).resolve() / args.experiment_name / args.mosa_timestamp).resolve()
    out_dir = run_root / "comparison"
    ensure_dir(out_dir)

    summary_df, per_target_df = collect_results(run_root)
    if summary_df.empty:
        raise FileNotFoundError(f"No test_metrics_summary.json files found under {run_root}")

    summary_df.to_csv(out_dir / "combined_summary.csv", index=False)
    per_target_df.to_csv(out_dir / "combined_per_target.csv", index=False)

    if args.experiment_name == "feature_augmentation":
        pairwise_df = build_feature_pairwise(per_target_df)
        plot_feature_aggregate(summary_df, out_dir)
        plot_feature_per_target(per_target_df, out_dir)
    else:
        pairwise_df = build_pseudolabel_pairwise(per_target_df)
        plot_pseudolabel_aggregate(summary_df, out_dir)
        plot_pseudolabel_per_target(per_target_df, out_dir)

    pairwise_df.to_csv(out_dir / "pairwise_per_target_comparison.csv", index=False)


if __name__ == "__main__":
    main()
