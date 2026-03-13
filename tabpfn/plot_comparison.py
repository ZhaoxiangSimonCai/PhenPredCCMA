#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


VARIANT_ORDER = ["original", "mosa_nan_only", "mosa_all"]
FRAME_ORDER = ["overlap", "expanded"]
FAMILY_ORDER = ["crisprcas9", "drugresponse"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and plot TabPFN CCMA/MOSA comparison outputs."
    )
    parser.add_argument("--reports-root", type=str, default="reports/tabpfn")
    parser.add_argument("--mosa-timestamp", type=str, required=True)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collect_results(run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, object]] = []
    per_target_rows: List[pd.DataFrame] = []

    for metrics_path in sorted(run_root.glob("*/*/*/metrics_test.json")):
        target_family, sample_frame, variant = metrics_path.relative_to(run_root).parts[:3]
        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.update(
            {
                "target_family": target_family,
                "sample_frame": sample_frame,
                "variant": variant,
            }
        )
        summary_rows.append(payload)

        per_target_path = metrics_path.parent / "metrics_test_per_target.csv"
        if per_target_path.exists():
            df = pd.read_csv(per_target_path)
            df["target_family"] = target_family
            df["sample_frame"] = sample_frame
            df["variant"] = variant
            per_target_rows.append(df)

    summary_df = pd.DataFrame(summary_rows)
    per_target_df = pd.concat(per_target_rows, ignore_index=True) if per_target_rows else pd.DataFrame()
    return summary_df, per_target_df


def plot_aggregate_test_r2(summary_df: pd.DataFrame, out_dir: Path) -> None:
    if summary_df.empty:
        return

    fig, axes = plt.subplots(1, len(FAMILY_ORDER), figsize=(10, 3.5), dpi=300, sharey=True)
    if len(FAMILY_ORDER) == 1:
        axes = [axes]

    for ax, family in zip(axes, FAMILY_ORDER):
        sub_df = summary_df[summary_df["target_family"] == family].copy()
        if sub_df.empty:
            ax.set_visible(False)
            continue
        sns.barplot(
            data=sub_df,
            x="sample_frame",
            y="test_r2",
            hue="variant",
            order=FRAME_ORDER,
            hue_order=VARIANT_ORDER,
            ax=ax,
        )
        ax.set(
            title=f"{family} test R2",
            xlabel="Sample frame",
            ylabel="Test R2",
        )
        ax.tick_params(axis="x", rotation=20)
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title("")

    plt.tight_layout()
    fig.savefig(out_dir / "aggregate_test_r2.png", bbox_inches="tight")
    plt.close(fig)


def plot_per_target_distribution(per_target_df: pd.DataFrame, out_dir: Path) -> None:
    if per_target_df.empty:
        return

    fig, axes = plt.subplots(
        len(FAMILY_ORDER),
        len(FRAME_ORDER),
        figsize=(12, 6),
        dpi=300,
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    for row_idx, family in enumerate(FAMILY_ORDER):
        for col_idx, frame in enumerate(FRAME_ORDER):
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
                y="test_r2",
                order=VARIANT_ORDER,
                ax=ax,
            )
            ax.set(
                title=f"{family} | {frame}",
                xlabel="Variant",
                ylabel="Per-target test R2",
            )
            ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    fig.savefig(out_dir / "per_target_r2_distribution.png", bbox_inches="tight")
    plt.close(fig)


def build_pairwise_comparisons(per_target_df: pd.DataFrame) -> pd.DataFrame:
    if per_target_df.empty:
        return pd.DataFrame()

    original = per_target_df[per_target_df["variant"] == "original"].copy()
    rows = []
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
        merged["delta_r2"] = (
            merged[f"test_r2_{compare_variant}"] - merged["test_r2_original"]
        )
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_pairwise_scatter(pairwise_df: pd.DataFrame, out_dir: Path) -> None:
    if pairwise_df.empty:
        return

    for family in FAMILY_ORDER:
        for frame in FRAME_ORDER:
            sub_df = pairwise_df[
                (pairwise_df["target_family"] == family)
                & (pairwise_df["sample_frame"] == frame)
            ].copy()
            if sub_df.empty:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), dpi=300, sharex=True, sharey=True)
            for ax, compare_variant in zip(axes, ["mosa_nan_only", "mosa_all"]):
                comp_df = sub_df[sub_df["compare_variant"] == compare_variant].copy()
                if comp_df.empty:
                    ax.set_visible(False)
                    continue
                sns.scatterplot(
                    data=comp_df,
                    x="test_r2_original",
                    y=f"test_r2_{compare_variant}",
                    s=18,
                    lw=0,
                    ax=ax,
                )
                ax.axline((0, 0), slope=1, color="black", lw=0.7)
                ax.set(
                    title=f"{compare_variant} vs original",
                    xlabel="Original per-target test R2",
                    ylabel=f"{compare_variant} per-target test R2",
                )
            fig.suptitle(f"{family} | {frame} | per-target R2 comparison", y=1.02)
            plt.tight_layout()
            fig.savefig(out_dir / f"scatter_{family}_{frame}.png", bbox_inches="tight")
            plt.close(fig)


def plot_delta_distributions(pairwise_df: pd.DataFrame, out_dir: Path) -> None:
    if pairwise_df.empty:
        return

    for family in FAMILY_ORDER:
        for frame in FRAME_ORDER:
            sub_df = pairwise_df[
                (pairwise_df["target_family"] == family)
                & (pairwise_df["sample_frame"] == frame)
            ].copy()
            if sub_df.empty:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), dpi=300, sharey=True)
            for ax, compare_variant in zip(axes, ["mosa_nan_only", "mosa_all"]):
                comp_df = sub_df[sub_df["compare_variant"] == compare_variant].copy()
                if comp_df.empty:
                    ax.set_visible(False)
                    continue
                sns.histplot(comp_df["delta_r2"], bins=25, kde=True, ax=ax)
                ax.axvline(comp_df["delta_r2"].mean(), color="black", lw=0.8, linestyle="--")
                ax.set(
                    title=f"{compare_variant} - original",
                    xlabel="Delta per-target test R2",
                    ylabel="Count",
                )
            fig.suptitle(f"{family} | {frame} | MOSA delta distributions", y=1.02)
            plt.tight_layout()
            fig.savefig(out_dir / f"delta_{family}_{frame}.png", bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    args = parse_args()
    run_root = (Path(args.reports_root).resolve() / args.mosa_timestamp).resolve()
    out_dir = run_root / "comparison"
    ensure_dir(out_dir)

    summary_df, per_target_df = collect_results(run_root)
    summary_df.to_csv(out_dir / "combined_summary.csv", index=False)
    per_target_df.to_csv(out_dir / "combined_per_target.csv", index=False)

    pairwise_df = build_pairwise_comparisons(per_target_df)
    pairwise_df.to_csv(out_dir / "pairwise_per_target_comparison.csv", index=False)

    plot_aggregate_test_r2(summary_df, out_dir)
    plot_per_target_distribution(per_target_df, out_dir)
    plot_pairwise_scatter(pairwise_df, out_dir)
    plot_delta_distributions(pairwise_df, out_dir)


if __name__ == "__main__":
    main()
