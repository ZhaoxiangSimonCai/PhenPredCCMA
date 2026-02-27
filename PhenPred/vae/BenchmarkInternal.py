import os
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

import PhenPred
from PhenPred.vae import plot_folder


class InternalBenchmark:
    def __init__(self, timestamp, data, vae_predicted, cvtest_datasets=None):
        self.timestamp = timestamp
        self.data = data
        self.vae_predicted = vae_predicted
        self.cvtest_datasets = cvtest_datasets or {}
        self.output_dir = f"{plot_folder}/internal"

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    @staticmethod
    def _safe_label(view_name):
        return PhenPred.OMIC_NAMES.get(view_name, view_name)

    @staticmethod
    def _safe_color(view_name):
        return PhenPred.OMIC_PALETTE.get(view_name, "#4c72b0")

    @staticmethod
    def _paired_arrays(df_true, df_pred):
        df_pred = df_pred.reindex(index=df_true.index, columns=df_true.columns)
        mask = df_true.notna() & df_pred.notna()
        x = df_true.where(mask).values.flatten()
        y = df_pred.where(mask).values.flatten()
        keep = np.isfinite(x) & np.isfinite(y)
        return x[keep], y[keep]

    @staticmethod
    def _corr_safe(a, b, method="pearson"):
        if len(a) < 3:
            return np.nan
        if np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        if method == "spearman":
            return stats.spearmanr(a, b)[0]
        return stats.pearsonr(a, b)[0]

    def _entry_metrics(self, df_true, df_pred, source):
        x, y = self._paired_arrays(df_true, df_pred)
        if len(x) == 0:
            return dict(
                source=source,
                n_obs=0,
                rmse=np.nan,
                mae=np.nan,
                pearson=np.nan,
                spearman=np.nan,
            )

        return dict(
            source=source,
            n_obs=len(x),
            rmse=float(np.sqrt(mean_squared_error(x, y))),
            mae=float(mean_absolute_error(x, y)),
            pearson=float(self._corr_safe(x, y, method="pearson")),
            spearman=float(self._corr_safe(x, y, method="spearman")),
        )

    def _feature_correlations(self, df_true, df_pred, min_n=15):
        df_pred = df_pred.reindex(index=df_true.index, columns=df_true.columns)
        rows = []
        for feature in df_true.columns:
            x = df_true[feature]
            y = df_pred[feature]
            mask = x.notna() & y.notna()
            if int(mask.sum()) < min_n:
                continue
            x = x[mask].values
            y = y[mask].values
            corr = self._corr_safe(x, y, method="pearson")
            rows.append(dict(feature=feature, n_obs=int(mask.sum()), pearson=corr))
        return pd.DataFrame(rows)

    def _plot_crispr_skew(self, skew_df, suffix):
        _, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=300)
        sns.scatterplot(data=skew_df, x="orig_skew", y="pred_skew", s=8, lw=0, ax=ax)
        sns.regplot(
            data=skew_df,
            x="orig_skew",
            y="pred_skew",
            scatter=False,
            color="#fc8d62",
            line_kws={"lw": 1},
            ax=ax,
        )
        ax.set(
            xlabel="Original CRISPR skew",
            ylabel=f"Predicted CRISPR skew ({suffix})",
            title=f"CRISPR skew preservation (N={len(skew_df):,})",
        )
        ax.axline((0, 0), slope=1, color="black", lw=0.5)
        PhenPred.save_figure(f"{self.output_dir}/{self.timestamp}_crispr_skew_{suffix}")

    def _plot_entry_metrics(self, metrics_df, source_name):
        if metrics_df.empty:
            return

        metrics_df = metrics_df.copy()
        metrics_df["view_label"] = metrics_df["view_name"].map(self._safe_label)
        view_order = (
            metrics_df.sort_values("pearson", ascending=False)["view_name"].tolist()
        )
        view_label_order = [self._safe_label(v) for v in view_order]

        corr_df = pd.melt(
            metrics_df,
            id_vars=["view_name", "view_label"],
            value_vars=["pearson", "spearman"],
            var_name="metric",
            value_name="value",
        )
        corr_df = corr_df[corr_df["value"].notna()]

        err_df = pd.melt(
            metrics_df,
            id_vars=["view_name", "view_label"],
            value_vars=["rmse", "mae"],
            var_name="metric",
            value_name="value",
        )
        err_df = err_df[err_df["value"].notna()]

        _, axes = plt.subplots(1, 2, figsize=(7, 2.3), dpi=300)

        if not corr_df.empty:
            sns.barplot(
                data=corr_df,
                x="view_label",
                y="value",
                hue="metric",
                order=view_label_order,
                ax=axes[0],
            )
        axes[0].set(
            xlabel="",
            ylabel="Correlation",
            title=f"Entry correlations ({source_name})",
            ylim=(-0.05, 1.05),
        )
        axes[0].tick_params(axis="x", rotation=30)

        if not err_df.empty:
            sns.barplot(
                data=err_df,
                x="view_label",
                y="value",
                hue="metric",
                order=view_label_order,
                ax=axes[1],
            )
        axes[1].set(
            xlabel="",
            ylabel="Error",
            title=f"Entry errors ({source_name})",
        )
        axes[1].tick_params(axis="x", rotation=30)

        for ax in axes:
            leg = ax.get_legend()
            if leg is not None:
                leg.set_title("")

        PhenPred.save_figure(
            f"{self.output_dir}/{self.timestamp}_{source_name}_entry_metrics_summary"
        )

    def _plot_feature_summary(self, feature_summary_df, source_name):
        if feature_summary_df.empty:
            return

        plot_df = feature_summary_df.copy()
        plot_df = plot_df.sort_values("pearson_median", ascending=False)
        view_names = plot_df["view_name"].tolist()
        x = np.arange(len(view_names))
        y = plot_df["pearson_median"].to_numpy()
        yerr_low = y - plot_df["pearson_q1"].to_numpy()
        yerr_high = plot_df["pearson_q3"].to_numpy() - y
        colors = [self._safe_color(v) for v in view_names]

        _, axes = plt.subplots(1, 2, figsize=(7, 2.3), dpi=300)

        axes[0].bar(x, y, color=colors)
        axes[0].errorbar(
            x=x,
            y=y,
            yerr=np.vstack([yerr_low, yerr_high]),
            fmt="none",
            ecolor="black",
            elinewidth=0.6,
            capsize=2,
        )
        axes[0].set(
            xlabel="",
            ylabel="Pearson",
            title=f"Feature correlation median/IQR ({source_name})",
            ylim=(-0.05, 1.05),
        )
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(
            [self._safe_label(v) for v in view_names], rotation=30, ha="right"
        )

        axes[1].bar(x, plot_df["n_features"], color=colors)
        axes[1].set(
            xlabel="",
            ylabel="# Features",
            title=f"Features with >=15 paired obs ({source_name})",
        )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(
            [self._safe_label(v) for v in view_names], rotation=30, ha="right"
        )

        PhenPred.save_figure(
            f"{self.output_dir}/{self.timestamp}_{source_name}_feature_summary_plot"
        )

    def _plot_feature_corr_distribution(self, feature_corr_df, source_name, view_name):
        if feature_corr_df.empty:
            return

        plot_df = feature_corr_df.copy()
        plot_df = plot_df[plot_df["pearson"].notna()]
        if plot_df.empty:
            return

        _, ax = plt.subplots(1, 1, figsize=(3.0, 2.0), dpi=300)
        sns.histplot(
            plot_df,
            x="pearson",
            bins=40,
            color=self._safe_color(view_name),
            stat="density",
            alpha=0.75,
            ax=ax,
        )
        med = float(plot_df["pearson"].median())
        ax.axvline(med, color="black", lw=0.9, linestyle="--")
        ax.set(
            xlabel="Per-feature Pearson",
            ylabel="Density",
            title=(
                f"{self._safe_label(view_name)} feature-correlation distribution "
                f"({source_name})"
            ),
        )
        PhenPred.save_figure(
            f"{self.output_dir}/{self.timestamp}_{source_name}_{view_name}_feature_corr_hist"
        )

    def _plot_sources_comparison(self, all_metrics, cv_metrics, all_feature, cv_feature):
        if all_metrics.empty or cv_metrics.empty:
            return

        merged = all_metrics.merge(
            cv_metrics,
            on="view_name",
            suffixes=("_all", "_cv"),
        )
        if merged.empty:
            return

        _, ax = plt.subplots(1, 1, figsize=(2.5, 2.4), dpi=300)
        for _, row in merged.iterrows():
            view = row["view_name"]
            ax.scatter(
                row["pearson_all"],
                row["pearson_cv"],
                s=28,
                color=self._safe_color(view),
                edgecolor="none",
            )
            ax.text(
                row["pearson_all"] + 0.005,
                row["pearson_cv"] + 0.005,
                self._safe_label(view),
                fontsize=5,
            )

        ax.axline((0, 0), slope=1, color="black", lw=0.6)
        ax.set(
            xlabel="Pearson (all-data predictions)",
            ylabel="Pearson (CV predictions)",
            title="Entry-level Pearson: all vs cv",
            xlim=(-0.05, 1.05),
            ylim=(-0.05, 1.05),
        )
        PhenPred.save_figure(
            f"{self.output_dir}/{self.timestamp}_all_vs_cv_entry_pearson"
        )

        if all_feature.empty or cv_feature.empty:
            return

        merged_feature = all_feature.merge(
            cv_feature, on="view_name", suffixes=("_all", "_cv")
        )
        if merged_feature.empty:
            return

        _, ax = plt.subplots(1, 1, figsize=(2.5, 2.4), dpi=300)
        for _, row in merged_feature.iterrows():
            view = row["view_name"]
            ax.scatter(
                row["pearson_median_all"],
                row["pearson_median_cv"],
                s=28,
                color=self._safe_color(view),
                edgecolor="none",
            )
            ax.text(
                row["pearson_median_all"] + 0.005,
                row["pearson_median_cv"] + 0.005,
                self._safe_label(view),
                fontsize=5,
            )

        ax.axline((0, 0), slope=1, color="black", lw=0.6)
        ax.set(
            xlabel="Median per-feature Pearson (all)",
            ylabel="Median per-feature Pearson (cv)",
            title="Feature-level Pearson median: all vs cv",
            xlim=(-0.05, 1.05),
            ylim=(-0.05, 1.05),
        )
        PhenPred.save_figure(
            f"{self.output_dir}/{self.timestamp}_all_vs_cv_feature_median_pearson"
        )

    def _run_single_source(self, source_name, pred_map):
        metrics_rows = []
        feature_summary_rows = []

        for view_name, df_true in self.data.dfs.items():
            if view_name not in pred_map:
                continue

            df_pred = pred_map[view_name]

            row = self._entry_metrics(df_true, df_pred, source=source_name)
            row["view_name"] = view_name
            metrics_rows.append(row)

            feature_corr_df = self._feature_correlations(df_true, df_pred)
            if feature_corr_df.empty:
                continue

            feature_corr_df.to_csv(
                f"{self.output_dir}/{self.timestamp}_{source_name}_{view_name}_feature_corr.csv.gz",
                compression="gzip",
                index=False,
            )
            self._plot_feature_corr_distribution(
                feature_corr_df, source_name=source_name, view_name=view_name
            )

            feature_summary_rows.append(
                dict(
                    source=source_name,
                    view_name=view_name,
                    n_features=int(feature_corr_df.shape[0]),
                    pearson_median=float(feature_corr_df["pearson"].median()),
                    pearson_q1=float(feature_corr_df["pearson"].quantile(0.25)),
                    pearson_q3=float(feature_corr_df["pearson"].quantile(0.75)),
                )
            )

        metrics_df = pd.DataFrame(metrics_rows)
        feature_summary_df = pd.DataFrame(feature_summary_rows)

        if not metrics_df.empty:
            metrics_df.to_csv(
                f"{self.output_dir}/{self.timestamp}_{source_name}_entry_metrics.csv",
                index=False,
            )
            self._plot_entry_metrics(metrics_df, source_name)

        if not feature_summary_df.empty:
            feature_summary_df.to_csv(
                f"{self.output_dir}/{self.timestamp}_{source_name}_feature_summary.csv",
                index=False,
            )
            self._plot_feature_summary(feature_summary_df, source_name)

        if "crisprcas9" in pred_map and "crisprcas9" in self.data.dfs:
            df_true = self.data.dfs["crisprcas9"]
            df_pred = pred_map["crisprcas9"].reindex(
                index=df_true.index, columns=df_true.columns
            )
            skew_df = pd.concat(
                [
                    df_true.apply(stats.skew).rename("orig_skew"),
                    df_pred.apply(stats.skew).rename("pred_skew"),
                ],
                axis=1,
            ).dropna()
            skew_df.to_csv(
                f"{self.output_dir}/{self.timestamp}_{source_name}_crispr_skew.csv"
            )
            if not skew_df.empty:
                self._plot_crispr_skew(skew_df, source_name)

        return metrics_df, feature_summary_df

    def run(self):
        all_metrics, all_feature = self._run_single_source("all", self.vae_predicted)

        cv_metrics, cv_feature = pd.DataFrame(), pd.DataFrame()
        if self.cvtest_datasets:
            cv_metrics, cv_feature = self._run_single_source("cv", self.cvtest_datasets)

        if not all_metrics.empty and not cv_metrics.empty:
            self._plot_sources_comparison(
                all_metrics=all_metrics,
                cv_metrics=cv_metrics,
                all_feature=all_feature,
                cv_feature=cv_feature,
            )
