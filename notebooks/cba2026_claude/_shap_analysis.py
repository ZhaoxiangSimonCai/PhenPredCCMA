"""CBA 2026 SHAP analysis — publication-quality figure generators.

Loads the CRISPR-Cas9 and drug-response SHAP exports for the selected +CNV
MOSA run (``20260505_131645``) and produces every figure and table consumed
by the two ``cba2026_claude`` notebooks.

All figures are built with the project ``_plot_style`` helpers:
- Okabe-Ito-derived colourblind-safe palette
- Three type scales (composite / column / full)
- 300 dpi PNG plus Type-42 editable PDF via ``save_figure``
- Reference legends pushed outside axes to avoid landing on data
- Layout uses ``gridspec`` and explicit ``subplots_adjust`` so corner text
  and tick labels never overlap.

Output goes to ``reports/cba2026_claude/shap_analysis/<TIMESTAMP>/``.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _plot_style import (
    PALETTE,
    SCATTER_ALPHA,
    configure_nature_style,
    panel_label,
    save_figure,
)


TIMESTAMP = "20260505_131645"

TARGET_META = {
    "crisprcas9": "CRISPR-Cas9",
    "drugresponse": "Drug response",
}

OMIC_ORDER = [
    "transcriptomics",
    "methylation",
    "drugresponse",
    "crisprcas9",
    "conditionals",
    "copynumber",
]
OMIC_DISPLAY = {
    "transcriptomics": "Transcriptomics",
    "methylation": "Methylation",
    "drugresponse": "Drug response",
    "crisprcas9": "CRISPR-Cas9",
    "conditionals": "Conditionals",
    "copynumber": "Copy number",
}
OMIC_COLORS = {
    "transcriptomics": "#0072B2",
    "methylation": "#009E73",
    "drugresponse": "#D55E00",
    "crisprcas9": "#CC79A7",
    "conditionals": "#666666",
    "copynumber": "#E69F00",
}
TARGET_COLORS = {
    "CRISPR-Cas9": PALETTE["new"],
    "Drug response": PALETTE["lost"],
}


def find_repo_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "reports" / "vae" / "files").exists() and (candidate / "docs").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root containing reports/vae/files and docs.")


ROOT = find_repo_root(Path(__file__).resolve().parent)
FILES_DIR = ROOT / "reports" / "vae" / "files"
COMPARE_DIR = ROOT / "reports" / "model_comparison_cnv_mosa_only" / "feature_augmentation" / TIMESTAMP
FIG_DIR = ROOT / "reports" / "cba2026_claude" / "shap_analysis" / TIMESTAMP
SINGLE_FIG_DIR = FIG_DIR / "singles"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SINGLE_FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def shap_paths(target_family: str) -> dict[str, Path]:
    return {
        "values":        FILES_DIR / f"{TIMESTAMP}_shap_values_{target_family}_mean_abs.csv.gz",
        "feature_rank":  FILES_DIR / f"{TIMESTAMP}_shap_feature_ranking_{target_family}_mean_abs.csv",
        "omic_rank":     FILES_DIR / f"{TIMESTAMP}_shap_omic_ranking_{target_family}_mean_abs.csv",
        "top_features":  FILES_DIR / f"{TIMESTAMP}_shap_values_top_features_{target_family}_mean_abs.feather",
    }


def require_files(paths: list[Path]) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        rel = "\n".join(str(p.relative_to(ROOT)) for p in missing)
        raise FileNotFoundError(f"Missing required input files:\n{rel}")


def load_shap_tables() -> dict[str, dict]:
    """Load every SHAP table once and return a single namespace dict.

    The returned dict has keys:
    - ``feature_rank``    : {family: DataFrame} per-family feature ranking
    - ``omic_rank``       : {family: DataFrame} per-family omic-layer importance
    - ``top_features``    : {family: DataFrame} per-target top-200 feather
    - ``feature_rank_all``: concatenated frame across families
    - ``omic_rank_all``   : concatenated frame across families
    - ``top_features_all``: concatenated frame across families
    """
    required: list[Path] = []
    for fam in TARGET_META:
        required.extend(shap_paths(fam).values())
    require_files(required)

    feature_rank: dict[str, pd.DataFrame] = {}
    omic_rank: dict[str, pd.DataFrame] = {}
    top_features: dict[str, pd.DataFrame] = {}

    for fam, label in TARGET_META.items():
        paths = shap_paths(fam)
        feature_rank[fam] = pd.read_csv(paths["feature_rank"]).assign(
            target_family=fam, target_label=label
        )
        omic_rank[fam] = pd.read_csv(paths["omic_rank"]).assign(
            target_family=fam, target_label=label
        )
        top_features[fam] = pd.read_feather(paths["top_features"]).assign(
            target_family=fam, target_label=label
        )

    return {
        "feature_rank": feature_rank,
        "omic_rank": omic_rank,
        "top_features": top_features,
        "feature_rank_all": pd.concat(feature_rank.values(), ignore_index=True),
        "omic_rank_all": pd.concat(omic_rank.values(), ignore_index=True),
        "top_features_all": pd.concat(top_features.values(), ignore_index=True),
    }


def load_performance() -> pd.DataFrame:
    """Build a per-target performance frame.

    Joins selected TabPFN (sample_frame=expanded, variant=mosa_all) against
    the random-forest overlap/original baseline and returns a tidy DataFrame
    with one row per (family, target) and a Pearson-r delta column.
    """
    require_files([
        COMPARE_DIR / "combined_per_target.csv",
        COMPARE_DIR / "summary_model_comparison.csv",
    ])
    per_target = pd.read_csv(COMPARE_DIR / "combined_per_target.csv")

    tabpfn = (
        per_target[
            (per_target["model_name"] == "tabpfn")
            & (per_target["sample_frame"] == "expanded")
            & (per_target["variant"] == "mosa_all")
        ]
        .rename(columns={
            "target": "target_name",
            "test_pearsonr": "test_pearsonr_tabpfn",
            "test_r2": "test_r2_tabpfn",
            "test_rmse": "test_rmse_tabpfn",
        })
    )
    rf = (
        per_target[
            (per_target["model_name"] == "random_forest")
            & (per_target["sample_frame"] == "overlap")
            & (per_target["variant"] == "original")
        ]
        .rename(columns={
            "target": "target_name",
            "test_pearsonr": "test_pearsonr_rf_baseline",
            "test_r2": "test_r2_rf_baseline",
            "test_rmse": "test_rmse_rf_baseline",
        })
    )
    cols_t = ["target_family", "target_name",
              "test_pearsonr_tabpfn", "test_r2_tabpfn", "test_rmse_tabpfn", "valid_test_n"]
    cols_r = ["target_family", "target_name",
              "test_pearsonr_rf_baseline", "test_r2_rf_baseline", "test_rmse_rf_baseline"]
    perf = tabpfn[cols_t].merge(rf[cols_r], on=["target_family", "target_name"], how="left")
    perf["delta_pearsonr_vs_rf"] = perf["test_pearsonr_tabpfn"] - perf["test_pearsonr_rf_baseline"]
    return perf


# ---------------------------------------------------------------------------
# Profile builders
# ---------------------------------------------------------------------------


def summarize_target_profile(top_features_family: pd.DataFrame) -> pd.DataFrame:
    """Compute target-level SHAP composition from the top-200 export.

    Returns one row per target with omic-layer shares, dominant layer,
    omic-entropy, and counts of copy-number features.
    """
    df = top_features_family.copy()
    family = df["target_family"].iloc[0]
    label = df["target_label"].iloc[0]
    df = df.sort_values(["target_name", "mean_abs_shap"], ascending=[True, False])

    total = df.groupby("target_name")["mean_abs_shap"].sum().rename("top200_total_shap")
    top1 = (df.groupby("target_name").head(1).groupby("target_name")["mean_abs_shap"].sum() / total).rename("top1_share")
    top10 = (df.groupby("target_name").head(10).groupby("target_name")["mean_abs_shap"].sum() / total).rename("top10_share")

    omic = df.groupby(["target_name", "omic_layer"], as_index=False)["mean_abs_shap"].sum()
    omic["share_top200"] = omic["mean_abs_shap"] / omic["target_name"].map(total)
    share = omic.pivot_table(index="target_name", columns="omic_layer", values="share_top200", fill_value=0)

    eps = 1e-12
    n_layers = max(share.shape[1], 1)
    entropy = (-(share * np.log(share + eps)).sum(axis=1) / np.log(n_layers)).rename("omic_entropy")
    dominant = share.idxmax(axis=1).rename("dominant_omic")
    dominant_share = share.max(axis=1).rename("dominant_omic_share")
    cnv_count = (
        df[df["omic_layer"] == "copynumber"]
        .groupby("target_name").size()
        .rename("copynumber_n_top200")
    )

    profile = pd.concat(
        [total, top1, top10, entropy, dominant, dominant_share, cnv_count],
        axis=1,
    ).fillna({"copynumber_n_top200": 0})
    share_named = share.add_prefix("share_").reset_index()
    profile = profile.reset_index().merge(share_named, on="target_name", how="left")
    for layer in OMIC_ORDER:
        col = f"share_{layer}"
        if col not in profile:
            profile[col] = 0.0
    profile["target_family"] = family
    profile["target_label"] = label
    profile["copynumber_n_top200"] = profile["copynumber_n_top200"].astype(int)
    return profile


def assemble_analysis_table(tables: dict, performance: pd.DataFrame) -> pd.DataFrame:
    """Per-target SHAP profile joined with selected TabPFN performance."""
    profile = pd.concat(
        [summarize_target_profile(df) for df in tables["top_features"].values()],
        ignore_index=True,
    )
    return profile.merge(performance, on=["target_family", "target_name"], how="left")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def clean_feature_label(name: str, *, width: int = 28, mark_mut: bool = True) -> str:
    """Drop the omic-layer prefix and turn ``__`` into '/'.

    When ``mark_mut`` is True, conditional mutation features get a leading
    italic-style ``mut`` prefix so their identity is obvious in bar plots.
    """
    name = str(name)
    if "_" in name:
        prefix, body = name.split("_", 1)
    else:
        prefix, body = "", name
    body = body.replace("__", " / ").replace("_", " ")
    if mark_mut and prefix == "conditionals" and body.startswith("mut "):
        body = body[4:]
        body = f"mut {body}"
    return textwrap.shorten(body, width=width, placeholder="…")


def feature_omic_lookup(tables: dict) -> dict:
    return dict(zip(
        tables["feature_rank_all"]["feature"],
        tables["feature_rank_all"]["omic_layer"],
    ))


def omic_legend_handles(layers_present: list[str]) -> list[Patch]:
    layers_present = [l for l in OMIC_ORDER if l in layers_present]
    return [Patch(facecolor=OMIC_COLORS[l], edgecolor="black",
                  linewidth=0.4, label=OMIC_DISPLAY[l]) for l in layers_present]


def stable_sort_omic(values) -> list[str]:
    s = set(values)
    return [o for o in OMIC_ORDER if o in s]


def fmt_sci(value: float) -> str:
    if value == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(value))))
    mant = value / (10 ** exp)
    return f"{mant:.2f}×10$^{{{exp}}}$"


# ---------------------------------------------------------------------------
# Figure 1 — Global omic-layer landscape (4-panel composite)
# ---------------------------------------------------------------------------


def fig1_global_landscape(tables: dict) -> Path:
    """Two-row composite:
    (a) absolute SHAP per omic layer × family,
    (b) within-family share per omic layer,
    (c) top-25 global features for CRISPR-Cas9,
    (d) top-25 global features for drug response.
    """
    configure_nature_style("composite")
    omic = tables["omic_rank_all"].copy()
    omic["share"] = omic["importance"] / omic.groupby("target_family")["importance"].transform("sum")
    omic["omic_label"] = omic["omic_layer"].map(OMIC_DISPLAY)

    fig = plt.figure(figsize=(7.2, 8.8))
    gs = GridSpec(
        nrows=2, ncols=2, figure=fig,
        height_ratios=[1.0, 1.95],
        width_ratios=[1.0, 1.0],
        hspace=0.42, wspace=0.45,
        left=0.10, right=0.98, top=0.96, bottom=0.13,
    )
    ax_abs = fig.add_subplot(gs[0, 0])
    ax_share = fig.add_subplot(gs[0, 1])
    ax_crispr = fig.add_subplot(gs[1, 0])
    ax_drug = fig.add_subplot(gs[1, 1])

    # ---- (a) absolute mean abs SHAP ---------------------------------------
    layers = [l for l in OMIC_ORDER if l in set(omic["omic_layer"])]
    y = np.arange(len(layers))
    bar_h = 0.36
    for offset, fam_label, color in [
        (-bar_h / 2, "CRISPR-Cas9", TARGET_COLORS["CRISPR-Cas9"]),
        (bar_h / 2, "Drug response", TARGET_COLORS["Drug response"]),
    ]:
        d = omic[omic["target_label"] == fam_label].set_index("omic_layer").reindex(layers)
        ax_abs.barh(
            y + offset, d["importance"], height=bar_h,
            color=color, edgecolor="black", linewidth=0.45,
            label=fam_label,
        )
    ax_abs.set_yticks(y)
    ax_abs.set_yticklabels([OMIC_DISPLAY[l] for l in layers])
    ax_abs.invert_yaxis()
    ax_abs.set_xlabel("Mean |SHAP|  (×10$^{-3}$)")
    # Convert ticks to ×10⁻³ so we don't need a sci-notation offset glued to the corner.
    ax_abs.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v * 1e3:.2f}"))
    ax_abs.legend(
        loc="lower right", bbox_to_anchor=(1.0, -0.03),
        frameon=False, handlelength=1.0, borderpad=0.2,
    )
    ax_abs.set_title("Absolute SHAP per omic layer")
    panel_label(ax_abs, "a", offset=(-0.30, 1.05))

    # ---- (b) within-family share ------------------------------------------
    for offset, fam_label, color in [
        (-bar_h / 2, "CRISPR-Cas9", TARGET_COLORS["CRISPR-Cas9"]),
        (bar_h / 2, "Drug response", TARGET_COLORS["Drug response"]),
    ]:
        d = omic[omic["target_label"] == fam_label].set_index("omic_layer").reindex(layers)
        ax_share.barh(
            y + offset, d["share"], height=bar_h,
            color=color, edgecolor="black", linewidth=0.45,
            label=fam_label,
        )
    ax_share.set_yticks(y)
    ax_share.set_yticklabels([OMIC_DISPLAY[l] for l in layers])
    ax_share.invert_yaxis()
    ax_share.set_xlabel("Share of within-family SHAP")
    ax_share.set_xlim(0, max(omic["share"]) * 1.15)
    ax_share.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax_share.legend(
        loc="lower right", bbox_to_anchor=(1.0, -0.03),
        frameon=False, handlelength=1.0, borderpad=0.2,
    )
    ax_share.set_title("Within-family share")
    panel_label(ax_share, "b", offset=(-0.22, 1.05))

    # ---- (c) and (d) top features per family ------------------------------
    layers_present_global = set()
    for ax, (fam, label), letter, offset in zip(
        [ax_crispr, ax_drug],
        TARGET_META.items(),
        ["c", "d"],
        [(-0.30, 1.04), (-0.22, 1.04)],
    ):
        d = (
            tables["feature_rank"][fam]
            .sort_values("importance", ascending=False)
            .head(25)
            .copy()
        )
        d = d.sort_values("importance", ascending=True)
        yy = np.arange(len(d))
        colors = [OMIC_COLORS.get(l, "#999999") for l in d["omic_layer"]]
        layers_present_global.update(d["omic_layer"])
        ax.barh(
            yy, d["importance"] * 1e3,  # show ×10⁻³ on the axis directly
            color=colors, edgecolor="black", linewidth=0.3, height=0.78,
        )
        ax.set_yticks(yy)
        ax.set_yticklabels([clean_feature_label(v, width=24) for v in d["feature"]])
        ax.set_xlabel("Mean |SHAP|  (×10$^{-3}$)")
        ax.set_title(label)
        ax.tick_params(axis="y", length=0, pad=2)
        ax.set_ylim(-0.6, len(d) - 0.4)
        ax.grid(axis="x", color="#e5e5e5", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
        panel_label(ax, letter, offset=offset)

    fig.legend(
        handles=omic_legend_handles(layers_present_global),
        loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 0.015),
        handlelength=1.2, columnspacing=1.8, handletextpad=0.55,
    )

    out = FIG_DIR / "fig1_global_landscape"
    save_figure(fig, out)
    return Path(str(out) + ".pdf")


# ---------------------------------------------------------------------------
# Figure 2 — Top-feature heatmap shared across families
# ---------------------------------------------------------------------------


def fig2_global_feature_heatmap(tables: dict, top_per_family: int = 25) -> Path:
    """Single-panel heatmap of the union of top features across families.

    Layout uses three side-by-side axes via gridspec: a thin omic-layer
    swatch column, the feature × family heatmap, and a vertical colour bar.
    Putting the swatches on a dedicated axis means feature labels cannot
    collide with the layer indicator.
    """
    configure_nature_style("column")
    rank_all = tables["feature_rank_all"]
    feature_union = (
        rank_all.sort_values("importance", ascending=False)
        .groupby("target_family")
        .head(top_per_family)["feature"]
        .drop_duplicates()
    )
    sub = rank_all[rank_all["feature"].isin(feature_union)].copy()
    layer_lookup = sub.drop_duplicates("feature").set_index("feature")["omic_layer"].to_dict()

    wide = sub.pivot_table(
        index="feature", columns="target_label", values="importance",
        aggfunc="first", fill_value=0,
    )
    wide["max_importance"] = wide.max(axis=1)
    wide = wide.sort_values("max_importance", ascending=False).head(40).drop(columns="max_importance")
    wide = wide[["CRISPR-Cas9", "Drug response"]]
    plot_mat = wide.copy() * 1e3   # display in ×10⁻³ units

    fig_height = max(4.8, 0.20 * len(plot_mat) + 1.4)
    fig = plt.figure(figsize=(5.2, fig_height))
    gs = GridSpec(
        nrows=1, ncols=3, figure=fig,
        width_ratios=[0.06, 1.0, 0.05],
        wspace=0.05,
        left=0.40, right=0.86, top=0.95, bottom=0.16,
    )
    ax_swatch = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    # ---- swatch column: one coloured rectangle per feature row ----
    for idx, feature in enumerate(plot_mat.index):
        layer = layer_lookup.get(feature, "conditionals")
        ax_swatch.add_patch(
            mpl.patches.Rectangle(
                (0.05, idx - 0.42),
                width=0.90, height=0.84,
                facecolor=OMIC_COLORS[layer],
                edgecolor="black", linewidth=0.3,
            )
        )
    ax_swatch.set_xlim(0, 1)
    ax_swatch.set_ylim(len(plot_mat) - 0.5, -0.5)
    ax_swatch.set_xticks([])
    ax_swatch.set_yticks(np.arange(len(plot_mat)))
    ax_swatch.set_yticklabels(
        [clean_feature_label(f, width=28) for f in plot_mat.index],
    )
    ax_swatch.tick_params(axis="y", length=0, pad=3)
    for spine in ax_swatch.spines.values():
        spine.set_visible(False)

    # ---- main heatmap ----
    im = ax.imshow(plot_mat.values, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(plot_mat.columns)))
    ax.set_xticklabels(plot_mat.columns)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_xlim(-0.5, len(plot_mat.columns) - 0.5)
    ax.set_ylim(len(plot_mat) - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ---- colour bar ----
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Mean |SHAP|  (×10$^{-3}$)")
    cbar.ax.tick_params(length=2.5, width=0.5)
    cbar.outline.set_linewidth(0.5)

    layers_present = [layer_lookup[f] for f in plot_mat.index]
    fig.legend(
        handles=omic_legend_handles(set(layers_present)),
        loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.55, 0.005),
        handlelength=1.0, columnspacing=1.4, handletextpad=0.5,
        fontsize=plt.rcParams["legend.fontsize"] - 0.5,
    )
    out = FIG_DIR / "fig2_global_feature_heatmap"
    save_figure(fig, out)
    return Path(str(out) + ".pdf")


# ---------------------------------------------------------------------------
# Figure 3 — Target-level omic composition (violin + strip)
# ---------------------------------------------------------------------------


def _violin_strip(ax, df: pd.DataFrame, layers: list[str]) -> None:
    """Draw horizontal violin + jitter for each omic layer.

    Layers without any data are still listed on the y-axis but get an empty
    row, keeping rows aligned across panels.
    """
    rng = np.random.default_rng(42)
    y_pos = np.arange(len(layers))
    for idx, layer in enumerate(layers):
        vals = df.loc[df["omic_layer"] == layer, "share_top200"].values
        if len(vals) >= 5:
            parts = ax.violinplot(
                vals, positions=[idx], vert=False,
                widths=0.78, showmeans=False, showmedians=False, showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(OMIC_COLORS[layer])
                body.set_edgecolor(OMIC_COLORS[layer])
                body.set_alpha(0.30)
        if len(vals):
            jitter = rng.uniform(-0.20, 0.20, size=len(vals))
            ax.scatter(
                vals, idx + jitter,
                s=4.5, color=OMIC_COLORS[layer],
                alpha=0.55, linewidths=0, rasterized=True,
            )
            med = float(np.median(vals))
            ax.plot(
                [med, med], [idx - 0.32, idx + 0.32],
                color="black", linewidth=1.0, solid_capstyle="butt",
            )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([OMIC_DISPLAY[l] for l in layers])
    ax.invert_yaxis()


def fig3_target_omic_composition(tables: dict) -> Path:
    """Per-target distribution of top-200 SHAP share by omic layer."""
    configure_nature_style("composite")
    target_omic = (
        tables["top_features_all"]
        .groupby(["target_family", "target_label", "target_name", "omic_layer"], as_index=False)["mean_abs_shap"]
        .sum()
    )
    total = target_omic.groupby(["target_family", "target_name"])["mean_abs_shap"].transform("sum")
    target_omic["share_top200"] = target_omic["mean_abs_shap"] / total
    layers = stable_sort_omic(target_omic["omic_layer"])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.85, bottom=0.22, wspace=0.07)

    for ax, (fam, label), letter in zip(axes, TARGET_META.items(), ["a", "b"]):
        sub = target_omic[target_omic["target_family"] == fam].copy()
        _violin_strip(ax, sub, layers)
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_xlabel("Share of top-200 SHAP per target")
        ax.set_title(label)
        ax.grid(axis="x", color="#e5e5e5", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
        panel_label(ax, letter, offset=(-0.25 if letter == "a" else -0.05, 1.04))

    # Figure-level caption explaining the dark line marker; placed in the
    # bottom margin so it never lands on top of any violin.
    fig.text(
        0.99, 0.02,
        "Vertical bar inside each violin: per-layer median across targets.",
        ha="right", va="bottom",
        fontsize=plt.rcParams["legend.fontsize"] - 0.5,
        color="#444444",
    )

    out = FIG_DIR / "fig3_target_omic_composition"
    save_figure(fig, out)
    return Path(str(out) + ".pdf")


# ---------------------------------------------------------------------------
# Figure 4 — Performance vs SHAP profile diagnostics
# ---------------------------------------------------------------------------


def fig4_performance_vs_profile(analysis: pd.DataFrame) -> Path:
    """Four-panel diagnostic: performance distribution, CNV share scatter,
    omic-entropy gain scatter, and dominant-omic boxplot."""
    configure_nature_style("composite")
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = GridSpec(
        nrows=2, ncols=2, figure=fig,
        wspace=0.34, hspace=0.55,
        left=0.09, right=0.98, top=0.93, bottom=0.13,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # ----- (a) performance distribution -----
    bins = np.linspace(-0.7, 0.95, 31)
    for label, color in TARGET_COLORS.items():
        vals = analysis.loc[analysis["target_label"] == label, "test_pearsonr_tabpfn"].dropna()
        ax1.hist(
            vals, bins=bins, histtype="step",
            color=color, linewidth=1.4,
            label=f"{label}  (n = {len(vals)})",
            density=True,
        )
    ax1.axvline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    ax1.set_xlabel("Selected TabPFN  Pearson r")
    ax1.set_ylabel("Density")
    ax1.set_title("Performance distribution")
    ax1.legend(loc="upper left", frameon=False, handlelength=1.4, borderaxespad=0.2)
    panel_label(ax1, "a", offset=(-0.21, 1.05))

    # ----- (b) CNV share vs r -----
    for label, color in TARGET_COLORS.items():
        sub = analysis[analysis["target_label"] == label]
        ax2.scatter(
            sub["share_copynumber"], sub["test_pearsonr_tabpfn"],
            s=10, color=color,
            alpha=SCATTER_ALPHA["new"] if label == "CRISPR-Cas9" else SCATTER_ALPHA["lost"],
            linewidths=0, label=label, rasterized=True,
        )
    ax2.set_xlabel("Copy-number share of top-200 SHAP")
    ax2.set_ylabel("Selected TabPFN  Pearson r")
    ax2.set_title("Copy-number attribution")
    ax2.set_xlim(left=-0.005)
    ax2.legend(loc="upper right", frameon=False, handlelength=1.0,
               markerscale=1.4, borderaxespad=0.2)
    panel_label(ax2, "b", offset=(-0.18, 1.05))

    # ----- (c) entropy vs gain -----
    for label, color in TARGET_COLORS.items():
        sub = analysis[analysis["target_label"] == label]
        ax3.scatter(
            sub["omic_entropy"], sub["delta_pearsonr_vs_rf"],
            s=10, color=color,
            alpha=SCATTER_ALPHA["new"] if label == "CRISPR-Cas9" else SCATTER_ALPHA["lost"],
            linewidths=0, label=label, rasterized=True,
        )
    ax3.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    ax3.set_xlabel("Omic entropy in top-200 SHAP")
    ax3.set_ylabel("ΔPearson r  vs RF baseline")
    ax3.set_title("Distributed attribution vs gain")
    panel_label(ax3, "c", offset=(-0.21, 1.05))

    # ----- (d) dominant omic boxplot, family-stratified -----
    layers = stable_sort_omic(analysis["dominant_omic"])
    bar_pos = np.arange(len(layers))
    width = 0.36
    for offset, label, color in [
        (-width / 2, "CRISPR-Cas9", TARGET_COLORS["CRISPR-Cas9"]),
        (width / 2, "Drug response", TARGET_COLORS["Drug response"]),
    ]:
        positions = bar_pos + offset
        sub = analysis[analysis["target_label"] == label]
        data = [
            sub.loc[sub["dominant_omic"] == l, "test_pearsonr_tabpfn"].dropna().values
            for l in layers
        ]
        present = [(p, d) for p, d in zip(positions, data) if len(d) > 0]
        if not present:
            continue
        positions_present = [p for p, _ in present]
        data_present = [d for _, d in present]
        bp = ax4.boxplot(
            data_present, positions=positions_present, widths=width * 0.85,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=1.0),
            whiskerprops=dict(color="black", linewidth=0.6),
            capprops=dict(color="black", linewidth=0.6),
            boxprops=dict(linewidth=0.5),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_edgecolor("black")
            patch.set_alpha(0.6)
    ax4.set_xticks(bar_pos)
    ax4.set_xticklabels([OMIC_DISPLAY[l] for l in layers], rotation=25, ha="right")
    ax4.set_ylabel("Selected TabPFN  Pearson r")
    ax4.set_title("Dominant top-200 layer")
    ax4.grid(axis="y", color="#e5e5e5", linewidth=0.4, zorder=0)
    ax4.set_axisbelow(True)

    handles = [
        Patch(facecolor=TARGET_COLORS["CRISPR-Cas9"], alpha=0.6,
              edgecolor="black", linewidth=0.5, label="CRISPR-Cas9"),
        Patch(facecolor=TARGET_COLORS["Drug response"], alpha=0.6,
              edgecolor="black", linewidth=0.5, label="Drug response"),
    ]
    ax4.legend(handles=handles, loc="upper right",
               frameon=False, handlelength=1.0, borderaxespad=0.2)
    panel_label(ax4, "d", offset=(-0.18, 1.05))

    out = FIG_DIR / "fig4_performance_vs_profile"
    save_figure(fig, out)
    return Path(str(out) + ".pdf")


# ---------------------------------------------------------------------------
# Figure 5 — Selected CRISPR targets, top features
# ---------------------------------------------------------------------------


def fig5_selected_crispr_top_features(
    tables: dict, analysis: pd.DataFrame,
    requested: tuple[str, ...] = ("TP53", "MDM2", "MDM4"),
    top_n: int = 18,
    backfill_with_top_gain: bool = True,
) -> tuple[Path, list[str], list[str]]:
    """Top-N SHAP features for each requested CRISPR target.

    Each panel uses an *independent* x-axis (since per-target SHAP scales
    differ by 5–10×) and shows the scale via a per-panel ×10⁻³ axis label.
    The panel title carries Pearson r and the gain over the RF baseline.

    Returns (path, plotted_targets, missing_targets).
    """
    configure_nature_style("composite")
    fam = "crisprcas9"
    fam_label = TARGET_META[fam]
    perf_lookup = analysis[analysis["target_family"] == fam].set_index("target_name")
    available = set(tables["top_features"][fam]["target_name"].unique())
    plotted_targets = [t for t in requested if t in available]
    missing = [t for t in requested if t not in available]
    if backfill_with_top_gain and missing:
        # Fill the empty slots with the top-Δ-r CRISPR targets so the panel
        # layout stays balanced and biologically informative.
        candidates = (
            perf_lookup.dropna(subset=["delta_pearsonr_vs_rf"])
            .sort_values("delta_pearsonr_vs_rf", ascending=False)
            .index.tolist()
        )
        for cand in candidates:
            if len(plotted_targets) >= len(requested):
                break
            if cand not in plotted_targets:
                plotted_targets.append(cand)
    if not plotted_targets:
        raise ValueError("None of the requested CRISPR targets are available in the SHAP export.")

    n_panels = len(plotted_targets)
    fig_w = 2.45 * n_panels + 0.5
    fig_h = 0.22 * top_n + 1.9
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h))
    if n_panels == 1:
        axes = np.array([axes])
    fig.subplots_adjust(left=0.11, right=0.98, top=0.86, bottom=0.16, wspace=0.95)

    layers_present: set[str] = set()
    rows_export: list[pd.DataFrame] = []

    for idx, (ax, target) in enumerate(zip(axes, plotted_targets)):
        d = (
            tables["top_features"][fam][tables["top_features"][fam]["target_name"] == target]
            .sort_values("mean_abs_shap", ascending=False)
            .head(top_n)
            .copy()
        )
        rows_export.append(d.assign(panel_target=target))
        d = d.sort_values("mean_abs_shap", ascending=True)
        yy = np.arange(len(d))
        colors = [OMIC_COLORS.get(l, "#999999") for l in d["omic_layer"]]
        layers_present.update(d["omic_layer"])

        ax.barh(
            yy, d["mean_abs_shap"] * 1e3,
            color=colors, edgecolor="black", linewidth=0.3,
            height=0.78,
        )
        ax.set_yticks(yy)
        ax.set_yticklabels(
            [clean_feature_label(v, width=22) for v in d["omics_feature"]],
        )
        ax.set_ylim(-0.6, len(d) - 0.4)
        ax.tick_params(axis="y", length=0, pad=2)
        row = perf_lookup.loc[target]
        title = (
            f"{target}\n"
            f"r = {row['test_pearsonr_tabpfn']:.2f}    "
            f"Δr = {row['delta_pearsonr_vs_rf']:+.2f}"
        )
        ax.set_title(title, loc="left", fontsize=plt.rcParams["axes.titlesize"] - 1)
        ax.set_xlabel("Mean |SHAP|  (×10$^{-3}$)")
        ax.grid(axis="x", color="#e5e5e5", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
        # Bump x-axis a touch so longest bar tip never touches the spine.
        xmax = d["mean_abs_shap"].max() * 1e3
        ax.set_xlim(0, xmax * 1.08)
        panel_label(ax, "abcdef"[idx], offset=(-0.55, 1.06))

    fig.legend(
        handles=omic_legend_handles(layers_present),
        loc="lower center", ncol=min(4, len(layers_present)),
        frameon=False, bbox_to_anchor=(0.5, 0.005),
        handlelength=1.2, columnspacing=1.6, handletextpad=0.55,
    )
    if missing:
        fig.text(
            0.01, 0.005,
            "Requested but unavailable in SHAP export: " + ", ".join(missing),
            ha="left", va="bottom",
            color="#666666",
            fontsize=plt.rcParams["legend.fontsize"] - 0.5,
        )
    out = FIG_DIR / "fig5_selected_crispr_top_features"
    save_figure(fig, out)

    panel_df = pd.concat(rows_export, ignore_index=True)
    panel_df.to_csv(FIG_DIR / "selected_crispr_top_features.csv", index=False)
    return Path(str(out) + ".pdf"), plotted_targets, missing


# ---------------------------------------------------------------------------
# Figure 6 — Selected CRISPR targets, omic composition heatmap
# ---------------------------------------------------------------------------


def fig6_selected_crispr_omic_heatmap(
    analysis: pd.DataFrame,
    targets: list[str] | None = None,
    requested: tuple[str, ...] = ("TP53", "MDM2", "MDM4"),
) -> Path:
    """Compact heatmap of top-200 SHAP omic-share for the selected targets.

    If ``targets`` is provided, those rows are used (in order). Otherwise we
    fall back to whichever of the ``requested`` targets are available in
    the analysis frame.
    """
    configure_nature_style("composite")
    fam_label = TARGET_META["crisprcas9"]
    available_names = set(analysis.loc[analysis["target_label"] == fam_label, "target_name"].unique())
    if targets is None:
        targets = [t for t in requested if t in available_names]
    if not targets:
        raise ValueError("No requested CRISPR targets available for the omic-composition heatmap.")

    sub = (
        analysis[(analysis["target_label"] == fam_label)
                 & (analysis["target_name"].isin(targets))]
        .set_index("target_name")
        .reindex(targets)
    )
    layers = OMIC_ORDER
    mat = sub[[f"share_{l}" for l in layers]].fillna(0)
    mat.columns = [OMIC_DISPLAY[l] for l in layers]

    fig_w = 6.6
    fig_h = max(3.4, 0.55 * len(targets) + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0.16, right=0.76, top=0.87, bottom=0.30)

    cmap = plt.colormaps.get_cmap("magma")
    im = ax.imshow(mat.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(mat)))
    ax.set_yticklabels(mat.index)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=0, pad=4)

    # Magma luminance: low values are dark → use white text; high values
    # are pale yellow → use black text. Threshold ≈ 0.55.
    luminance_threshold = 0.55
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            ax.text(
                j, i, f"{v:.2f}",
                ha="center", va="center",
                color="white" if v < luminance_threshold else "black",
                fontsize=plt.rcParams["xtick.labelsize"] - 0.5,
            )

    cax = fig.add_axes([0.79, 0.30, 0.025, 0.57])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Top-200 SHAP share")
    cbar.ax.tick_params(length=2.5, width=0.5)
    cbar.outline.set_linewidth(0.5)
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0, decimals=0))

    ax.set_title(f"{fam_label} — top-200 SHAP composition", loc="left")
    out = FIG_DIR / "fig6_selected_crispr_omic_composition"
    save_figure(fig, out)
    return Path(str(out) + ".pdf")


# ---------------------------------------------------------------------------
# Figure 7 (NEW) — CNV gain volcano
# ---------------------------------------------------------------------------


def _place_callout_labels(
    ax,
    points: pd.DataFrame,
    *,
    name_col: str = "target_name",
    x_col: str,
    y_col: str,
    column_axfrac: float,
    y_top_axfrac: float,
    y_bottom_axfrac: float,
    fontsize: float,
    line_color: str = "#888888",
):
    """Place labels in a vertical callout column on the axes.

    The column is positioned by axes-relative coordinates (so it doesn't
    drift with x-axis range), and the labels are stacked between
    *y_top_axfrac* and *y_bottom_axfrac* so each one gets the same vertical
    slot regardless of how clustered the source y-values are. A thin leader
    line connects every marker to its label.
    """
    pts = points.sort_values(y_col, ascending=False).reset_index(drop=True)
    n = len(pts)
    if n == 0:
        return
    if n == 1:
        ax_label_y = np.array([(y_top_axfrac + y_bottom_axfrac) / 2.0])
    else:
        ax_label_y = np.linspace(y_top_axfrac, y_bottom_axfrac, n)

    # Convert axes-relative label positions to data y coords for the
    # leader-line endpoint and the text anchor.
    inv = ax.transAxes.transform
    fig = ax.figure
    fig.canvas.draw()
    for axfrac_y, (_, row) in zip(ax_label_y, pts.iterrows()):
        # Map axes-fraction coords through the axes transform pair.
        display_xy = ax.transAxes.transform((column_axfrac, axfrac_y))
        data_xy = ax.transData.inverted().transform(display_xy)
        lx, ly = float(data_xy[0]), float(data_xy[1])
        ax.plot(
            [row[x_col], lx], [row[y_col], ly],
            color=line_color, linewidth=0.5, alpha=0.7, zorder=4,
        )
        ax.text(
            lx, ly, row[name_col],
            fontsize=fontsize, ha="left", va="center",
            color="black", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85),
        )


def fig7_cnv_gain_volcano(analysis: pd.DataFrame, label_top: int = 10) -> Path:
    """Volcano-style plot: ΔPearson-r vs CNV share, separated by family.

    Highlights the targets where adding the +CNV MOSA feature set both
    increased CNV-attribution share and improved performance over the RF
    baseline. Top gainers are connected to a callout column on the right
    of each panel so labels never overlap with each other or with markers.
    """
    configure_nature_style("composite")
    fig, axes = plt.subplots(
        1, 2, figsize=(8.2, 4.0), sharey=True,
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.18, wspace=0.10)

    label_color = {"CRISPR-Cas9": TARGET_COLORS["CRISPR-Cas9"],
                   "Drug response": TARGET_COLORS["Drug response"]}

    for ax, (fam, label), letter in zip(axes, TARGET_META.items(), ["a", "b"]):
        sub = analysis[analysis["target_family"] == fam].dropna(
            subset=["share_copynumber", "delta_pearsonr_vs_rf", "test_pearsonr_tabpfn"]
        ).copy()

        # Panel-specific x-axis range; pad right so callout column has space.
        data_xmax = max(sub["share_copynumber"].max(), 0.02)
        ax.set_xlim(-data_xmax * 0.025, data_xmax * 1.55)

        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
        ax.scatter(
            sub["share_copynumber"], sub["delta_pearsonr_vs_rf"],
            s=14, color=label_color[label],
            alpha=SCATTER_ALPHA["new"] if label == "CRISPR-Cas9" else SCATTER_ALPHA["lost"],
            linewidths=0, rasterized=True,
        )
        winners = sub.nlargest(label_top, "delta_pearsonr_vs_rf")
        ax.scatter(
            winners["share_copynumber"], winners["delta_pearsonr_vs_rf"],
            s=22, facecolor="none", edgecolor=PALETTE["highlight"],
            linewidths=0.9, zorder=5,
        )

        ax.set_xlabel("Copy-number share of top-200 SHAP")
        if letter == "a":
            ax.set_ylabel("ΔPearson r  vs RF baseline")
        ax.set_title(label)
        ax.grid(color="#e5e5e5", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
        # Force the y-limits before computing axes-relative label positions.
        cur_ymin, cur_ymax = ax.get_ylim()
        # Single corner annotation summarising the highlight rule.
        ax.text(
            0.02, 0.04,
            f"Outlined / labelled: top {label_top} ΔPearson r",
            transform=ax.transAxes,
            ha="left", va="bottom",
            color=PALETTE["highlight"],
            fontsize=plt.rcParams["legend.fontsize"] - 0.5,
        )
        panel_label(ax, letter, offset=(-0.16 if letter == "a" else -0.05, 1.04))

        _place_callout_labels(
            ax, winners,
            x_col="share_copynumber", y_col="delta_pearsonr_vs_rf",
            column_axfrac=0.72,
            y_top_axfrac=0.97,
            y_bottom_axfrac=0.40,
            fontsize=plt.rcParams["xtick.labelsize"] - 0.5,
        )

    out = FIG_DIR / "fig7_cnv_gain_volcano"
    save_figure(fig, out)
    return Path(str(out) + ".pdf")


# ---------------------------------------------------------------------------
# Figure 8 (NEW) — SHAP concentration / Lorenz curves
# ---------------------------------------------------------------------------


def fig8_shap_concentration(tables: dict) -> Path:
    """Lorenz-style cumulative SHAP curves per target family.

    For every target we sort top-200 features by mean |SHAP| in descending
    order and compute the cumulative fraction of total SHAP. Plotting the
    median curve plus a 10–90% band makes it easy to see how concentrated
    attribution is, and how much a typical target's signal lives in the
    head vs the tail of the ranking.
    """
    configure_nature_style("composite")
    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.4), sharex=True, sharey=True,
    )
    fig.subplots_adjust(left=0.10, right=0.99, top=0.88, bottom=0.20, wspace=0.08)

    for ax, (fam, label), letter in zip(axes, TARGET_META.items(), ["a", "b"]):
        df = tables["top_features"][fam].copy()
        df = df.sort_values(["target_name", "mean_abs_shap"], ascending=[True, False])
        per_target_total = df.groupby("target_name")["mean_abs_shap"].transform("sum")
        df["frac_of_total"] = df["mean_abs_shap"] / per_target_total
        df["rank"] = df.groupby("target_name").cumcount() + 1
        df["cum"] = df.groupby("target_name")["frac_of_total"].cumsum()

        # Aggregate across targets at each integer rank.
        max_rank = int(df["rank"].max())
        ranks = np.arange(1, max_rank + 1)
        wide = df.pivot_table(index="target_name", columns="rank", values="cum")
        wide = wide.reindex(columns=ranks).ffill(axis=1)
        median = wide.median(axis=0).values
        q10 = wide.quantile(0.10, axis=0).values
        q90 = wide.quantile(0.90, axis=0).values

        ax.fill_between(
            ranks, q10, q90,
            color=TARGET_COLORS[label], alpha=0.18, linewidth=0,
        )
        ax.plot(
            ranks, median,
            color=TARGET_COLORS[label], linewidth=1.6,
            label=f"{label} median (n={wide.shape[0]})",
        )
        ax.plot(ranks, q10, color=TARGET_COLORS[label],
                linewidth=0.6, linestyle="--", alpha=0.7)
        ax.plot(ranks, q90, color=TARGET_COLORS[label],
                linewidth=0.6, linestyle="--", alpha=0.7)

        # Reference: top-10 share marker.
        top10_med = float(np.interp(10, ranks, median))
        ax.plot([10, 10], [0, top10_med], color=PALETTE["highlight"],
                linewidth=0.8, linestyle=":")
        ax.plot([0, 10], [top10_med, top10_med], color=PALETTE["highlight"],
                linewidth=0.8, linestyle=":")
        ax.scatter([10], [top10_med], s=18, color=PALETTE["highlight"],
                   zorder=5, linewidths=0)
        ax.text(
            0.98, 0.05,
            f"Top-10 captures {top10_med * 100:.0f}% of top-200 SHAP\n"
            f"(median across {wide.shape[0]} targets)",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=plt.rcParams["legend.fontsize"] - 0.5,
            color="#222222",
        )

        ax.set_xlim(1, max_rank)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Feature rank within top-200")
        if letter == "a":
            ax.set_ylabel("Cumulative share of top-200 SHAP")
        ax.set_title(label)
        ax.grid(color="#e5e5e5", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
        panel_label(ax, letter, offset=(-0.18 if letter == "a" else -0.05, 1.04))

    out = FIG_DIR / "fig8_shap_concentration"
    save_figure(fig, out)
    return Path(str(out) + ".pdf")


# ---------------------------------------------------------------------------
# Tables (mirroring original notebooks)
# ---------------------------------------------------------------------------


def write_tables(tables: dict, analysis: pd.DataFrame) -> dict[str, Path]:
    """Re-emit the original tables and the new analysis frame."""
    rank_all = tables["feature_rank_all"]
    top_all = tables["top_features_all"]

    cnv_global = (
        rank_all[rank_all["omic_layer"] == "copynumber"]
        .sort_values(["target_family", "importance"], ascending=[True, False])
        .groupby("target_family")
        .head(25)
        .reset_index(drop=True)
    )
    cnv_hits = (
        top_all[top_all["omic_layer"] == "copynumber"]
        .groupby(["target_family", "target_label", "omics_feature"], as_index=False)
        .agg(
            n_targets=("target_name", "nunique"),
            mean_importance=("mean_abs_shap", "mean"),
            max_importance=("mean_abs_shap", "max"),
        )
        .sort_values(["target_family", "n_targets", "mean_importance"],
                     ascending=[True, False, False])
    )

    target_omic = (
        top_all.groupby(["target_family", "target_label", "target_name", "omic_layer"], as_index=False)["mean_abs_shap"]
        .sum()
    )
    total = target_omic.groupby(["target_family", "target_name"])["mean_abs_shap"].transform("sum")
    target_omic["share_top200"] = target_omic["mean_abs_shap"] / total
    cnv_target_share = (
        target_omic[target_omic["omic_layer"] == "copynumber"]
        .sort_values(["target_family", "share_top200"], ascending=[True, False])
    )

    paths: dict[str, Path] = {}
    for name, df in [
        ("table_global_copynumber_feature_ranking", cnv_global),
        ("table_copynumber_feature_frequency_top200", cnv_hits),
        ("table_target_copynumber_share_top200", cnv_target_share),
        ("target_level_shap_performance_summary", analysis),
    ]:
        path = FIG_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        paths[name] = path

    cnv_top = (
        analysis.sort_values(["target_family", "share_copynumber"], ascending=[True, False])
        .groupby("target_family")
        .head(25)
    )
    cnv_top.to_csv(FIG_DIR / "targets_high_copynumber_shap_share.csv", index=False)
    paths["targets_high_copynumber_shap_share"] = FIG_DIR / "targets_high_copynumber_shap_share.csv"

    gains: list[pd.DataFrame] = []
    for fam in TARGET_META:
        fam_df = analysis[analysis["target_family"] == fam]
        gains.append(fam_df.nlargest(20, "delta_pearsonr_vs_rf").assign(direction="top_gain"))
        gains.append(fam_df.nsmallest(20, "delta_pearsonr_vs_rf").assign(direction="top_loss"))
    gains_df = pd.concat(gains, ignore_index=True)
    gains_df.to_csv(FIG_DIR / "targets_top_gain_loss_vs_rf_baseline.csv", index=False)
    paths["targets_top_gain_loss_vs_rf_baseline"] = FIG_DIR / "targets_top_gain_loss_vs_rf_baseline.csv"
    return paths


# ---------------------------------------------------------------------------
# Single-panel renderers
# ---------------------------------------------------------------------------
#
# Every composite figure has a "singles" counterpart that emits the same
# panels as standalone slide-friendly PNG + PDF files. The singles use the
# `column` type scale (one panel ≈ 89 mm wide, 11 / 10 / 8.5 pt) so that
# typography stays correct when a single panel is shown at a larger size
# than its slot in the composite.


def _save_single(fig, stem: str) -> Path:
    """Helper: save into ``SINGLE_FIG_DIR`` and return the PDF path."""
    out = SINGLE_FIG_DIR / stem
    save_figure(fig, out)
    return Path(str(out) + ".pdf")


def _draw_omic_paired_bars(
    ax, omic: pd.DataFrame, value_col: str, *,
    layers: list[str],
    legend_loc: str = "lower right",
) -> None:
    y = np.arange(len(layers))
    bar_h = 0.36
    for offset, fam_label, color in [
        (-bar_h / 2, "CRISPR-Cas9", TARGET_COLORS["CRISPR-Cas9"]),
        (bar_h / 2, "Drug response", TARGET_COLORS["Drug response"]),
    ]:
        d = omic[omic["target_label"] == fam_label].set_index("omic_layer").reindex(layers)
        ax.barh(
            y + offset, d[value_col], height=bar_h,
            color=color, edgecolor="black", linewidth=0.45,
            label=fam_label,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([OMIC_DISPLAY[l] for l in layers])
    ax.invert_yaxis()
    ax.legend(
        loc=legend_loc, frameon=False,
        handlelength=1.0, borderpad=0.2,
    )


def single_fig1a_omic_abs(tables: dict) -> Path:
    configure_nature_style("column")
    omic = tables["omic_rank_all"].copy()
    layers = [l for l in OMIC_ORDER if l in set(omic["omic_layer"])]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    fig.subplots_adjust(left=0.28, right=0.96, top=0.92, bottom=0.18)
    _draw_omic_paired_bars(ax, omic, "importance", layers=layers)
    ax.set_xlabel("Mean |SHAP|  (×10$^{-3}$)")
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v * 1e3:.2f}"))
    ax.set_title("Absolute SHAP per omic layer")
    return _save_single(fig, "single_fig1a_omic_absolute")


def single_fig1b_omic_share(tables: dict) -> Path:
    configure_nature_style("column")
    omic = tables["omic_rank_all"].copy()
    omic["share"] = omic["importance"] / omic.groupby("target_family")["importance"].transform("sum")
    layers = [l for l in OMIC_ORDER if l in set(omic["omic_layer"])]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    fig.subplots_adjust(left=0.28, right=0.96, top=0.92, bottom=0.18)
    _draw_omic_paired_bars(ax, omic, "share", layers=layers)
    ax.set_xlabel("Share of within-family SHAP")
    ax.set_xlim(0, max(omic["share"]) * 1.15)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title("Within-family share")
    return _save_single(fig, "single_fig1b_omic_share")


def _draw_top_features_bar(ax, d: pd.DataFrame, *, title: str) -> set[str]:
    """Horizontal bar of top-N features coloured by omic layer."""
    d = d.sort_values("importance", ascending=True)
    yy = np.arange(len(d))
    colors = [OMIC_COLORS.get(l, "#999999") for l in d["omic_layer"]]
    ax.barh(
        yy, d["importance"] * 1e3,
        color=colors, edgecolor="black", linewidth=0.3, height=0.78,
    )
    ax.set_yticks(yy)
    ax.set_yticklabels([clean_feature_label(v, width=24) for v in d["feature"]])
    ax.set_xlabel("Mean |SHAP|  (×10$^{-3}$)")
    ax.set_title(title)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylim(-0.6, len(d) - 0.4)
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    return set(d["omic_layer"])


def _single_top_features_panel(tables: dict, family: str, label: str, stem: str) -> Path:
    configure_nature_style("column")
    d = (
        tables["feature_rank"][family]
        .sort_values("importance", ascending=False)
        .head(25)
        .copy()
    )
    fig, ax = plt.subplots(figsize=(4.4, 6.4))
    fig.subplots_adjust(left=0.36, right=0.97, top=0.94, bottom=0.18)
    layers_present = _draw_top_features_bar(ax, d, title=label)
    fig.legend(
        handles=omic_legend_handles(layers_present),
        loc="lower center", ncol=min(3, len(layers_present)),
        frameon=False, bbox_to_anchor=(0.5, 0.005),
        handlelength=1.0, columnspacing=1.4, handletextpad=0.5,
    )
    return _save_single(fig, stem)


def single_fig1c_top_features_crispr(tables: dict) -> Path:
    return _single_top_features_panel(
        tables, "crisprcas9", "CRISPR-Cas9",
        "single_fig1c_top_features_crispr",
    )


def single_fig1d_top_features_drug(tables: dict) -> Path:
    return _single_top_features_panel(
        tables, "drugresponse", "Drug response",
        "single_fig1d_top_features_drug",
    )


def single_fig2_global_feature_heatmap(tables: dict, top_per_family: int = 25) -> Path:
    """Standalone version of Fig 2 — already single-panel in the composite."""
    # Reuse the composite renderer but redirect the output path.
    rank_all = tables["feature_rank_all"]
    feature_union = (
        rank_all.sort_values("importance", ascending=False)
        .groupby("target_family")
        .head(top_per_family)["feature"]
        .drop_duplicates()
    )
    sub = rank_all[rank_all["feature"].isin(feature_union)].copy()
    layer_lookup = sub.drop_duplicates("feature").set_index("feature")["omic_layer"].to_dict()
    wide = sub.pivot_table(
        index="feature", columns="target_label", values="importance",
        aggfunc="first", fill_value=0,
    )
    wide["max_importance"] = wide.max(axis=1)
    wide = wide.sort_values("max_importance", ascending=False).head(40).drop(columns="max_importance")
    wide = wide[["CRISPR-Cas9", "Drug response"]]
    plot_mat = wide.copy() * 1e3

    configure_nature_style("column")
    fig_h = max(5.0, 0.20 * len(plot_mat) + 1.4)
    fig = plt.figure(figsize=(5.4, fig_h))
    gs = GridSpec(
        nrows=1, ncols=3, figure=fig,
        width_ratios=[0.06, 1.0, 0.05],
        wspace=0.05,
        left=0.40, right=0.86, top=0.95, bottom=0.16,
    )
    ax_swatch = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    for idx, feature in enumerate(plot_mat.index):
        layer = layer_lookup.get(feature, "conditionals")
        ax_swatch.add_patch(mpl.patches.Rectangle(
            (0.05, idx - 0.42), 0.90, 0.84,
            facecolor=OMIC_COLORS[layer], edgecolor="black", linewidth=0.3,
        ))
    ax_swatch.set_xlim(0, 1)
    ax_swatch.set_ylim(len(plot_mat) - 0.5, -0.5)
    ax_swatch.set_xticks([])
    ax_swatch.set_yticks(np.arange(len(plot_mat)))
    ax_swatch.set_yticklabels([clean_feature_label(f, width=28) for f in plot_mat.index])
    ax_swatch.tick_params(axis="y", length=0, pad=3)
    for spine in ax_swatch.spines.values():
        spine.set_visible(False)

    im = ax.imshow(plot_mat.values, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(plot_mat.columns)))
    ax.set_xticklabels(plot_mat.columns)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_xlim(-0.5, len(plot_mat.columns) - 0.5)
    ax.set_ylim(len(plot_mat) - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Mean |SHAP|  (×10$^{-3}$)")
    cbar.ax.tick_params(length=2.5, width=0.5)
    cbar.outline.set_linewidth(0.5)

    layers_present = [layer_lookup[f] for f in plot_mat.index]
    fig.legend(
        handles=omic_legend_handles(set(layers_present)),
        loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.55, 0.005),
        handlelength=1.0, columnspacing=1.4, handletextpad=0.5,
        fontsize=plt.rcParams["legend.fontsize"] - 0.5,
    )
    return _save_single(fig, "single_fig2_global_feature_heatmap")


def _draw_violin_strip_panel(ax, sub: pd.DataFrame, layers: list[str], title: str) -> None:
    _violin_strip(ax, sub, layers)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Share of top-200 SHAP per target")
    ax.set_title(title)
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


def _single_omic_composition(tables: dict, family: str, label: str, stem: str) -> Path:
    configure_nature_style("column")
    target_omic = (
        tables["top_features_all"]
        .groupby(["target_family", "target_label", "target_name", "omic_layer"], as_index=False)["mean_abs_shap"]
        .sum()
    )
    total = target_omic.groupby(["target_family", "target_name"])["mean_abs_shap"].transform("sum")
    target_omic["share_top200"] = target_omic["mean_abs_shap"] / total
    layers = stable_sort_omic(target_omic["omic_layer"])
    sub = target_omic[target_omic["target_family"] == family].copy()

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    fig.subplots_adjust(left=0.28, right=0.97, top=0.90, bottom=0.20)
    _draw_violin_strip_panel(ax, sub, layers, label)
    fig.text(
        0.99, 0.02,
        "Vertical bar inside each violin: per-layer median across targets.",
        ha="right", va="bottom",
        fontsize=plt.rcParams["legend.fontsize"] - 0.5,
        color="#444444",
    )
    return _save_single(fig, stem)


def single_fig3a_composition_crispr(tables: dict) -> Path:
    return _single_omic_composition(tables, "crisprcas9", "CRISPR-Cas9",
                                     "single_fig3a_composition_crispr")


def single_fig3b_composition_drug(tables: dict) -> Path:
    return _single_omic_composition(tables, "drugresponse", "Drug response",
                                     "single_fig3b_composition_drug")


def single_fig4a_performance_distribution(analysis: pd.DataFrame) -> Path:
    configure_nature_style("column")
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.92, bottom=0.18)
    bins = np.linspace(-0.7, 0.95, 31)
    for label, color in TARGET_COLORS.items():
        vals = analysis.loc[analysis["target_label"] == label, "test_pearsonr_tabpfn"].dropna()
        ax.hist(
            vals, bins=bins, histtype="step",
            color=color, linewidth=1.4,
            label=f"{label}  (n = {len(vals)})",
            density=True,
        )
    ax.axvline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_xlabel("Selected TabPFN  Pearson r")
    ax.set_ylabel("Density")
    ax.set_title("Performance distribution")
    ax.legend(loc="upper left", frameon=False, handlelength=1.4, borderaxespad=0.2)
    return _save_single(fig, "single_fig4a_performance_distribution")


def single_fig4b_cnv_share_scatter(analysis: pd.DataFrame) -> Path:
    configure_nature_style("column")
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.92, bottom=0.18)
    for label, color in TARGET_COLORS.items():
        sub = analysis[analysis["target_label"] == label]
        ax.scatter(
            sub["share_copynumber"], sub["test_pearsonr_tabpfn"],
            s=12, color=color,
            alpha=SCATTER_ALPHA["new"] if label == "CRISPR-Cas9" else SCATTER_ALPHA["lost"],
            linewidths=0, label=label, rasterized=True,
        )
    ax.set_xlabel("Copy-number share of top-200 SHAP")
    ax.set_ylabel("Selected TabPFN  Pearson r")
    ax.set_title("Copy-number attribution vs performance")
    ax.set_xlim(left=-0.005)
    ax.legend(loc="upper right", frameon=False, handlelength=1.0,
              markerscale=1.4, borderaxespad=0.2)
    return _save_single(fig, "single_fig4b_cnv_share_scatter")


def single_fig4c_entropy_gain_scatter(analysis: pd.DataFrame) -> Path:
    configure_nature_style("column")
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.92, bottom=0.18)
    for label, color in TARGET_COLORS.items():
        sub = analysis[analysis["target_label"] == label]
        ax.scatter(
            sub["omic_entropy"], sub["delta_pearsonr_vs_rf"],
            s=12, color=color,
            alpha=SCATTER_ALPHA["new"] if label == "CRISPR-Cas9" else SCATTER_ALPHA["lost"],
            linewidths=0, label=label, rasterized=True,
        )
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_xlabel("Omic entropy in top-200 SHAP")
    ax.set_ylabel("ΔPearson r  vs RF baseline")
    ax.set_title("Distributed attribution vs gain")
    ax.legend(loc="upper right", frameon=False, handlelength=1.0,
              markerscale=1.4, borderaxespad=0.2)
    return _save_single(fig, "single_fig4c_entropy_gain_scatter")


def single_fig4d_dominant_omic_box(analysis: pd.DataFrame) -> Path:
    configure_nature_style("column")
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    fig.subplots_adjust(left=0.18, right=0.97, top=0.92, bottom=0.30)
    layers = stable_sort_omic(analysis["dominant_omic"])
    bar_pos = np.arange(len(layers))
    width = 0.36
    for offset, label, color in [
        (-width / 2, "CRISPR-Cas9", TARGET_COLORS["CRISPR-Cas9"]),
        (width / 2, "Drug response", TARGET_COLORS["Drug response"]),
    ]:
        positions = bar_pos + offset
        sub = analysis[analysis["target_label"] == label]
        data = [
            sub.loc[sub["dominant_omic"] == l, "test_pearsonr_tabpfn"].dropna().values
            for l in layers
        ]
        present = [(p, d) for p, d in zip(positions, data) if len(d) > 0]
        if not present:
            continue
        bp = ax.boxplot(
            [d for _, d in present],
            positions=[p for p, _ in present],
            widths=width * 0.85,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=1.0),
            whiskerprops=dict(color="black", linewidth=0.6),
            capprops=dict(color="black", linewidth=0.6),
            boxprops=dict(linewidth=0.5),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_edgecolor("black")
            patch.set_alpha(0.6)
    ax.set_xticks(bar_pos)
    ax.set_xticklabels([OMIC_DISPLAY[l] for l in layers], rotation=25, ha="right")
    ax.set_ylabel("Selected TabPFN  Pearson r")
    ax.set_title("Dominant top-200 layer")
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    handles = [
        Patch(facecolor=TARGET_COLORS["CRISPR-Cas9"], alpha=0.6,
              edgecolor="black", linewidth=0.5, label="CRISPR-Cas9"),
        Patch(facecolor=TARGET_COLORS["Drug response"], alpha=0.6,
              edgecolor="black", linewidth=0.5, label="Drug response"),
    ]
    ax.legend(handles=handles, loc="upper left",
              frameon=False, handlelength=1.0, borderaxespad=0.2)
    return _save_single(fig, "single_fig4d_dominant_omic_box")


def _draw_target_top_features(ax, d: pd.DataFrame, target: str, perf_row, top_n: int) -> set[str]:
    d = d.sort_values("mean_abs_shap", ascending=False).head(top_n)
    d = d.sort_values("mean_abs_shap", ascending=True)
    yy = np.arange(len(d))
    colors = [OMIC_COLORS.get(l, "#999999") for l in d["omic_layer"]]
    ax.barh(
        yy, d["mean_abs_shap"] * 1e3,
        color=colors, edgecolor="black", linewidth=0.3, height=0.78,
    )
    ax.set_yticks(yy)
    ax.set_yticklabels([clean_feature_label(v, width=22) for v in d["omics_feature"]])
    ax.set_ylim(-0.6, len(d) - 0.4)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlabel("Mean |SHAP|  (×10$^{-3}$)")
    title = (
        f"{target}    "
        f"r = {perf_row['test_pearsonr_tabpfn']:.2f}    "
        f"Δr = {perf_row['delta_pearsonr_vs_rf']:+.2f}"
    )
    ax.set_title(title)
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    xmax = d["mean_abs_shap"].max() * 1e3
    ax.set_xlim(0, xmax * 1.08)
    return set(d["omic_layer"])


def render_singles_fig5_per_target(
    tables: dict, analysis: pd.DataFrame, plotted_targets: list[str],
    top_n: int = 18,
) -> list[Path]:
    """One standalone figure per requested CRISPR target."""
    configure_nature_style("column")
    fam = "crisprcas9"
    perf_lookup = analysis[analysis["target_family"] == fam].set_index("target_name")
    out_paths: list[Path] = []
    for target in plotted_targets:
        d = tables["top_features"][fam][tables["top_features"][fam]["target_name"] == target]
        fig, ax = plt.subplots(figsize=(4.6, 0.22 * top_n + 1.6))
        fig.subplots_adjust(left=0.32, right=0.97, top=0.90, bottom=0.20)
        layers_present = _draw_target_top_features(ax, d, target, perf_lookup.loc[target], top_n)
        fig.legend(
            handles=omic_legend_handles(layers_present),
            loc="lower center", ncol=min(4, len(layers_present)),
            frameon=False, bbox_to_anchor=(0.5, 0.005),
            handlelength=1.0, columnspacing=1.4, handletextpad=0.5,
        )
        out_paths.append(_save_single(fig, f"single_fig5_top_features_{target}"))
    return out_paths


def single_fig6_selected_crispr_omic_heatmap(
    analysis: pd.DataFrame, targets: list[str],
) -> Path:
    configure_nature_style("column")
    fam_label = TARGET_META["crisprcas9"]
    sub = (
        analysis[(analysis["target_label"] == fam_label)
                 & (analysis["target_name"].isin(targets))]
        .set_index("target_name").reindex(targets)
    )
    layers = OMIC_ORDER
    mat = sub[[f"share_{l}" for l in layers]].fillna(0)
    mat.columns = [OMIC_DISPLAY[l] for l in layers]

    fig_h = max(3.4, 0.55 * len(targets) + 2.2)
    fig, ax = plt.subplots(figsize=(6.4, fig_h))
    fig.subplots_adjust(left=0.16, right=0.78, top=0.88, bottom=0.30)
    cmap = plt.colormaps.get_cmap("magma")
    im = ax.imshow(mat.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(mat)))
    ax.set_yticklabels(mat.index)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=0, pad=4)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            ax.text(
                j, i, f"{v:.2f}",
                ha="center", va="center",
                color="white" if v < 0.55 else "black",
                fontsize=plt.rcParams["xtick.labelsize"] - 0.5,
            )
    cax = fig.add_axes([0.81, 0.30, 0.025, 0.58])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Top-200 SHAP share")
    cbar.ax.tick_params(length=2.5, width=0.5)
    cbar.outline.set_linewidth(0.5)
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title(f"{fam_label} — top-200 SHAP composition", loc="left")
    return _save_single(fig, "single_fig6_selected_crispr_omic_composition")


def _draw_volcano_panel(ax, sub: pd.DataFrame, label: str, label_top: int) -> None:
    color = TARGET_COLORS[label]
    data_xmax = max(sub["share_copynumber"].max(), 0.02)
    ax.set_xlim(-data_xmax * 0.025, data_xmax * 1.55)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.scatter(
        sub["share_copynumber"], sub["delta_pearsonr_vs_rf"],
        s=14, color=color,
        alpha=SCATTER_ALPHA["new"] if label == "CRISPR-Cas9" else SCATTER_ALPHA["lost"],
        linewidths=0, rasterized=True,
    )
    winners = sub.nlargest(label_top, "delta_pearsonr_vs_rf")
    ax.scatter(
        winners["share_copynumber"], winners["delta_pearsonr_vs_rf"],
        s=22, facecolor="none", edgecolor=PALETTE["highlight"],
        linewidths=0.9, zorder=5,
    )
    ax.set_xlabel("Copy-number share of top-200 SHAP")
    ax.set_ylabel("ΔPearson r  vs RF baseline")
    ax.set_title(label)
    ax.grid(color="#e5e5e5", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.text(
        0.02, 0.04,
        f"Outlined / labelled: top {label_top} ΔPearson r",
        transform=ax.transAxes,
        ha="left", va="bottom",
        color=PALETTE["highlight"],
        fontsize=plt.rcParams["legend.fontsize"] - 0.5,
    )
    _place_callout_labels(
        ax, winners,
        x_col="share_copynumber", y_col="delta_pearsonr_vs_rf",
        column_axfrac=0.72,
        y_top_axfrac=0.97,
        y_bottom_axfrac=0.40,
        fontsize=plt.rcParams["xtick.labelsize"] - 0.5,
    )


def _single_volcano(analysis: pd.DataFrame, family: str, label: str, stem: str,
                     label_top: int = 10) -> Path:
    configure_nature_style("column")
    sub = analysis[analysis["target_family"] == family].dropna(
        subset=["share_copynumber", "delta_pearsonr_vs_rf", "test_pearsonr_tabpfn"]
    )
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    fig.subplots_adjust(left=0.14, right=0.98, top=0.92, bottom=0.16)
    _draw_volcano_panel(ax, sub, label, label_top)
    return _save_single(fig, stem)


def single_fig7a_volcano_crispr(analysis: pd.DataFrame) -> Path:
    return _single_volcano(analysis, "crisprcas9", "CRISPR-Cas9",
                            "single_fig7a_volcano_crispr")


def single_fig7b_volcano_drug(analysis: pd.DataFrame) -> Path:
    return _single_volcano(analysis, "drugresponse", "Drug response",
                            "single_fig7b_volcano_drug")


def _draw_concentration_panel(ax, df_family: pd.DataFrame, label: str) -> None:
    color = TARGET_COLORS[label]
    df = df_family.copy()
    df = df.sort_values(["target_name", "mean_abs_shap"], ascending=[True, False])
    per_target_total = df.groupby("target_name")["mean_abs_shap"].transform("sum")
    df["frac_of_total"] = df["mean_abs_shap"] / per_target_total
    df["rank"] = df.groupby("target_name").cumcount() + 1
    df["cum"] = df.groupby("target_name")["frac_of_total"].cumsum()
    max_rank = int(df["rank"].max())
    ranks = np.arange(1, max_rank + 1)
    wide = df.pivot_table(index="target_name", columns="rank", values="cum")
    wide = wide.reindex(columns=ranks).ffill(axis=1)
    median = wide.median(axis=0).values
    q10 = wide.quantile(0.10, axis=0).values
    q90 = wide.quantile(0.90, axis=0).values
    ax.fill_between(ranks, q10, q90, color=color, alpha=0.18, linewidth=0)
    ax.plot(ranks, median, color=color, linewidth=1.6,
            label=f"{label} median (n={wide.shape[0]})")
    ax.plot(ranks, q10, color=color, linewidth=0.6, linestyle="--", alpha=0.7)
    ax.plot(ranks, q90, color=color, linewidth=0.6, linestyle="--", alpha=0.7)
    top10_med = float(np.interp(10, ranks, median))
    ax.plot([10, 10], [0, top10_med], color=PALETTE["highlight"],
            linewidth=0.8, linestyle=":")
    ax.plot([0, 10], [top10_med, top10_med], color=PALETTE["highlight"],
            linewidth=0.8, linestyle=":")
    ax.scatter([10], [top10_med], s=18, color=PALETTE["highlight"],
               zorder=5, linewidths=0)
    ax.text(
        0.98, 0.05,
        f"Top-10 captures {top10_med * 100:.0f}% of top-200 SHAP\n"
        f"(median across {wide.shape[0]} targets)",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=plt.rcParams["legend.fontsize"] - 0.5,
        color="#222222",
    )
    ax.set_xlim(1, max_rank)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Feature rank within top-200")
    ax.set_ylabel("Cumulative share of top-200 SHAP")
    ax.set_title(label)
    ax.grid(color="#e5e5e5", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


def _single_concentration(tables: dict, family: str, label: str, stem: str) -> Path:
    configure_nature_style("column")
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.92, bottom=0.18)
    _draw_concentration_panel(ax, tables["top_features"][family], label)
    return _save_single(fig, stem)


def single_fig8a_concentration_crispr(tables: dict) -> Path:
    return _single_concentration(tables, "crisprcas9", "CRISPR-Cas9",
                                  "single_fig8a_concentration_crispr")


def single_fig8b_concentration_drug(tables: dict) -> Path:
    return _single_concentration(tables, "drugresponse", "Drug response",
                                  "single_fig8b_concentration_drug")


def render_all_singles(tables: dict, analysis: pd.DataFrame,
                        plotted_targets: list[str]) -> list[Path]:
    """Produce every single-panel figure and return the list of PDF paths."""
    paths: list[Path] = [
        single_fig1a_omic_abs(tables),
        single_fig1b_omic_share(tables),
        single_fig1c_top_features_crispr(tables),
        single_fig1d_top_features_drug(tables),
        single_fig2_global_feature_heatmap(tables),
        single_fig3a_composition_crispr(tables),
        single_fig3b_composition_drug(tables),
        single_fig4a_performance_distribution(analysis),
        single_fig4b_cnv_share_scatter(analysis),
        single_fig4c_entropy_gain_scatter(analysis),
        single_fig4d_dominant_omic_box(analysis),
    ]
    paths.extend(render_singles_fig5_per_target(tables, analysis, plotted_targets))
    paths.extend([
        single_fig6_selected_crispr_omic_heatmap(analysis, plotted_targets),
        single_fig7a_volcano_crispr(analysis),
        single_fig7b_volcano_drug(analysis),
        single_fig8a_concentration_crispr(tables),
        single_fig8b_concentration_drug(tables),
    ])
    return paths


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_all() -> None:
    tables = load_shap_tables()
    performance = load_performance()
    analysis = assemble_analysis_table(tables, performance)

    fig1 = fig1_global_landscape(tables)
    fig2 = fig2_global_feature_heatmap(tables)
    fig3 = fig3_target_omic_composition(tables)
    fig4 = fig4_performance_vs_profile(analysis)
    fig5, plotted, missing = fig5_selected_crispr_top_features(tables, analysis)
    fig6 = fig6_selected_crispr_omic_heatmap(analysis, targets=plotted)
    fig7 = fig7_cnv_gain_volcano(analysis)
    fig8 = fig8_shap_concentration(tables)

    write_tables(tables, analysis)

    for label, p in [
        ("fig1", fig1), ("fig2", fig2), ("fig3", fig3), ("fig4", fig4),
        ("fig5", fig5), ("fig6", fig6), ("fig7", fig7), ("fig8", fig8),
    ]:
        print(f"{label}: {p.relative_to(ROOT)}")
    if missing:
        print(f"Note: requested CRISPR targets unavailable in SHAP export: {missing}")
    print(f"Figure 5/6 plotted targets: {plotted}")

    print()
    print("Single-panel figures:")
    for p in render_all_singles(tables, analysis, plotted):
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    run_all()
