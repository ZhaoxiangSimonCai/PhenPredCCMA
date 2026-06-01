"""CBA 2026 prediction analysis — publication-quality figure generators.

Mirrors the structure of ``_shap_analysis.py`` but for the prediction story
(TabPFN vs Random Forest, with vs without MOSA augmentation) on the selected
+CNV MOSA *union variant* run ``20260511_174623`` (``min_views_per_sample=1``).

Outputs land in ``reports/cba2026_claude_union/prediction_analysis/<TIMESTAMP>/``
with single-panel exports under ``…/singles/``. Every figure is built with
the ``_plot_style`` helpers (Okabe-Ito palette, three type scales, 300 dpi
PNG plus Type-42 editable PDF).

Public entry points:
- ``run_all()`` — regenerate every composite, every single, and every CSV.
- ``load_per_target_long()`` / ``load_summary_long()`` — return the long-form
  master frames consumed by the notebooks.
- ``load_decomposed()`` — returns a per-target wide frame with the four
  TabPFN/RF × original/mosa_all levers needed for Figs P2, P3, P7, P8.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _plot_style import (
    PALETTE,
    SCATTER_ALPHA,
    configure_nature_style,
    panel_label,
    save_figure,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIMESTAMP = "20260511_174623"
OLDER_TIMESTAMP = "20260313_162348"   # no-CNV reference run for Fig P11

FAMILY_ORDER = ["crisprcas9", "drugresponse"]
FAMILY_DISPLAY = {"crisprcas9": "CRISPR-Cas12", "drugresponse": "Drug response"}

VARIANT_ORDER = ["original", "mosa_nan_only", "mosa_all"]
VARIANT_DISPLAY = {
    "original": "Original",
    "mosa_nan_only": "MOSA (gaps only)",
    "mosa_all": "MOSA (all)",
}

FRAME_ORDER = ["overlap", "expanded"]
FRAME_DISPLAY = {"overlap": "Overlap", "expanded": "Expanded"}

MODEL_ORDER = ["tabpfn", "random_forest"]
MODEL_DISPLAY = {"tabpfn": "TabPFN", "random_forest": "Random Forest"}

# Two-way contrast — TabPFN is the focus, RF is the reference baseline.
MODEL_COLORS = {
    "tabpfn": PALETTE["new"],
    "random_forest": PALETTE["common"],
}
FAMILY_COLORS = {
    "crisprcas9": PALETTE["new"],
    "drugresponse": PALETTE["lost"],
}
VARIANT_COLORS = {
    "original": PALETTE["common"],
    "mosa_nan_only": PALETTE["accent"],
    "mosa_all": PALETTE["new"],
}

# The abstract's headline conditions.
HEADLINE_FRAME = "expanded"
HEADLINE_VARIANT = "mosa_all"
BASELINE_FRAME = "overlap"
BASELINE_VARIANT = "original"

# Two-condition headline view used by Fig P1 and the single-panel exports —
# the abstract compares only these two columns (everything else lives in the
# decomposition Fig P3 and the strategy heatmap Fig P4a/b).
HEADLINE_CONDITIONS = [
    (BASELINE_FRAME, BASELINE_VARIANT),
    (HEADLINE_FRAME, HEADLINE_VARIANT),
]
HEADLINE_CONDITION_LABEL = {
    (BASELINE_FRAME, BASELINE_VARIANT): "Overlap\n× Original",
    (HEADLINE_FRAME, HEADLINE_VARIANT): "Expanded\n× MOSA",
}

OMIC_DISPLAY = {
    "transcriptomics": "Transcriptomics",
    "methylation": "Methylation",
    "drugresponse": "Drug response",
    "crisprcas9": "CRISPR-Cas12",
    "copynumber": "Copy number",
    "conditionals": "Conditionals",
}
OMIC_COLORS = {
    "transcriptomics": "#0072B2",
    "methylation": "#009E73",
    "drugresponse": "#D55E00",
    "crisprcas9": "#CC79A7",
    "conditionals": "#666666",
    "copynumber": "#E69F00",
}
# Imputation accuracy is only meaningful where we have measured ground truth
# in the CCMA splits. Conditionals = mutations, no continuous imputation.
IMPUTATION_OMICS = ["crisprcas9", "drugresponse", "copynumber",
                    "transcriptomics", "methylation"]


# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------


def find_repo_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "reports" / "vae" / "files").exists() and (candidate / "docs").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root containing reports/vae/files and docs.")


ROOT = find_repo_root(Path(__file__).resolve().parent)

COMPARE_DIR = ROOT / "reports" / "model_comparison_cnv_mosa_only_union" / "feature_augmentation" / TIMESTAMP
TABPFN_DIR = ROOT / "reports" / "tabpfn_cnv_mosa_only_union" / "feature_augmentation" / TIMESTAMP
RF_DIR = ROOT / "reports" / "random_forest_cnv_mosa_only_union" / "feature_augmentation" / TIMESTAMP

OLDER_COMPARE_DIR = ROOT / "reports" / "model_comparison" / "feature_augmentation" / OLDER_TIMESTAMP

VAE_FILES_DIR = ROOT / "reports" / "vae" / "files"
DATA_DIR = ROOT / "data" / "clines" / "ccma_processed"

FIG_DIR = ROOT / "reports" / "cba2026_claude_union" / "prediction_analysis" / TIMESTAMP
SINGLE_FIG_DIR = FIG_DIR / "singles"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SINGLE_FIG_DIR.mkdir(parents=True, exist_ok=True)


def require_files(paths: list[Path]) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        rel = "\n".join(str(p.relative_to(ROOT)) for p in missing)
        raise FileNotFoundError(f"Missing required input files:\n{rel}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_summary_long() -> pd.DataFrame:
    """Long-form summary: 24 rows = 6 variant×frame × 2 family × 2 model."""
    require_files([COMPARE_DIR / "combined_summary.csv"])
    df = pd.read_csv(COMPARE_DIR / "combined_summary.csv")
    df["variant"] = pd.Categorical(df["variant"], categories=VARIANT_ORDER, ordered=True)
    df["sample_frame"] = pd.Categorical(df["sample_frame"], categories=FRAME_ORDER, ordered=True)
    df["target_family"] = pd.Categorical(df["target_family"], categories=FAMILY_ORDER, ordered=True)
    df["model_name"] = pd.Categorical(df["model_name"], categories=MODEL_ORDER, ordered=True)
    return df.sort_values(["target_family", "model_name", "sample_frame", "variant"]).reset_index(drop=True)


def load_per_target_long() -> pd.DataFrame:
    """Long-form per-target: ~12000 rows, one per family × frame × variant × model × target."""
    require_files([COMPARE_DIR / "combined_per_target.csv"])
    df = pd.read_csv(COMPARE_DIR / "combined_per_target.csv")
    df["variant"] = pd.Categorical(df["variant"], categories=VARIANT_ORDER, ordered=True)
    df["sample_frame"] = pd.Categorical(df["sample_frame"], categories=FRAME_ORDER, ordered=True)
    df["target_family"] = pd.Categorical(df["target_family"], categories=FAMILY_ORDER, ordered=True)
    df["model_name"] = pd.Categorical(df["model_name"], categories=MODEL_ORDER, ordered=True)
    return df


def load_summary_wide() -> pd.DataFrame:
    require_files([COMPARE_DIR / "summary_model_comparison.csv"])
    return pd.read_csv(COMPARE_DIR / "summary_model_comparison.csv")


def load_per_target_wide() -> pd.DataFrame:
    require_files([COMPARE_DIR / "per_target_model_comparison.csv"])
    return pd.read_csv(COMPARE_DIR / "per_target_model_comparison.csv")


def load_decomposed() -> pd.DataFrame:
    """Per-target wide frame with the four headline levers.

    Returns one row per (target_family, target_name) and columns:
    - r_rf_orig:        RF, overlap × original   (abstract baseline)
    - r_rf_mosa_all:    RF, expanded × mosa_all  (RF with MOSA augmentation)
    - r_tabpfn_orig:    TabPFN, overlap × original
    - r_tabpfn_mosa_all:TabPFN, expanded × mosa_all (abstract headline)
    - delta_headline:   r_tabpfn_mosa_all − r_rf_orig (paired Δ in the abstract)
    - delta_mosa_at_rf, delta_mosa_at_tabpfn (within-model effects, both at expanded)
    - delta_model_at_orig, delta_model_at_mosa_all (within-data effects)
    - valid_test_n_*    (per-condition test sizes)
    """
    long = load_per_target_long()
    levers = {
        ("random_forest", "overlap", "original"):     "r_rf_orig",
        ("random_forest", "expanded", "original"):    "r_rf_orig_exp",
        ("random_forest", "expanded", "mosa_all"):    "r_rf_mosa_all",
        ("tabpfn", "overlap", "original"):            "r_tabpfn_orig",
        ("tabpfn", "expanded", "original"):           "r_tabpfn_orig_exp",
        ("tabpfn", "expanded", "mosa_all"):           "r_tabpfn_mosa_all",
        ("random_forest", "overlap", "mosa_nan_only"):"r_rf_overlap_mnan",
        ("random_forest", "expanded", "mosa_nan_only"):"r_rf_expanded_mnan",
        ("tabpfn", "overlap", "mosa_nan_only"):       "r_tabpfn_overlap_mnan",
        ("tabpfn", "expanded", "mosa_nan_only"):      "r_tabpfn_expanded_mnan",
    }
    pieces = []
    for (model, frame, variant), col in levers.items():
        sub = long[
            (long["model_name"] == model)
            & (long["sample_frame"] == frame)
            & (long["variant"] == variant)
        ][["target_family", "target", "test_pearsonr", "test_r2", "valid_test_n"]].copy()
        sub = sub.rename(columns={
            "target": "target_name",
            "test_pearsonr": col,
            "test_r2": col.replace("r_", "r2_"),
            "valid_test_n": f"n_{col[2:]}",
        })
        pieces.append(sub.set_index(["target_family", "target_name"]))
    out = pd.concat(pieces, axis=1).reset_index()

    out["delta_headline"] = out["r_tabpfn_mosa_all"] - out["r_rf_orig"]

    # Within-model MOSA effect — hold sample_frame=expanded so the contrast
    # is "augmentation strategy", not "cohort growth".
    out["delta_mosa_at_rf"] = out["r_rf_mosa_all"] - out["r_rf_orig_exp"]
    out["delta_mosa_at_tabpfn"] = out["r_tabpfn_mosa_all"] - out["r_tabpfn_orig_exp"]

    # Within-data model effect — same data state, switch model.
    out["delta_model_at_orig"] = out["r_tabpfn_orig"] - out["r_rf_orig"]
    out["delta_model_at_mosa_all"] = out["r_tabpfn_mosa_all"] - out["r_rf_mosa_all"]

    return out


def load_test_matrices(family: str, frame: str, variant: str, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (truth, prediction) wide frames for one run."""
    base = (TABPFN_DIR if model == "tabpfn" else RF_DIR) / family / frame / variant
    truth = pd.read_csv(base / "test_truth_wide.csv.gz", index_col=0)
    pred = pd.read_csv(base / "test_predictions_wide.csv.gz", index_col=0)
    return truth, pred


def _align_to_cells_x_features(measured: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Normalise *measured* to (cells × features), matching *reference*'s axes.

    The CCMA processed splits are stored cells×features for some omics
    (CRISPR, copy number) and features×cells for others (drug, RNA-seq,
    methylation); the MOSA imputed matrices are always cells×features. We
    detect orientation by intersecting indices.
    """
    overlap_idx = len(measured.index.intersection(reference.index))
    overlap_col = len(measured.columns.intersection(reference.index))
    return measured if overlap_idx >= overlap_col else measured.T


def load_imputation_pair(omic: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Return (measured_truth, mosa_imputed) for held-out test cells.

    Returns None if either file is missing (e.g. for omic layers that don't
    have a CCMA mosa_test split on disk). Orientation of the measured matrix
    is auto-detected so the result is always cells × features.
    """
    truth_path = DATA_DIR / f"{omic}_ccma_mosa_test.csv"
    pred_path = VAE_FILES_DIR / f"{TIMESTAMP}_imputed_{omic}_test_all.csv.gz"
    if not truth_path.exists() or not pred_path.exists():
        return None
    truth = pd.read_csv(truth_path, index_col=0)
    pred = pd.read_csv(pred_path, index_col=0)
    truth = _align_to_cells_x_features(truth, pred)
    common_cells = truth.index.intersection(pred.index)
    common_feats = truth.columns.intersection(pred.columns)
    if not len(common_cells) or not len(common_feats):
        return None
    return truth.loc[common_cells, common_feats], pred.loc[common_cells, common_feats]


def load_imputation_scope() -> pd.DataFrame:
    """Per-omic counts of measured cells/features and the count of cells
    filled by MOSA on top of that.

    MOSA outputs are always cells × features; the CCMA splits flip
    orientation for some omics, so we use the MOSA columns as the canonical
    feature set and align measured to that.
    """
    rows = []
    for omic in IMPUTATION_OMICS + ["conditionals"]:
        measured_path = DATA_DIR / f"{omic}_ccma.csv"
        mosa_path = VAE_FILES_DIR / f"{TIMESTAMP}_imputed_{omic}.csv.gz"
        try:
            measured = pd.read_csv(measured_path, index_col=0) if measured_path.exists() else None
        except Exception:
            measured = None
        try:
            mosa = pd.read_csv(mosa_path, index_col=0) if mosa_path.exists() else None
        except Exception:
            mosa = None
        if measured is None and mosa is None:
            continue
        if measured is not None and mosa is not None:
            measured = _align_to_cells_x_features(measured, mosa)
        n_cells_measured = int(measured.shape[0]) if measured is not None else 0
        n_features = int(measured.shape[1]) if measured is not None else (
            int(mosa.shape[1]) if mosa is not None else 0
        )
        n_cells_mosa = int(mosa.shape[0]) if mosa is not None else 0
        cells_added = max(n_cells_mosa - n_cells_measured, 0) if measured is not None and mosa is not None else 0
        rows.append({
            "omic_layer": omic,
            "n_cells_measured": n_cells_measured,
            "n_features": n_features,
            "n_cells_after_mosa": n_cells_mosa,
            "n_cells_added_by_mosa": cells_added,
        })
    return pd.DataFrame(rows)


def load_older_summary() -> pd.DataFrame | None:
    """Older non-CNV summary frame for the CNV-vs-no-CNV ablation (Fig P11)."""
    p = OLDER_COMPARE_DIR / "combined_summary.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df


def load_older_per_target() -> pd.DataFrame | None:
    p = OLDER_COMPARE_DIR / "combined_per_target.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def bootstrap_mean_ci(values, n_boot: int = 2000, alpha: float = 0.05,
                      seed: int = 13) -> tuple[float, float, float]:
    """Return (mean, lower, upper) of a percentile bootstrap CI."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    if arr.size < 5:
        return (float(arr.mean()), float(arr.mean()), float(arr.mean()))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (float(arr.mean()), lo, hi)


def wilcoxon_paired(a, b) -> tuple[float, int]:
    """Two-sided Wilcoxon signed-rank test on aligned vectors. Returns (p, n)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if a.size < 5:
        return (float("nan"), int(a.size))
    try:
        res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return (float(res.pvalue), int(a.size))
    except ValueError:
        return (float("nan"), int(a.size))


def fmt_pvalue(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-3:
        return f"p = {p:.1e}"
    return f"p = {p:.3f}"


# ---------------------------------------------------------------------------
# Figure P1 — headline performance grid
# ---------------------------------------------------------------------------


def _agg_with_ci(per_target_long: pd.DataFrame) -> pd.DataFrame:
    """Mean per-target Pearson r with 95 % bootstrap CI per condition."""
    keys = ["target_family", "model_name", "sample_frame", "variant"]
    rows = []
    for key, grp in per_target_long.groupby(keys, observed=True):
        m, lo, hi = bootstrap_mean_ci(grp["test_pearsonr"].values)
        m_r2, *_ = bootstrap_mean_ci(grp["test_r2"].values)
        rows.append({
            "target_family": key[0],
            "model_name": key[1],
            "sample_frame": key[2],
            "variant": key[3],
            "mean_r": m, "ci_lo": lo, "ci_hi": hi,
            "mean_r2": m_r2,
            "n_targets": int((~grp["test_pearsonr"].isna()).sum()),
        })
    return pd.DataFrame(rows)


def figP1_headline_grid(summary_long: pd.DataFrame, per_target_long: pd.DataFrame) -> Path:
    """Composite of four panels:
    (a) mean per-target Pearson r with bootstrap CIs,
    (b) mean per-target R² (negative R² means worse than mean predictor),
    (c) pooled (sample-level) Pearson r from the run summaries,
    (d) train_n cohort sizes used by each variant × frame.
    """
    configure_nature_style("composite")
    agg = _agg_with_ci(per_target_long)

    cond_keys = list(HEADLINE_CONDITIONS)
    n_cond = len(cond_keys)
    bar_w = 0.42

    fig = plt.figure(figsize=(7.4, 7.6))
    gs = GridSpec(
        nrows=4, ncols=2, figure=fig,
        height_ratios=[1.0, 1.0, 1.0, 1.0],
        hspace=0.55, wspace=0.30,
        left=0.10, right=0.985, top=0.96, bottom=0.10,
    )
    ax_r_crispr = fig.add_subplot(gs[0, 0])
    ax_r_drug = fig.add_subplot(gs[0, 1])
    ax_r2_crispr = fig.add_subplot(gs[1, 0])
    ax_r2_drug = fig.add_subplot(gs[1, 1])
    ax_pool_crispr = fig.add_subplot(gs[2, 0])
    ax_pool_drug = fig.add_subplot(gs[2, 1])
    ax_n_crispr = fig.add_subplot(gs[3, 0])
    ax_n_drug = fig.add_subplot(gs[3, 1])

    def _bars(ax, value_col, ci_cols, family, *, ylabel: str, show_legend=False):
        for model_idx, model in enumerate(MODEL_ORDER):
            offs = (model_idx - 0.5) * bar_w
            x = np.arange(n_cond) + offs
            ys, err_lo, err_hi = [], [], []
            for frame, variant in cond_keys:
                row = agg[
                    (agg["target_family"] == family)
                    & (agg["model_name"] == model)
                    & (agg["sample_frame"] == frame)
                    & (agg["variant"] == variant)
                ]
                if len(row) and not pd.isna(row[value_col].iloc[0]):
                    v = float(row[value_col].iloc[0])
                    ys.append(v)
                    if ci_cols:
                        lo, hi = ci_cols
                        err_lo.append(max(v - float(row[lo].iloc[0]), 0))
                        err_hi.append(max(float(row[hi].iloc[0]) - v, 0))
                else:
                    ys.append(np.nan)
                    err_lo.append(0); err_hi.append(0)
            yerr = np.array([err_lo, err_hi]) if ci_cols else None
            ax.bar(
                x, ys, width=bar_w, color=MODEL_COLORS[model],
                edgecolor="black", linewidth=0.45,
                yerr=yerr, error_kw=dict(linewidth=0.7, ecolor="#333333", capsize=1.5),
                label=MODEL_DISPLAY[model] if show_legend else None,
            )
        ax.set_xticks(np.arange(n_cond))
        ax.set_xticklabels(
            [HEADLINE_CONDITION_LABEL[(f, v)] for f, v in cond_keys],
            fontsize=plt.rcParams["xtick.labelsize"],
        )
        ax.tick_params(axis="x", pad=2, length=2.5)
        ax.set_ylabel(ylabel)
        ax.set_title(FAMILY_DISPLAY[family])
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)

    _bars(ax_r_crispr, "mean_r", ("ci_lo", "ci_hi"), "crisprcas9",
          ylabel="Mean per-target Pearson r", show_legend=True)
    _bars(ax_r_drug, "mean_r", ("ci_lo", "ci_hi"), "drugresponse",
          ylabel="Mean per-target Pearson r")
    _bars(ax_r2_crispr, "mean_r2", None, "crisprcas9",
          ylabel="Mean per-target R²")
    _bars(ax_r2_drug, "mean_r2", None, "drugresponse",
          ylabel="Mean per-target R²")

    # ---- pooled Pearson r — read directly from the run summaries ----------
    summ = summary_long.copy()
    for ax, family in [(ax_pool_crispr, "crisprcas9"), (ax_pool_drug, "drugresponse")]:
        for model_idx, model in enumerate(MODEL_ORDER):
            offs = (model_idx - 0.5) * bar_w
            x = np.arange(n_cond) + offs
            ys = []
            for frame, variant in cond_keys:
                row = summ[
                    (summ["target_family"] == family)
                    & (summ["model_name"] == model)
                    & (summ["sample_frame"] == frame)
                    & (summ["variant"] == variant)
                ]
                ys.append(float(row["pooled_test_pearsonr"].iloc[0]) if len(row) else np.nan)
            ax.bar(
                x, ys, width=bar_w, color=MODEL_COLORS[model],
                edgecolor="black", linewidth=0.45,
            )
        ax.set_xticks(np.arange(n_cond))
        ax.set_xticklabels(
            [HEADLINE_CONDITION_LABEL[(f, v)] for f, v in cond_keys],
            fontsize=plt.rcParams["xtick.labelsize"],
        )
        ax.set_ylabel("Pooled Pearson r")
        ax.set_title(FAMILY_DISPLAY[family])
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)

    # ---- train_n bars ------------------------------------------------------
    for ax, family in [(ax_n_crispr, "crisprcas9"), (ax_n_drug, "drugresponse")]:
        for model_idx, model in enumerate(MODEL_ORDER):
            offs = (model_idx - 0.5) * bar_w
            x = np.arange(n_cond) + offs
            ys = []
            for frame, variant in cond_keys:
                row = summ[
                    (summ["target_family"] == family)
                    & (summ["model_name"] == model)
                    & (summ["sample_frame"] == frame)
                    & (summ["variant"] == variant)
                ]
                ys.append(int(row["train_n"].iloc[0]) if len(row) else np.nan)
            ax.bar(
                x, ys, width=bar_w, color=MODEL_COLORS[model],
                edgecolor="black", linewidth=0.45,
            )
        ax.set_xticks(np.arange(n_cond))
        ax.set_xticklabels(
            [HEADLINE_CONDITION_LABEL[(f, v)] for f, v in cond_keys],
            fontsize=plt.rcParams["xtick.labelsize"],
        )
        ax.set_ylabel("Training cell lines")
        ax.set_title(FAMILY_DISPLAY[family])
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)

    # Panel labels.
    panel_label(ax_r_crispr, "a", offset=(-0.20, 1.05))
    panel_label(ax_r_drug, "b", offset=(-0.18, 1.05))
    panel_label(ax_r2_crispr, "c", offset=(-0.20, 1.05))
    panel_label(ax_r2_drug, "d", offset=(-0.18, 1.05))
    panel_label(ax_pool_crispr, "e", offset=(-0.20, 1.05))
    panel_label(ax_pool_drug, "f", offset=(-0.18, 1.05))
    panel_label(ax_n_crispr, "g", offset=(-0.20, 1.05))
    panel_label(ax_n_drug, "h", offset=(-0.18, 1.05))

    fig.legend(
        handles=[
            Patch(facecolor=MODEL_COLORS["tabpfn"], edgecolor="black",
                  linewidth=0.4, label="TabPFN"),
            Patch(facecolor=MODEL_COLORS["random_forest"], edgecolor="black",
                  linewidth=0.4, label="Random Forest"),
        ],
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.005),
        frameon=False, handlelength=1.2, columnspacing=2.0,
    )

    out = FIG_DIR / "figP1_headline_performance"
    save_figure(fig, out)
    return Path(str(out) + ".pdf")


# ---------------------------------------------------------------------------
# Figure P2 — paired per-target RF↔TabPFN scatter
# ---------------------------------------------------------------------------


def _draw_paired_scatter(ax, df: pd.DataFrame, color: str, *, x_col: str, y_col: str,
                         x_label: str, y_label: str, title: str) -> None:
    valid = df.dropna(subset=[x_col, y_col])
    x = valid[x_col].values
    y = valid[y_col].values
    if x.size == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        return
    n = x.size
    n_above = int((y > x).sum())
    p_wilcoxon, _ = wilcoxon_paired(y, x)
    lim_lo = min(x.min(), y.min(), -0.1) - 0.05
    lim_hi = max(x.max(), y.max(), 0.6) + 0.05
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], linestyle="--",
            color="#888888", linewidth=0.7, zorder=1)
    ax.axhline(0, color="#cccccc", linewidth=0.5, zorder=0)
    ax.axvline(0, color="#cccccc", linewidth=0.5, zorder=0)
    ax.scatter(
        x, y, s=8.0, color=color, alpha=SCATTER_ALPHA["new"],
        linewidths=0.0, rasterized=True, zorder=2,
    )
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(color="#e5e5e5", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.text(
        0.04, 0.96,
        f"n = {n}\nabove diag = {n_above} ({n_above / max(n, 1):.0%})\n{fmt_pvalue(p_wilcoxon)} (Wilcoxon)",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=plt.rcParams["legend.fontsize"] - 0.5,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85),
    )


def figP2_paired_scatter(decomposed: pd.DataFrame) -> Path:
    """Two-panel paired scatter: RF baseline vs TabPFN+MOSA per target."""
    configure_nature_style("composite")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.8))
    fig.subplots_adjust(left=0.10, right=0.985, top=0.88, bottom=0.18, wspace=0.32)
    for ax, family, letter in zip(axes, FAMILY_ORDER, ["a", "b"]):
        sub = decomposed[decomposed["target_family"] == family]
        _draw_paired_scatter(
            ax, sub, FAMILY_COLORS[family],
            x_col="r_rf_orig", y_col="r_tabpfn_mosa_all",
            x_label="RF (overlap × original)  Pearson r",
            y_label="TabPFN (expanded × MOSA)  Pearson r",
            title=FAMILY_DISPLAY[family],
        )
        panel_label(ax, letter, offset=(-0.20, 1.05))
    out = FIG_DIR / "figP2_paired_scatter"
    save_figure(fig, out)
    return Path(str(out) + ".pdf")


# ---------------------------------------------------------------------------
# Figure P3 — Δr decomposition
# ---------------------------------------------------------------------------


def _draw_violin(ax, data_by_family: dict[str, np.ndarray]) -> None:
    positions = np.arange(len(data_by_family))
    rng = np.random.default_rng(7)
    for idx, (family, values) in enumerate(data_by_family.items()):
        values = values[~np.isnan(values)]
        if values.size >= 5:
            parts = ax.violinplot(
                values, positions=[idx], vert=True,
                widths=0.7, showmeans=False, showmedians=False, showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(FAMILY_COLORS[family])
                body.set_edgecolor(FAMILY_COLORS[family])
                body.set_alpha(0.30)
        if values.size:
            jitter = rng.uniform(-0.18, 0.18, size=values.size)
            ax.scatter(
                np.full_like(values, idx) + jitter, values,
                s=4.5, color=FAMILY_COLORS[family],
                alpha=0.55, linewidths=0, rasterized=True,
            )
            med = float(np.median(values))
            ax.plot([idx - 0.30, idx + 0.30], [med, med],
                    color="black", linewidth=1.0, solid_capstyle="butt")
    ax.set_xticks(positions)
    ax.set_xticklabels([FAMILY_DISPLAY[f] for f in data_by_family])
    ax.axhline(0, color="#888888", linewidth=0.6, linestyle="--")


def _draw_winrate(ax, data_by_family: dict[str, np.ndarray]) -> None:
    for family, values in data_by_family.items():
        v = values[~np.isnan(values)]
        if v.size == 0:
            continue
        s = np.sort(v)[::-1]
        # Cumulative fraction of targets with delta ≥ x.
        ax.plot(
            s, np.arange(1, s.size + 1) / s.size,
            color=FAMILY_COLORS[family], linewidth=1.4,
            label=FAMILY_DISPLAY[family],
        )
    ax.axvline(0, color="#888888", linewidth=0.6, linestyle="--")
    ax.set_xlabel("ΔPearson r threshold (TabPFN+MOSA − RF baseline)")
    ax.set_ylabel("Cumulative fraction of targets ≥ Δ")
    ax.legend(frameon=False, handlelength=1.2)
    ax.grid(color="#e5e5e5", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


def _mean_with_label(ax, x, vals, color, label_y_offset=0.005, fmt="{:.3f}"):
    m, lo, hi = bootstrap_mean_ci(vals)
    ax.errorbar(
        [x], [m], yerr=[[max(m - lo, 0)], [max(hi - m, 0)]],
        fmt="o", color=color, ecolor="#333333", capsize=2.0, linewidth=0.7,
        markersize=5,
    )
    ax.text(x + 0.07, m, fmt.format(m),
            va="center", ha="left", fontsize=plt.rcParams["legend.fontsize"] - 0.5)
    return m


def _draw_lever_panel(ax, decomposed: pd.DataFrame, *, lever_cols: dict[str, tuple[str, str]],
                      title: str, ylabel: str) -> None:
    """Each tick on x is a (family, lever_label) pair. Family is encoded by
    point colour so the tick label only carries the lever name; this avoids
    text-overlap when the panel is narrow."""
    pos = 0
    ticks = []
    labels = []
    for family in FAMILY_ORDER:
        for label, (col, color_key) in lever_cols.items():
            sub = decomposed[decomposed["target_family"] == family][col].dropna().values
            color = FAMILY_COLORS[family] if color_key == "family" else PALETTE[color_key]
            _mean_with_label(ax, pos, sub, color)
            ticks.append(pos)
            labels.append(label)
            pos += 1
        pos += 0.7  # gap between families
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=22, ha="right",
                       fontsize=plt.rcParams["xtick.labelsize"] - 1.0)
    # Add a faint family separator + family annotation below the ticks.
    family_centres = []
    n_levers = len(lever_cols)
    for fi, family in enumerate(FAMILY_ORDER):
        start = fi * (n_levers + 0.7)
        end = start + n_levers - 1
        family_centres.append((family, (start + end) / 2))
        if fi > 0:
            ax.axvline(start - 0.85, color="#cccccc", linewidth=0.6, linestyle=":")
    ax.axhline(0, color="#888888", linewidth=0.6, linestyle="--")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    # Family band along the bottom — small text below x ticks.
    for family, centre in family_centres:
        ax.annotate(
            FAMILY_DISPLAY[family],
            xy=(centre, 0), xytext=(0, -38),
            xycoords=("data", "axes fraction"),
            textcoords="offset points",
            ha="center", va="top",
            color=FAMILY_COLORS[family],
            fontsize=plt.rcParams["legend.fontsize"] - 0.5,
            fontweight="bold",
        )


def figP3_decomposition(decomposed: pd.DataFrame) -> Path:
    """Δr decomposition: distribution, win-rate, MOSA-only, model-only."""
    configure_nature_style("composite")
    fig = plt.figure(figsize=(7.4, 7.4))
    gs = GridSpec(
        nrows=2, ncols=2, figure=fig,
        hspace=0.85, wspace=0.40,
        left=0.10, right=0.985, top=0.95, bottom=0.13,
    )
    ax_dist = fig.add_subplot(gs[0, 0])
    ax_win = fig.add_subplot(gs[0, 1])
    ax_mosa = fig.add_subplot(gs[1, 0])
    ax_model = fig.add_subplot(gs[1, 1])

    by_family = {
        family: decomposed.loc[decomposed["target_family"] == family, "delta_headline"].values
        for family in FAMILY_ORDER
    }
    _draw_violin(ax_dist, by_family)
    ax_dist.set_ylabel("ΔPearson r  (TabPFN+MOSA − RF baseline)")
    ax_dist.set_title("Per-target ΔPearson r distribution")
    ax_dist.grid(axis="y", color="#e5e5e5", linewidth=0.4, zorder=0)
    ax_dist.set_axisbelow(True)
    panel_label(ax_dist, "a", offset=(-0.20, 1.05))

    _draw_winrate(ax_win, by_family)
    ax_win.set_title("Cumulative win-rate")
    panel_label(ax_win, "b", offset=(-0.20, 1.05))

    _draw_lever_panel(
        ax_mosa, decomposed,
        lever_cols={
            "RF, MOSA effect": ("delta_mosa_at_rf", "family"),
            "TabPFN, MOSA effect": ("delta_mosa_at_tabpfn", "family"),
        },
        title="MOSA-augmentation effect (frame fixed: expanded)",
        ylabel="Mean ΔPearson r vs original",
    )
    panel_label(ax_mosa, "c", offset=(-0.20, 1.05))

    _draw_lever_panel(
        ax_model, decomposed,
        lever_cols={
            "Original data": ("delta_model_at_orig", "family"),
            "MOSA data": ("delta_model_at_mosa_all", "family"),
        },
        title="TabPFN − RF (model effect)",
        ylabel="Mean ΔPearson r (TabPFN − RF)",
    )
    panel_label(ax_model, "d", offset=(-0.20, 1.05))

    out = FIG_DIR / "figP3_decomposition"
    save_figure(fig, out)
    return Path(str(out) + ".pdf")


# ---------------------------------------------------------------------------
# Figure P4 — strategy & sample frame
# ---------------------------------------------------------------------------


def _heatmap_panel(ax, summary_long: pd.DataFrame, family: str, value_col: str,
                   *, title: str, fmt: str = "{:.3f}") -> None:
    """Variant × (model × frame) heatmap of a single metric for one family."""
    cols = [(model, frame) for model in MODEL_ORDER for frame in FRAME_ORDER]
    col_labels = [f"{MODEL_DISPLAY[m]}\n{FRAME_DISPLAY[f]}" for m, f in cols]
    rows = VARIANT_ORDER
    mat = np.full((len(rows), len(cols)), np.nan)
    for i, variant in enumerate(rows):
        for j, (model, frame) in enumerate(cols):
            r = summary_long[
                (summary_long["target_family"] == family)
                & (summary_long["model_name"] == model)
                & (summary_long["sample_frame"] == frame)
                & (summary_long["variant"] == variant)
            ]
            if len(r):
                mat[i, j] = float(r[value_col].iloc[0])
    cmap = plt.get_cmap("viridis")
    finite = mat[np.isfinite(mat)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(col_labels, fontsize=plt.rcParams["xtick.labelsize"] - 1.0)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([VARIANT_DISPLAY[v] for v in rows])
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                # Light text on dark cells, dark text on light cells.
                normed = (v - vmin) / max(vmax - vmin, 1e-9)
                color = "white" if normed < 0.55 else "black"
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        color=color, fontsize=plt.rcParams["legend.fontsize"] - 0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", length=0, pad=2)


def figP4_strategy_frame(summary_long: pd.DataFrame) -> Path:
    """Variant × frame × model heatmap of mean Pearson r, plus train_n bars."""
    configure_nature_style("composite")
    fig = plt.figure(figsize=(7.4, 6.6))
    gs = GridSpec(
        nrows=2, ncols=2, figure=fig,
        hspace=0.55, wspace=0.30,
        left=0.13, right=0.985, top=0.92, bottom=0.10,
    )
    ax_hm_crispr = fig.add_subplot(gs[0, 0])
    ax_hm_drug = fig.add_subplot(gs[0, 1])
    ax_n_crispr = fig.add_subplot(gs[1, 0])
    ax_n_drug = fig.add_subplot(gs[1, 1])

    _heatmap_panel(ax_hm_crispr, summary_long, "crisprcas9", "test_pearsonr",
                   title=f"{FAMILY_DISPLAY['crisprcas9']} — mean per-target Pearson r")
    _heatmap_panel(ax_hm_drug, summary_long, "drugresponse", "test_pearsonr",
                   title=f"{FAMILY_DISPLAY['drugresponse']} — mean per-target Pearson r")
    panel_label(ax_hm_crispr, "a", offset=(-0.18, 1.07))
    panel_label(ax_hm_drug, "b", offset=(-0.18, 1.07))

    bar_w = 0.42
    cond_keys = list(HEADLINE_CONDITIONS)
    cond_labels = [HEADLINE_CONDITION_LABEL[k] for k in cond_keys]
    for ax, family in [(ax_n_crispr, "crisprcas9"), (ax_n_drug, "drugresponse")]:
        for model_idx, model in enumerate(MODEL_ORDER):
            offs = (model_idx - 0.5) * bar_w
            x = np.arange(len(cond_keys)) + offs
            ys = []
            for frame, variant in cond_keys:
                r = summary_long[
                    (summary_long["target_family"] == family)
                    & (summary_long["model_name"] == model)
                    & (summary_long["sample_frame"] == frame)
                    & (summary_long["variant"] == variant)
                ]
                ys.append(int(r["train_n"].iloc[0]) if len(r) else np.nan)
            ax.bar(
                x, ys, width=bar_w, color=MODEL_COLORS[model],
                edgecolor="black", linewidth=0.45,
                label=MODEL_DISPLAY[model] if family == "crisprcas9" else None,
            )
        ax.set_xticks(np.arange(len(cond_keys)))
        ax.set_xticklabels(cond_labels, fontsize=plt.rcParams["xtick.labelsize"])
        ax.set_ylabel("Training cell lines")
        ax.set_title(FAMILY_DISPLAY[family])
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
    panel_label(ax_n_crispr, "c", offset=(-0.18, 1.05))
    panel_label(ax_n_drug, "d", offset=(-0.18, 1.05))

    fig.legend(
        handles=[
            Patch(facecolor=MODEL_COLORS["tabpfn"], edgecolor="black",
                  linewidth=0.4, label="TabPFN"),
            Patch(facecolor=MODEL_COLORS["random_forest"], edgecolor="black",
                  linewidth=0.4, label="Random Forest"),
        ],
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.005),
        frameon=False, handlelength=1.2, columnspacing=2.0,
    )

    out = FIG_DIR / "figP4_strategy_frame"
    save_figure(fig, out)
    return Path(str(out) + ".pdf")


# ---------------------------------------------------------------------------
# Figure P5 — top winners per family
# ---------------------------------------------------------------------------


def _draw_top_winners(ax, decomposed: pd.DataFrame, family: str, top_n: int = 25) -> None:
    sub = decomposed[decomposed["target_family"] == family].dropna(
        subset=["delta_headline", "r_rf_orig", "r_tabpfn_mosa_all"])
    sub = sub.nlargest(top_n, "delta_headline").sort_values("delta_headline", ascending=True)
    yy = np.arange(len(sub))
    color = FAMILY_COLORS[family]
    ax.barh(
        yy, sub["delta_headline"], height=0.78,
        color=color, edgecolor="black", linewidth=0.35,
        alpha=0.85, label="ΔPearson r (TabPFN+MOSA − RF baseline)",
    )
    ax.scatter(
        sub["r_rf_orig"], yy, s=10, color="black", marker="|",
        linewidth=1.0, zorder=4, label="RF baseline Pearson r",
    )
    ax.scatter(
        sub["r_tabpfn_mosa_all"], yy, s=10, color=color, marker="|",
        linewidth=1.0, zorder=4, label="TabPFN+MOSA Pearson r",
    )
    ax.axvline(0, color="#888888", linewidth=0.6, linestyle="--", zorder=0)
    ax.set_yticks(yy)
    ax.set_yticklabels(sub["target_name"].values, fontsize=plt.rcParams["ytick.labelsize"] - 0.5)
    ax.set_xlabel("Pearson r  (•) and ΔPearson r (bar)")
    ax.set_title(f"{FAMILY_DISPLAY[family]} — top {top_n} ΔPearson r gainers")
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylim(-0.6, len(sub) - 0.4)
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


def figP5_top_winners(decomposed: pd.DataFrame, top_n: int = 25) -> Path:
    configure_nature_style("composite")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 6.4))
    fig.subplots_adjust(left=0.13, right=0.985, top=0.93, bottom=0.16, wspace=0.55)
    for ax, family, letter in zip(axes, FAMILY_ORDER, ["a", "b"]):
        _draw_top_winners(ax, decomposed, family, top_n=top_n)
        panel_label(ax, letter, offset=(-0.40, 1.04))

    fig.legend(
        handles=[
            Line2D([0], [0], color="black", marker="|", linestyle="none",
                   markersize=8, markeredgewidth=1.4, label="RF baseline r"),
            Line2D([0], [0], color=PALETTE["new"], marker="|", linestyle="none",
                   markersize=8, markeredgewidth=1.4, label="TabPFN+MOSA r (CRISPR)"),
            Line2D([0], [0], color=PALETTE["lost"], marker="|", linestyle="none",
                   markersize=8, markeredgewidth=1.4, label="TabPFN+MOSA r (drug)"),
        ],
        loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01),
        frameon=False, handlelength=1.4, handletextpad=0.6, columnspacing=2.0,
    )

    out = FIG_DIR / "figP5_top_winners"
    save_figure(fig, out)
    return Path(str(out) + ".pdf")


# ---------------------------------------------------------------------------
# Figure P6 — selected target deep-dives (y_true vs y_pred)
# ---------------------------------------------------------------------------


def select_deep_dive_targets(decomposed: pd.DataFrame) -> dict[str, list[str]]:
    """Pick MDM2/MDM4 + top CRISPR gainer + top three drug gainers.

    Falls back to top-3 per family if the requested CRISPR targets are
    missing from the per-target frame.
    """
    chosen: dict[str, list[str]] = {"crisprcas9": [], "drugresponse": []}
    crispr = decomposed[decomposed["target_family"] == "crisprcas9"].dropna(
        subset=["delta_headline", "r_tabpfn_mosa_all"])
    crispr_top = crispr.nlargest(20, "delta_headline")
    for requested in ["MDM2", "MDM4"]:
        if requested in set(crispr["target_name"]):
            chosen["crisprcas9"].append(requested)
    for name in crispr_top["target_name"]:
        if name not in chosen["crisprcas9"]:
            chosen["crisprcas9"].append(name)
        if len(chosen["crisprcas9"]) >= 3:
            break
    chosen["crisprcas9"] = chosen["crisprcas9"][:3]

    drug = decomposed[decomposed["target_family"] == "drugresponse"].dropna(
        subset=["delta_headline", "r_tabpfn_mosa_all"])
    chosen["drugresponse"] = drug.nlargest(3, "delta_headline")["target_name"].tolist()
    return chosen


def _scatter_truth_vs_pred(ax, truth: pd.Series, tabpfn: pd.Series, rf: pd.Series,
                            family: str, target: str) -> None:
    common = truth.index.intersection(tabpfn.index).intersection(rf.index)
    t = truth.loc[common].astype(float)
    yp = tabpfn.loc[common].astype(float)
    yr = rf.loc[common].astype(float)
    mask = ~(t.isna() | yp.isna() | yr.isna())
    t, yp, yr = t[mask], yp[mask], yr[mask]
    if t.empty:
        ax.text(0.5, 0.5, f"{target}: no overlap", transform=ax.transAxes, ha="center")
        return
    lims = [
        float(min(t.min(), yp.min(), yr.min())) - 0.2,
        float(max(t.max(), yp.max(), yr.max())) + 0.2,
    ]
    ax.plot(lims, lims, linestyle="--", color="#888888", linewidth=0.7, zorder=1)
    ax.scatter(t, yr, s=14, color=PALETTE["common"], alpha=0.6, linewidths=0.0,
               label="RF baseline", zorder=2)
    ax.scatter(t, yp, s=14, color=FAMILY_COLORS[family], alpha=0.85, linewidths=0.0,
               label="TabPFN+MOSA", zorder=3)
    r_p = float(np.corrcoef(t, yp)[0, 1]) if t.size > 1 else float("nan")
    r_r = float(np.corrcoef(t, yr)[0, 1]) if t.size > 1 else float("nan")
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    ax.set_title(target)
    ax.text(
        0.04, 0.96,
        f"TabPFN r = {r_p:.2f}\nRF r = {r_r:.2f}\nn = {t.size}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=plt.rcParams["legend.fontsize"] - 0.5,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.85),
    )
    ax.grid(color="#e5e5e5", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


def figP6_target_deepdives(decomposed: pd.DataFrame,
                            selected: dict[str, list[str]] | None = None) -> tuple[Path, dict[str, list[str]]]:
    """6-panel composite: 3 CRISPR + 3 drug y_true vs y_pred deep-dives."""
    if selected is None:
        selected = select_deep_dive_targets(decomposed)
    configure_nature_style("composite")
    fig = plt.figure(figsize=(7.6, 5.0))
    gs = GridSpec(
        nrows=2, ncols=3, figure=fig,
        hspace=0.55, wspace=0.40,
        left=0.085, right=0.985, top=0.93, bottom=0.16,
    )
    truth_pred = {}
    for family in FAMILY_ORDER:
        t_t, t_p = load_test_matrices(family, HEADLINE_FRAME, HEADLINE_VARIANT, "tabpfn")
        r_t, r_p = load_test_matrices(family, BASELINE_FRAME, BASELINE_VARIANT, "random_forest")
        truth_pred[family] = (t_t, t_p, r_t, r_p)

    panel_letters = "abcdef"
    for col, target in enumerate(selected["crisprcas9"]):
        ax = fig.add_subplot(gs[0, col])
        t_t, t_p, r_t, r_p = truth_pred["crisprcas9"]
        if target not in t_t.columns:
            ax.text(0.5, 0.5, f"{target}\nnot in test set",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
        else:
            _scatter_truth_vs_pred(
                ax, t_t[target], t_p[target],
                r_p[target] if target in r_p.columns else pd.Series(np.nan, index=t_t.index),
                "crisprcas9", target,
            )
        panel_label(ax, panel_letters[col], offset=(-0.20, 1.08))
    for col, target in enumerate(selected["drugresponse"]):
        ax = fig.add_subplot(gs[1, col])
        t_t, t_p, r_t, r_p = truth_pred["drugresponse"]
        if target not in t_t.columns:
            ax.text(0.5, 0.5, f"{target}\nnot in test set",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
        else:
            _scatter_truth_vs_pred(
                ax, t_t[target], t_p[target],
                r_p[target] if target in r_p.columns else pd.Series(np.nan, index=t_t.index),
                "drugresponse", target,
            )
        panel_label(ax, panel_letters[col + 3], offset=(-0.20, 1.08))

    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", color=PALETTE["common"],
                   markersize=5, markeredgewidth=0, label="RF baseline (overlap × original)"),
            Line2D([0], [0], marker="o", linestyle="none", color=PALETTE["new"],
                   markersize=5, markeredgewidth=0, label="TabPFN+MOSA (CRISPR-Cas12)"),
            Line2D([0], [0], marker="o", linestyle="none", color=PALETTE["lost"],
                   markersize=5, markeredgewidth=0, label="TabPFN+MOSA (drug response)"),
        ],
        loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01),
        frameon=False, handlelength=1.0, handletextpad=0.5, columnspacing=2.0,
    )

    out = FIG_DIR / "figP6_target_deepdives"
    save_figure(fig, out)
    return Path(str(out) + ".pdf"), selected


# ---------------------------------------------------------------------------
# Figure P7 — where TabPFN+MOSA helps most
# ---------------------------------------------------------------------------


def figP7_helps_most(decomposed: pd.DataFrame) -> Path:
    """4-panel scatter set covering baseline-r, target activity, n_test, model×data quadrant."""
    configure_nature_style("composite")
    fig = plt.figure(figsize=(7.4, 6.4))
    gs = GridSpec(
        nrows=2, ncols=2, figure=fig,
        hspace=0.55, wspace=0.35,
        left=0.10, right=0.985, top=0.95, bottom=0.10,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    def _scatter(ax, x_col: str, y_col: str, x_label: str, y_label: str, title: str,
                 hline_zero: bool = True) -> None:
        for family in FAMILY_ORDER:
            sub = decomposed[decomposed["target_family"] == family].dropna(subset=[x_col, y_col])
            ax.scatter(
                sub[x_col], sub[y_col], s=8,
                color=FAMILY_COLORS[family], alpha=SCATTER_ALPHA["new"],
                linewidths=0, rasterized=True, label=FAMILY_DISPLAY[family],
            )
        if hline_zero:
            ax.axhline(0, color="#888888", linewidth=0.6, linestyle="--")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(color="#e5e5e5", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)

    _scatter(
        ax_a, "r_rf_orig", "delta_headline",
        "RF baseline Pearson r", "ΔPearson r (TabPFN+MOSA − RF)",
        "Lift vs baseline strength",
    )
    panel_label(ax_a, "a", offset=(-0.20, 1.05))

    # Target activity proxy: standard deviation of MOSA-imputed predictions —
    # a quick proxy for "how variable is this target across the cohort". Use
    # valid_test_n cell lines actually predicted.
    if "n_tabpfn_mosa_all" in decomposed.columns:
        _scatter(
            ax_b, "n_tabpfn_mosa_all", "delta_headline",
            "Test cell lines (TabPFN+MOSA)", "ΔPearson r",
            "Lift vs n_test",
        )
    else:
        ax_b.set_visible(False)
    panel_label(ax_b, "b", offset=(-0.20, 1.05))

    _scatter(
        ax_c, "delta_mosa_at_tabpfn", "delta_model_at_mosa_all",
        "MOSA effect within TabPFN  (mosa_all − original)",
        "Model effect at MOSA data  (TabPFN − RF)",
        "Quadrant: which lever helps each target",
        hline_zero=True,
    )
    ax_c.axvline(0, color="#888888", linewidth=0.6, linestyle="--")
    panel_label(ax_c, "c", offset=(-0.20, 1.05))

    # Per-family bootstrap mean Δ — summary view of the levers.
    levers = [
        ("RF: original → MOSA", "delta_mosa_at_rf"),
        ("TabPFN: original → MOSA", "delta_mosa_at_tabpfn"),
        ("RF → TabPFN, original", "delta_model_at_orig"),
        ("RF → TabPFN, MOSA", "delta_model_at_mosa_all"),
    ]
    pos = 0
    ticks, labels = [], []
    bar_w = 0.35
    for label, col in levers:
        for offs_idx, family in enumerate(FAMILY_ORDER):
            sub = decomposed[decomposed["target_family"] == family][col].dropna().values
            m, lo, hi = bootstrap_mean_ci(sub)
            x = pos + (offs_idx - 0.5) * bar_w
            ax_d.bar(
                x, m, width=bar_w, color=FAMILY_COLORS[family],
                edgecolor="black", linewidth=0.45,
                yerr=[[max(m - lo, 0)], [max(hi - m, 0)]],
                error_kw=dict(linewidth=0.7, ecolor="#333333", capsize=1.5),
            )
        ticks.append(pos)
        labels.append(label)
        pos += 1
    ax_d.set_xticks(ticks)
    ax_d.set_xticklabels(labels, rotation=20, ha="right",
                         fontsize=plt.rcParams["xtick.labelsize"] - 1.0)
    ax_d.axhline(0, color="#888888", linewidth=0.6, linestyle="--")
    ax_d.set_ylabel("Mean ΔPearson r")
    ax_d.set_title("Lever summary (95 % bootstrap CI)")
    ax_d.grid(axis="y", color="#e5e5e5", linewidth=0.4, zorder=0)
    ax_d.set_axisbelow(True)
    panel_label(ax_d, "d", offset=(-0.20, 1.05))

    fig.legend(
        handles=[
            Patch(facecolor=FAMILY_COLORS["crisprcas9"], edgecolor="black",
                  linewidth=0.4, label="CRISPR-Cas12"),
            Patch(facecolor=FAMILY_COLORS["drugresponse"], edgecolor="black",
                  linewidth=0.4, label="Drug response"),
        ],
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.005),
        frameon=False, handlelength=1.2, columnspacing=2.0,
    )

    out = FIG_DIR / "figP7_helps_most"
    save_figure(fig, out)
    return Path(str(out) + ".pdf")


# ---------------------------------------------------------------------------
# Figure P8 — augmentation strategy at the target level
# ---------------------------------------------------------------------------


def _draw_strategy_slope(ax, decomposed: pd.DataFrame, family: str) -> None:
    cols_for_model = {
        "tabpfn": ["r_tabpfn_orig_exp", "r_tabpfn_expanded_mnan", "r_tabpfn_mosa_all"],
        "random_forest": ["r_rf_orig_exp", "r_rf_expanded_mnan", "r_rf_mosa_all"],
    }
    sub = decomposed[decomposed["target_family"] == family]
    rng = np.random.default_rng(11)
    for model, cols in cols_for_model.items():
        valid = sub.dropna(subset=cols)
        if valid.empty:
            continue
        x = np.arange(len(cols))
        for _, row in valid.iterrows():
            jit = rng.uniform(-0.04, 0.04)
            ax.plot(
                x + jit, [row[c] for c in cols],
                color=MODEL_COLORS[model], alpha=0.10, linewidth=0.5, zorder=1,
            )
        median = [valid[c].median() for c in cols]
        ax.plot(
            x, median, color=MODEL_COLORS[model], linewidth=2.0,
            marker="o", markersize=4.5, zorder=4,
            label=f"{MODEL_DISPLAY[model]} median",
        )
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([VARIANT_DISPLAY[v] for v in ["original", "mosa_nan_only", "mosa_all"]])
    ax.set_ylabel("Per-target Pearson r  (sample frame: expanded)")
    ax.set_title(FAMILY_DISPLAY[family])
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, handlelength=1.4, loc="best")


def figP8_strategy_per_target(decomposed: pd.DataFrame) -> Path:
    configure_nature_style("composite")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8))
    fig.subplots_adjust(left=0.10, right=0.985, top=0.88, bottom=0.18, wspace=0.30)
    for ax, family, letter in zip(axes, FAMILY_ORDER, ["a", "b"]):
        _draw_strategy_slope(ax, decomposed, family)
        panel_label(ax, letter, offset=(-0.20, 1.05))
    out = FIG_DIR / "figP8_strategy_per_target"
    save_figure(fig, out)
    return Path(str(out) + ".pdf")


# ---------------------------------------------------------------------------
# Figure P9 — MOSA imputation accuracy (scatter per omic)
# ---------------------------------------------------------------------------


def _scatter_imputation(ax, truth: pd.DataFrame, pred: pd.DataFrame, omic: str) -> dict:
    t = truth.values.flatten().astype(float)
    p = pred.values.flatten().astype(float)
    mask = np.isfinite(t) & np.isfinite(p)
    t, p = t[mask], p[mask]
    if t.size == 0:
        ax.text(0.5, 0.5, "no measured values", transform=ax.transAxes, ha="center")
        return {"omic": omic, "pearson_r": np.nan, "n_pairs": 0,
                "n_cells": truth.shape[0], "n_features": truth.shape[1]}
    if t.size > 50_000:
        rng = np.random.default_rng(31)
        idx = rng.choice(t.size, size=50_000, replace=False)
        t_plot = t[idx]; p_plot = p[idx]
    else:
        t_plot, p_plot = t, p
    color = OMIC_COLORS.get(omic, "#888888")
    lims = [float(min(t.min(), p.min())) - 0.2, float(max(t.max(), p.max())) + 0.2]
    ax.plot(lims, lims, linestyle="--", color="#888888", linewidth=0.6, zorder=1)
    ax.scatter(t_plot, p_plot, s=2.0, color=color, alpha=0.20,
               linewidths=0, rasterized=True, zorder=2)
    r = float(np.corrcoef(t, p)[0, 1])
    non_negative_measured = {"drugresponse", "transcriptomics", "methylation"}
    lim_lo = 0.0 if omic in non_negative_measured else lims[0]
    ax.set_xlim(lim_lo, lims[1]); ax.set_ylim(lim_lo, lims[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Measured")
    ax.set_ylabel("MOSA imputed")
    ax.set_title(OMIC_DISPLAY.get(omic, omic))
    ax.text(
        0.04, 0.96,
        f"r = {r:.2f}\nn = {t.size:,}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=plt.rcParams["legend.fontsize"] - 0.5,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.85),
    )
    ax.grid(color="#e5e5e5", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    return {"omic": omic, "pearson_r": r, "n_pairs": int(t.size),
            "n_cells": truth.shape[0], "n_features": truth.shape[1]}


def figP9_imputation_accuracy() -> tuple[Path, pd.DataFrame]:
    pairs = []
    omic_data = []
    for omic in IMPUTATION_OMICS:
        loaded = load_imputation_pair(omic)
        if loaded is None:
            continue
        omic_data.append((omic, loaded))
    if not omic_data:
        raise FileNotFoundError("No imputation diagnostic pairs found on disk.")
    n = len(omic_data)
    cols = 3
    rows = (n + cols - 1) // cols
    configure_nature_style("composite")
    fig = plt.figure(figsize=(7.4, 2.6 * rows + 0.4))
    gs = GridSpec(
        nrows=rows, ncols=cols, figure=fig,
        hspace=0.55, wspace=0.45,
        left=0.085, right=0.985, top=0.93, bottom=0.10,
    )
    summaries = []
    panel_letters = "abcdefgh"
    for idx, (omic, (truth, pred)) in enumerate(omic_data):
        r_idx, c_idx = divmod(idx, cols)
        ax = fig.add_subplot(gs[r_idx, c_idx])
        s = _scatter_imputation(ax, truth, pred, omic)
        summaries.append(s)
        panel_label(ax, panel_letters[idx], offset=(-0.22, 1.06))
    out = FIG_DIR / "figP9_imputation_accuracy"
    save_figure(fig, out)
    return Path(str(out) + ".pdf"), pd.DataFrame(summaries)


# ---------------------------------------------------------------------------
# Figure P10 — imputation scope
# ---------------------------------------------------------------------------


def figP10_imputation_scope(summary_long: pd.DataFrame) -> Path:
    """Two-panel: cohort-fill bar + per-condition train_n step plot."""
    scope = load_imputation_scope()
    configure_nature_style("composite")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.88, bottom=0.26, wspace=0.30)

    # Panel a: per-omic bar of measured cells vs cells after MOSA.
    ax_a = axes[0]
    omics = scope["omic_layer"].tolist()
    ypos = np.arange(len(omics))
    bar_h = 0.36
    ax_a.barh(
        ypos - bar_h / 2, scope["n_cells_measured"], height=bar_h,
        color="#888888", edgecolor="black", linewidth=0.4, label="Measured cells",
    )
    ax_a.barh(
        ypos + bar_h / 2, scope["n_cells_after_mosa"], height=bar_h,
        color=PALETTE["new"], edgecolor="black", linewidth=0.4, label="After MOSA",
    )
    ax_a.set_yticks(ypos)
    ax_a.set_yticklabels([OMIC_DISPLAY.get(o, o) for o in omics])
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Cell lines")
    ax_a.set_title("Cohort coverage by omic layer")
    ax_a.grid(axis="x", color="#e5e5e5", linewidth=0.4, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
                frameon=False, handlelength=1.0)
    panel_label(ax_a, "a", offset=(-0.30, 1.08))
    for i, (m, mosa) in enumerate(zip(scope["n_cells_measured"], scope["n_cells_after_mosa"])):
        added = mosa - m
        if added > 0:
            ax_a.text(
                mosa + 5, i + bar_h / 2, f"+{added}",
                va="center", fontsize=plt.rcParams["legend.fontsize"] - 0.5,
                color=PALETTE["new"],
            )

    # Panel b: train_n vs Pearson r per condition (drug response only as
    # rotating proxy — both families on the same axis).
    ax_b = axes[1]
    plotted_families = set()
    for family in FAMILY_ORDER:
        sub = summary_long[summary_long["target_family"] == family]
        for model in MODEL_ORDER:
            sub_m = sub[sub["model_name"] == model]
            if sub_m.empty:
                continue
            ax_b.scatter(
                sub_m["train_n"], sub_m["test_pearsonr"],
                s=22, marker="o" if model == "tabpfn" else "s",
                facecolor=FAMILY_COLORS[family],
                edgecolor=MODEL_COLORS[model], linewidth=0.8,
                label=f"{FAMILY_DISPLAY[family]} • {MODEL_DISPLAY[model]}",
                zorder=4,
            )
            plotted_families.add(family)
    ax_b.set_xlabel("Training cell lines")
    ax_b.set_ylabel("Mean per-target Pearson r")
    ax_b.set_title("Performance vs cohort size")
    ax_b.grid(color="#e5e5e5", linewidth=0.4, zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.legend(frameon=False, handlelength=1.0,
                fontsize=plt.rcParams["legend.fontsize"] - 1.0,
                loc="best")
    panel_label(ax_b, "b", offset=(-0.20, 1.08))

    out = FIG_DIR / "figP10_imputation_scope"
    save_figure(fig, out)
    return Path(str(out) + ".pdf")


# ---------------------------------------------------------------------------
# Figure P11 — CNV-vs-no-CNV MOSA ablation
# ---------------------------------------------------------------------------


def _per_target_mean_for_run(per_target: pd.DataFrame | None, *, run_label: str) -> pd.DataFrame:
    if per_target is None or per_target.empty:
        return pd.DataFrame(columns=["run", "target_family", "model_name", "sample_frame",
                                     "variant", "test_pearsonr"])
    return (
        per_target
        .groupby(["target_family", "model_name", "sample_frame", "variant"], observed=True)["test_pearsonr"]
        .mean()
        .reset_index()
        .assign(run=run_label)
    )


def figP11_cnv_ablation() -> Path | None:
    """CNV-vs-no-CNV ablation: bars + per-target scatter at expanded × mosa_all."""
    older_per_target = load_older_per_target()
    new_per_target = load_per_target_long().rename(columns={"target": "target_name"})
    if older_per_target is None:
        # Render a placeholder note — keep API stable.
        configure_nature_style("composite")
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        ax.axis("off")
        ax.text(0.5, 0.5,
                f"No older non-CNV run found at {OLDER_COMPARE_DIR.relative_to(ROOT)};\n"
                "skipping CNV ablation.",
                ha="center", va="center")
        out = FIG_DIR / "figP11_cnv_ablation"
        save_figure(fig, out)
        return Path(str(out) + ".pdf")

    older_per_target = older_per_target.rename(columns={"target": "target_name"})

    # Build family-level mean Pearson r at each run × condition.
    means_new = (
        new_per_target
        .groupby(["target_family", "model_name", "sample_frame", "variant"], observed=True)["test_pearsonr"]
        .mean().reset_index().assign(run="with_cnv")
    )
    means_old = (
        older_per_target
        .groupby(["target_family", "model_name", "sample_frame", "variant"], observed=True)["test_pearsonr"]
        .mean().reset_index().assign(run="without_cnv")
    )
    means = pd.concat([means_new, means_old], ignore_index=True)
    headline = means[
        (means["sample_frame"] == HEADLINE_FRAME)
        & (means["variant"] == HEADLINE_VARIANT)
    ]

    configure_nature_style("composite")
    fig = plt.figure(figsize=(7.4, 4.0))
    gs = GridSpec(
        nrows=1, ncols=2, figure=fig,
        wspace=0.40,
        left=0.10, right=0.985, top=0.86, bottom=0.18,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # Panel a — grouped bars, family × model, with/without CNV.
    bar_w = 0.36
    cond_keys = [(family, model) for family in FAMILY_ORDER for model in MODEL_ORDER]
    cond_labels = [f"{FAMILY_DISPLAY[f]}\n{MODEL_DISPLAY[m]}" for f, m in cond_keys]
    for run_idx, run_label in enumerate(["without_cnv", "with_cnv"]):
        offs = (run_idx - 0.5) * bar_w
        x = np.arange(len(cond_keys)) + offs
        ys = []
        for family, model in cond_keys:
            r = headline[
                (headline["run"] == run_label)
                & (headline["target_family"] == family)
                & (headline["model_name"] == model)
            ]
            ys.append(float(r["test_pearsonr"].iloc[0]) if len(r) else np.nan)
        ax_a.bar(
            x, ys, width=bar_w,
            color=PALETTE["common"] if run_label == "without_cnv" else PALETTE["new"],
            edgecolor="black", linewidth=0.45,
            label="Without CNV" if run_label == "without_cnv" else "With CNV",
        )
    ax_a.set_xticks(np.arange(len(cond_keys)))
    ax_a.set_xticklabels(cond_labels, fontsize=plt.rcParams["xtick.labelsize"] - 1.0)
    ax_a.set_ylabel("Mean per-target Pearson r")
    ax_a.set_title(f"Headline condition: {FRAME_DISPLAY[HEADLINE_FRAME]} × "
                   f"{VARIANT_DISPLAY[HEADLINE_VARIANT]}")
    ax_a.legend(frameon=False, handlelength=1.2, loc="upper left")
    ax_a.grid(axis="y", color="#e5e5e5", linewidth=0.4, zorder=0)
    ax_a.set_axisbelow(True)
    panel_label(ax_a, "a", offset=(-0.18, 1.08))

    # Panel b — per-target scatter (no-CNV vs with-CNV) for TabPFN headline.
    new_pivot = (
        new_per_target[
            (new_per_target["sample_frame"] == HEADLINE_FRAME)
            & (new_per_target["variant"] == HEADLINE_VARIANT)
            & (new_per_target["model_name"] == "tabpfn")
        ][["target_family", "target_name", "test_pearsonr"]]
        .rename(columns={"test_pearsonr": "r_with_cnv"})
    )
    old_pivot = (
        older_per_target[
            (older_per_target["sample_frame"] == HEADLINE_FRAME)
            & (older_per_target["variant"] == HEADLINE_VARIANT)
            & (older_per_target["model_name"] == "tabpfn")
        ][["target_family", "target_name", "test_pearsonr"]]
        .rename(columns={"test_pearsonr": "r_without_cnv"})
    )
    paired = new_pivot.merge(old_pivot, on=["target_family", "target_name"], how="inner")
    paired = paired.dropna(subset=["r_with_cnv", "r_without_cnv"])

    lim_lo = float(paired[["r_with_cnv", "r_without_cnv"]].min().min()) - 0.05
    lim_hi = float(paired[["r_with_cnv", "r_without_cnv"]].max().max()) + 0.05
    ax_b.plot([lim_lo, lim_hi], [lim_lo, lim_hi], linestyle="--", color="#888888", linewidth=0.7)
    for family in FAMILY_ORDER:
        sub = paired[paired["target_family"] == family]
        ax_b.scatter(
            sub["r_without_cnv"], sub["r_with_cnv"], s=8,
            color=FAMILY_COLORS[family], alpha=SCATTER_ALPHA["new"],
            linewidths=0, rasterized=True,
            label=FAMILY_DISPLAY[family],
        )
    ax_b.set_xlim(lim_lo, lim_hi)
    ax_b.set_ylim(lim_lo, lim_hi)
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.set_xlabel("Pearson r — MOSA without CNV (TabPFN)")
    ax_b.set_ylabel("Pearson r — MOSA with CNV (TabPFN)")
    ax_b.set_title("Per-target CNV ablation")
    ax_b.legend(frameon=False, handlelength=1.0, loc="upper left",
                fontsize=plt.rcParams["legend.fontsize"] - 0.5)
    ax_b.grid(color="#e5e5e5", linewidth=0.4, zorder=0)
    ax_b.set_axisbelow(True)
    panel_label(ax_b, "b", offset=(-0.20, 1.08))

    n_above = int((paired["r_with_cnv"] > paired["r_without_cnv"]).sum())
    p, _ = wilcoxon_paired(paired["r_with_cnv"], paired["r_without_cnv"])
    ax_b.text(
        0.04, 0.04,
        f"n = {len(paired)}\nabove diag = {n_above} ({n_above / max(len(paired), 1):.0%})\n{fmt_pvalue(p)}",
        transform=ax_b.transAxes, ha="left", va="bottom",
        fontsize=plt.rcParams["legend.fontsize"] - 0.5,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.85),
    )

    out = FIG_DIR / "figP11_cnv_ablation"
    save_figure(fig, out)
    return Path(str(out) + ".pdf")


# ---------------------------------------------------------------------------
# Figure P12 — TabPFN vs RF per-target consistency within each data scope
# ---------------------------------------------------------------------------

# Two same-data, same-frame contrasts. Both models see the same training and
# test split inside each scope, so any per-target divergence is purely the
# learner choice (TabPFN vs RF).
CONSISTENCY_SCOPES = [
    ("orig",     "Original",       "r_rf_orig",     "r_tabpfn_orig"),
    ("mosa_all", "MOSA expanded",  "r_rf_mosa_all", "r_tabpfn_mosa_all"),
]


def _pearson_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if x.size < 3:
        return (float("nan"), float("nan"), int(x.size))
    pr = float(stats.pearsonr(x, y).statistic)
    sr = float(stats.spearmanr(x, y).statistic)
    return (pr, sr, int(x.size))


def _draw_consistency_scatter(ax, df: pd.DataFrame, color: str, *,
                              x_col: str, y_col: str,
                              x_label: str, y_label: str, title: str) -> None:
    valid = df.dropna(subset=[x_col, y_col])
    x = valid[x_col].values
    y = valid[y_col].values
    if x.size == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        return
    pr, sr, n = _pearson_spearman(x, y)
    lim_lo = min(float(x.min()), float(y.min()), -0.1) - 0.05
    lim_hi = max(float(x.max()), float(y.max()), 0.6) + 0.05
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], linestyle="--",
            color="#888888", linewidth=0.7, zorder=1)
    ax.axhline(0, color="#cccccc", linewidth=0.5, zorder=0)
    ax.axvline(0, color="#cccccc", linewidth=0.5, zorder=0)
    ax.scatter(
        x, y, s=8.0, color=color, alpha=SCATTER_ALPHA["new"],
        linewidths=0.0, rasterized=True, zorder=2,
    )
    # OLS fit only when there is enough non-degenerate data.
    if n >= 5 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.array([lim_lo, lim_hi])
        ax.plot(xs, slope * xs + intercept,
                color="#333333", linewidth=0.8, zorder=3)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(color="#e5e5e5", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.text(
        0.04, 0.96,
        f"n = {n}\nPearson r = {pr:.3f}\nSpearman " + "ρ" + f" = {sr:.3f}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=plt.rcParams["legend.fontsize"] - 0.5,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85),
    )


def figP12_model_consistency(decomposed: pd.DataFrame) -> Path:
    """2x2 grid: per-target RF r vs TabPFN r, within each data scope.

    Rows = scope (Original then MOSA expanded), cols = target family.
    Each scope keeps both models on the same train/test split so the
    scatter isolates learner agreement.
    """
    configure_nature_style("composite")
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.2))
    fig.subplots_adjust(left=0.10, right=0.985, top=0.93, bottom=0.10,
                        wspace=0.32, hspace=0.42)
    panel_letters = [["a", "b"], ["c", "d"]]
    for ri, (_scope_key, scope_label, x_col, y_col) in enumerate(CONSISTENCY_SCOPES):
        for ci, family in enumerate(FAMILY_ORDER):
            ax = axes[ri, ci]
            sub = decomposed[decomposed["target_family"] == family]
            _draw_consistency_scatter(
                ax, sub, FAMILY_COLORS[family],
                x_col=x_col, y_col=y_col,
                x_label="Random Forest  Pearson r",
                y_label="TabPFN  Pearson r",
                title=f"{FAMILY_DISPLAY[family]} — {scope_label}",
            )
            panel_label(ax, panel_letters[ri][ci], offset=(-0.20, 1.05))
    out = FIG_DIR / "figP12_model_consistency"
    save_figure(fig, out)
    return Path(str(out) + ".pdf")


def compute_model_consistency(decomposed: pd.DataFrame) -> pd.DataFrame:
    """Per (family, scope) Pearson and Spearman between RF and TabPFN
    per-target Pearson r vectors."""
    rows = []
    for scope_key, scope_label, x_col, y_col in CONSISTENCY_SCOPES:
        for family in FAMILY_ORDER:
            sub = decomposed[decomposed["target_family"] == family]
            x = sub[x_col].values
            y = sub[y_col].values
            pr, sr, n = _pearson_spearman(x, y)
            mask = ~(np.isnan(x) | np.isnan(y))
            xv, yv = x[mask], y[mask]
            rows.append({
                "target_family": family,
                "scope": scope_key,
                "scope_label": scope_label,
                "rf_column": x_col,
                "tabpfn_column": y_col,
                "n_targets": n,
                "pearson_r": pr,
                "spearman_rho": sr,
                "mean_rf_r": float(np.mean(xv)) if xv.size else float("nan"),
                "mean_tabpfn_r": float(np.mean(yv)) if yv.size else float("nan"),
                "frac_tabpfn_better": float(np.mean(yv > xv)) if yv.size else float("nan"),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def write_prediction_tables(summary_long: pd.DataFrame, per_target_long: pd.DataFrame,
                             decomposed: pd.DataFrame,
                             selected_targets: dict[str, list[str]] | None = None) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    # Per-condition summary with bootstrap CI + Wilcoxon p (TabPFN+MOSA vs RF baseline).
    rows = []
    long = per_target_long.copy()
    for (family, model, frame, variant), grp in long.groupby(
        ["target_family", "model_name", "sample_frame", "variant"], observed=True
    ):
        m, lo, hi = bootstrap_mean_ci(grp["test_pearsonr"].values)
        rows.append({
            "target_family": family,
            "model_name": model,
            "sample_frame": frame,
            "variant": variant,
            "n_targets": int((~grp["test_pearsonr"].isna()).sum()),
            "mean_pearsonr": m,
            "ci_lo": lo,
            "ci_hi": hi,
        })
    cond_summary = pd.DataFrame(rows)

    wilcoxon_rows = []
    for family in FAMILY_ORDER:
        sub = decomposed[decomposed["target_family"] == family]
        for label, ref_col, target_col in [
            ("TabPFN+MOSA vs RF baseline", "r_rf_orig", "r_tabpfn_mosa_all"),
            ("RF MOSA vs RF original (expanded)", "r_rf_orig_exp", "r_rf_mosa_all"),
            ("TabPFN MOSA vs TabPFN original (expanded)", "r_tabpfn_orig_exp", "r_tabpfn_mosa_all"),
            ("TabPFN vs RF on MOSA data", "r_rf_mosa_all", "r_tabpfn_mosa_all"),
        ]:
            p, n = wilcoxon_paired(sub[target_col], sub[ref_col])
            wilcoxon_rows.append({
                "target_family": family,
                "comparison": label,
                "n_paired": n,
                "wilcoxon_p": p,
                "mean_target": float(np.nanmean(sub[target_col])),
                "mean_ref": float(np.nanmean(sub[ref_col])),
            })
    wilcoxon_summary = pd.DataFrame(wilcoxon_rows)

    paths["prediction_summary_by_condition"] = FIG_DIR / "prediction_summary_by_condition.csv"
    cond_summary.to_csv(paths["prediction_summary_by_condition"], index=False)

    paths["prediction_wilcoxon_summary"] = FIG_DIR / "prediction_wilcoxon_summary.csv"
    wilcoxon_summary.to_csv(paths["prediction_wilcoxon_summary"], index=False)

    paths["prediction_per_target_decomposed"] = FIG_DIR / "prediction_per_target_decomposed.csv"
    decomposed.to_csv(paths["prediction_per_target_decomposed"], index=False)

    # Top gainers / losers per family (top-25 each direction).
    gain_rows = []
    for family in FAMILY_ORDER:
        sub = decomposed[decomposed["target_family"] == family].dropna(subset=["delta_headline"])
        gain_rows.append(sub.nlargest(25, "delta_headline").assign(direction="top_gain"))
        gain_rows.append(sub.nsmallest(25, "delta_headline").assign(direction="top_loss"))
    gain_df = pd.concat(gain_rows, ignore_index=True)
    paths["prediction_top_gainers"] = FIG_DIR / "prediction_top_gainers.csv"
    gain_df.to_csv(paths["prediction_top_gainers"], index=False)

    # Selected target residuals for the deep-dive (Fig P6).
    if selected_targets is None:
        selected_targets = select_deep_dive_targets(decomposed)
    residual_rows = []
    for family, targets in selected_targets.items():
        try:
            t_t, t_p = load_test_matrices(family, HEADLINE_FRAME, HEADLINE_VARIANT, "tabpfn")
            r_t, r_p = load_test_matrices(family, BASELINE_FRAME, BASELINE_VARIANT, "random_forest")
        except FileNotFoundError:
            continue
        for target in targets:
            if target not in t_t.columns:
                continue
            cells = t_t.index
            truth = t_t[target]
            tab = t_p[target]
            rf = r_p[target] if target in r_p.columns else pd.Series(np.nan, index=cells)
            for cell in cells:
                residual_rows.append({
                    "target_family": family,
                    "target_name": target,
                    "sample_id": cell,
                    "y_true": float(truth.loc[cell]) if not pd.isna(truth.loc[cell]) else np.nan,
                    "y_pred_tabpfn": float(tab.loc[cell]) if not pd.isna(tab.loc[cell]) else np.nan,
                    "y_pred_rf": float(rf.loc[cell]) if cell in rf.index and not pd.isna(rf.loc[cell]) else np.nan,
                })
    if residual_rows:
        paths["prediction_selected_target_residuals"] = FIG_DIR / "prediction_selected_target_residuals.csv"
        pd.DataFrame(residual_rows).to_csv(paths["prediction_selected_target_residuals"], index=False)

    # Imputation scope.
    scope = load_imputation_scope()
    paths["prediction_imputation_scope"] = FIG_DIR / "prediction_imputation_scope.csv"
    scope.to_csv(paths["prediction_imputation_scope"], index=False)

    # TabPFN vs RF consistency within each data scope (Fig P12).
    consistency = compute_model_consistency(decomposed)
    paths["prediction_model_consistency"] = FIG_DIR / "prediction_model_consistency.csv"
    consistency.to_csv(paths["prediction_model_consistency"], index=False)

    return paths


# ---------------------------------------------------------------------------
# Single-panel renderers
# ---------------------------------------------------------------------------


def _save_single(fig, stem: str) -> Path:
    out = SINGLE_FIG_DIR / stem
    save_figure(fig, out)
    return Path(str(out) + ".pdf")


def _bars_single(family: str, value_col: str, ci_cols: tuple[str, str] | None,
                 *, ylabel: str, title: str, summary_long: pd.DataFrame,
                 per_target_long: pd.DataFrame, fig_size=(4.0, 3.2),
                 stem: str) -> Path:
    configure_nature_style("column")
    fig, ax = plt.subplots(figsize=fig_size)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.88, bottom=0.22)
    cond_keys = list(HEADLINE_CONDITIONS)
    bar_w = 0.42
    if value_col == "mean_r":
        agg = _agg_with_ci(per_target_long)
        getv = lambda model, frame, variant: agg[
            (agg["target_family"] == family)
            & (agg["model_name"] == model)
            & (agg["sample_frame"] == frame)
            & (agg["variant"] == variant)
        ]
    else:
        getv = lambda model, frame, variant: summary_long[
            (summary_long["target_family"] == family)
            & (summary_long["model_name"] == model)
            & (summary_long["sample_frame"] == frame)
            & (summary_long["variant"] == variant)
        ]
    for model_idx, model in enumerate(MODEL_ORDER):
        offs = (model_idx - 0.5) * bar_w
        x = np.arange(len(cond_keys)) + offs
        ys, err_lo, err_hi = [], [], []
        for frame, variant in cond_keys:
            r = getv(model, frame, variant)
            if len(r) and not pd.isna(r[value_col].iloc[0]):
                v = float(r[value_col].iloc[0])
                ys.append(v)
                if ci_cols:
                    lo, hi = ci_cols
                    err_lo.append(max(v - float(r[lo].iloc[0]), 0))
                    err_hi.append(max(float(r[hi].iloc[0]) - v, 0))
            else:
                ys.append(np.nan); err_lo.append(0); err_hi.append(0)
        yerr = np.array([err_lo, err_hi]) if ci_cols else None
        ax.bar(
            x, ys, width=bar_w, color=MODEL_COLORS[model],
            edgecolor="black", linewidth=0.45,
            yerr=yerr, error_kw=dict(linewidth=0.7, ecolor="#333333", capsize=1.5),
            label=MODEL_DISPLAY[model],
        )
    ax.set_xticks(np.arange(len(cond_keys)))
    ax.set_xticklabels(
        [HEADLINE_CONDITION_LABEL[(f, v)] for f, v in cond_keys],
        fontsize=plt.rcParams["xtick.labelsize"],
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, handlelength=1.2, loc="best")
    return _save_single(fig, stem)


def single_figP1a_mean_r_crispr(summary_long, per_target_long) -> Path:
    return _bars_single("crisprcas9", "mean_r", ("ci_lo", "ci_hi"),
                        ylabel="Mean per-target Pearson r",
                        title=f"{FAMILY_DISPLAY['crisprcas9']} — mean Pearson r (95 % bootstrap CI)",
                        summary_long=summary_long, per_target_long=per_target_long,
                        stem="single_figP1a_mean_r_crispr")


def single_figP1b_mean_r_drug(summary_long, per_target_long) -> Path:
    return _bars_single("drugresponse", "mean_r", ("ci_lo", "ci_hi"),
                        ylabel="Mean per-target Pearson r",
                        title=f"{FAMILY_DISPLAY['drugresponse']} — mean Pearson r (95 % bootstrap CI)",
                        summary_long=summary_long, per_target_long=per_target_long,
                        stem="single_figP1b_mean_r_drug")


def single_figP1c_pooled_crispr(summary_long, per_target_long) -> Path:
    return _bars_single("crisprcas9", "pooled_test_pearsonr", None,
                        ylabel="Pooled Pearson r",
                        title=f"{FAMILY_DISPLAY['crisprcas9']} — pooled Pearson r",
                        summary_long=summary_long, per_target_long=per_target_long,
                        stem="single_figP1c_pooled_crispr")


def single_figP1d_pooled_drug(summary_long, per_target_long) -> Path:
    return _bars_single("drugresponse", "pooled_test_pearsonr", None,
                        ylabel="Pooled Pearson r",
                        title=f"{FAMILY_DISPLAY['drugresponse']} — pooled Pearson r",
                        summary_long=summary_long, per_target_long=per_target_long,
                        stem="single_figP1d_pooled_drug")


def single_figP1e_train_n_crispr(summary_long, per_target_long) -> Path:
    return _bars_single("crisprcas9", "train_n", None,
                        ylabel="Training cell lines",
                        title=f"{FAMILY_DISPLAY['crisprcas9']} — cohort size",
                        summary_long=summary_long, per_target_long=per_target_long,
                        stem="single_figP1e_train_n_crispr")


def single_figP1f_train_n_drug(summary_long, per_target_long) -> Path:
    return _bars_single("drugresponse", "train_n", None,
                        ylabel="Training cell lines",
                        title=f"{FAMILY_DISPLAY['drugresponse']} — cohort size",
                        summary_long=summary_long, per_target_long=per_target_long,
                        stem="single_figP1f_train_n_drug")


def single_figP2a_paired_crispr(decomposed: pd.DataFrame) -> Path:
    configure_nature_style("column")
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.92, bottom=0.16)
    sub = decomposed[decomposed["target_family"] == "crisprcas9"]
    _draw_paired_scatter(
        ax, sub, FAMILY_COLORS["crisprcas9"],
        x_col="r_rf_orig", y_col="r_tabpfn_mosa_all",
        x_label="RF (overlap × original)  Pearson r",
        y_label="TabPFN (expanded × MOSA)  Pearson r",
        title=FAMILY_DISPLAY["crisprcas9"],
    )
    return _save_single(fig, "single_figP2a_paired_crispr")


def single_figP2b_paired_drug(decomposed: pd.DataFrame) -> Path:
    configure_nature_style("column")
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.92, bottom=0.16)
    sub = decomposed[decomposed["target_family"] == "drugresponse"]
    _draw_paired_scatter(
        ax, sub, FAMILY_COLORS["drugresponse"],
        x_col="r_rf_orig", y_col="r_tabpfn_mosa_all",
        x_label="RF (overlap × original)  Pearson r",
        y_label="TabPFN (expanded × MOSA)  Pearson r",
        title=FAMILY_DISPLAY["drugresponse"],
    )
    return _save_single(fig, "single_figP2b_paired_drug")


def single_figP3a_distribution(decomposed: pd.DataFrame) -> Path:
    configure_nature_style("column")
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    fig.subplots_adjust(left=0.18, right=0.97, top=0.88, bottom=0.16)
    by_family = {
        family: decomposed.loc[decomposed["target_family"] == family, "delta_headline"].values
        for family in FAMILY_ORDER
    }
    _draw_violin(ax, by_family)
    ax.set_ylabel("ΔPearson r  (TabPFN+MOSA − RF baseline)")
    ax.set_title("Per-target ΔPearson r distribution")
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    return _save_single(fig, "single_figP3a_distribution")


def single_figP3b_winrate(decomposed: pd.DataFrame) -> Path:
    configure_nature_style("column")
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    fig.subplots_adjust(left=0.18, right=0.97, top=0.88, bottom=0.18)
    by_family = {
        family: decomposed.loc[decomposed["target_family"] == family, "delta_headline"].values
        for family in FAMILY_ORDER
    }
    _draw_winrate(ax, by_family)
    ax.set_title("Cumulative win-rate")
    return _save_single(fig, "single_figP3b_winrate")


def single_figP4a_heatmap_crispr(summary_long: pd.DataFrame) -> Path:
    configure_nature_style("column")
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    fig.subplots_adjust(left=0.30, right=0.97, top=0.85, bottom=0.20)
    _heatmap_panel(ax, summary_long, "crisprcas9", "test_pearsonr",
                   title=f"{FAMILY_DISPLAY['crisprcas9']} — mean per-target Pearson r")
    return _save_single(fig, "single_figP4a_heatmap_crispr")


def single_figP4b_heatmap_drug(summary_long: pd.DataFrame) -> Path:
    configure_nature_style("column")
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    fig.subplots_adjust(left=0.30, right=0.97, top=0.85, bottom=0.20)
    _heatmap_panel(ax, summary_long, "drugresponse", "test_pearsonr",
                   title=f"{FAMILY_DISPLAY['drugresponse']} — mean per-target Pearson r")
    return _save_single(fig, "single_figP4b_heatmap_drug")


def single_figP5_top_winners_crispr(decomposed: pd.DataFrame) -> Path:
    configure_nature_style("column")
    fig, ax = plt.subplots(figsize=(4.6, 6.4))
    fig.subplots_adjust(left=0.32, right=0.95, top=0.93, bottom=0.10)
    _draw_top_winners(ax, decomposed, "crisprcas9", top_n=25)
    return _save_single(fig, "single_figP5_top_winners_crispr")


def single_figP5_top_winners_drug(decomposed: pd.DataFrame) -> Path:
    configure_nature_style("column")
    fig, ax = plt.subplots(figsize=(4.6, 6.4))
    fig.subplots_adjust(left=0.32, right=0.95, top=0.93, bottom=0.10)
    _draw_top_winners(ax, decomposed, "drugresponse", top_n=25)
    return _save_single(fig, "single_figP5_top_winners_drug")


def render_singles_figP6_per_target(
    decomposed: pd.DataFrame,
    selected: dict[str, list[str]],
) -> list[Path]:
    """One slide-friendly y_true vs y_pred scatter per selected target."""
    out_paths: list[Path] = []
    for family, targets in selected.items():
        try:
            t_t, t_p = load_test_matrices(family, HEADLINE_FRAME, HEADLINE_VARIANT, "tabpfn")
            r_t, r_p = load_test_matrices(family, BASELINE_FRAME, BASELINE_VARIANT, "random_forest")
        except FileNotFoundError:
            continue
        for target in targets:
            if target not in t_t.columns:
                continue
            configure_nature_style("column")
            fig, ax = plt.subplots(figsize=(4.0, 4.0))
            fig.subplots_adjust(left=0.18, right=0.97, top=0.92, bottom=0.16)
            _scatter_truth_vs_pred(
                ax, t_t[target], t_p[target],
                r_p[target] if target in r_p.columns else pd.Series(np.nan, index=t_t.index),
                family, target,
            )
            out_paths.append(_save_single(fig, f"single_figP6_deepdive_{family}_{target}"))
    return out_paths


def single_figP7d_lever_summary(decomposed: pd.DataFrame) -> Path:
    configure_nature_style("column")
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.88, bottom=0.30)
    levers = [
        ("RF: original → MOSA", "delta_mosa_at_rf"),
        ("TabPFN: original → MOSA", "delta_mosa_at_tabpfn"),
        ("RF → TabPFN, original", "delta_model_at_orig"),
        ("RF → TabPFN, MOSA", "delta_model_at_mosa_all"),
    ]
    pos = 0
    bar_w = 0.35
    ticks, labels = [], []
    for label, col in levers:
        for offs_idx, family in enumerate(FAMILY_ORDER):
            sub = decomposed[decomposed["target_family"] == family][col].dropna().values
            m, lo, hi = bootstrap_mean_ci(sub)
            x = pos + (offs_idx - 0.5) * bar_w
            ax.bar(
                x, m, width=bar_w, color=FAMILY_COLORS[family],
                edgecolor="black", linewidth=0.45,
                yerr=[[max(m - lo, 0)], [max(hi - m, 0)]],
                error_kw=dict(linewidth=0.7, ecolor="#333333", capsize=1.5),
                label=FAMILY_DISPLAY[family] if pos == 0 else None,
            )
        ticks.append(pos)
        labels.append(label)
        pos += 1
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=20, ha="right",
                       fontsize=plt.rcParams["xtick.labelsize"] - 1.0)
    ax.axhline(0, color="#888888", linewidth=0.6, linestyle="--")
    ax.set_ylabel("Mean ΔPearson r")
    ax.set_title("Lever summary (95 % bootstrap CI)")
    ax.legend(frameon=False, handlelength=1.2, loc="best")
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    return _save_single(fig, "single_figP7d_lever_summary")


def single_figP8_strategy_crispr(decomposed: pd.DataFrame) -> Path:
    configure_nature_style("column")
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    fig.subplots_adjust(left=0.18, right=0.97, top=0.88, bottom=0.18)
    _draw_strategy_slope(ax, decomposed, "crisprcas9")
    return _save_single(fig, "single_figP8_strategy_crispr")


def single_figP8_strategy_drug(decomposed: pd.DataFrame) -> Path:
    configure_nature_style("column")
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    fig.subplots_adjust(left=0.18, right=0.97, top=0.88, bottom=0.18)
    _draw_strategy_slope(ax, decomposed, "drugresponse")
    return _save_single(fig, "single_figP8_strategy_drug")


def single_figP10a_scope(summary_long: pd.DataFrame) -> Path:
    configure_nature_style("column")
    scope = load_imputation_scope()
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    fig.subplots_adjust(left=0.30, right=0.96, top=0.90, bottom=0.28)
    omics = scope["omic_layer"].tolist()
    ypos = np.arange(len(omics))
    bar_h = 0.36
    ax.barh(
        ypos - bar_h / 2, scope["n_cells_measured"], height=bar_h,
        color="#888888", edgecolor="black", linewidth=0.4, label="Measured cells",
    )
    ax.barh(
        ypos + bar_h / 2, scope["n_cells_after_mosa"], height=bar_h,
        color=PALETTE["new"], edgecolor="black", linewidth=0.4, label="After MOSA",
    )
    ax.set_yticks(ypos)
    ax.set_yticklabels([OMIC_DISPLAY.get(o, o) for o in omics])
    ax.invert_yaxis()
    ax.set_xlabel("Cell lines")
    ax.set_title("Cohort coverage by omic layer")
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, handlelength=1.0, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2)
    for i, (m, mosa) in enumerate(zip(scope["n_cells_measured"], scope["n_cells_after_mosa"])):
        added = mosa - m
        if added > 0:
            ax.text(mosa + 5, i + bar_h / 2, f"+{added}",
                    va="center", fontsize=plt.rcParams["legend.fontsize"] - 0.5,
                    color=PALETTE["new"])
    return _save_single(fig, "single_figP10a_scope")


def _single_figP12_panel(decomposed: pd.DataFrame, *, family: str,
                          scope_key: str, scope_label: str,
                          x_col: str, y_col: str, stem: str) -> Path:
    configure_nature_style("column")
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.92, bottom=0.16)
    sub = decomposed[decomposed["target_family"] == family]
    _draw_consistency_scatter(
        ax, sub, FAMILY_COLORS[family],
        x_col=x_col, y_col=y_col,
        x_label="Random Forest  Pearson r",
        y_label="TabPFN  Pearson r",
        title=f"{FAMILY_DISPLAY[family]} — {scope_label}",
    )
    return _save_single(fig, stem)


def single_figP12a_consistency_crispr_orig(decomposed: pd.DataFrame) -> Path:
    return _single_figP12_panel(
        decomposed, family="crisprcas9", scope_key="orig", scope_label="Original",
        x_col="r_rf_orig", y_col="r_tabpfn_orig",
        stem="single_figP12a_consistency_crispr_orig",
    )


def single_figP12b_consistency_drug_orig(decomposed: pd.DataFrame) -> Path:
    return _single_figP12_panel(
        decomposed, family="drugresponse", scope_key="orig", scope_label="Original",
        x_col="r_rf_orig", y_col="r_tabpfn_orig",
        stem="single_figP12b_consistency_drug_orig",
    )


def single_figP12c_consistency_crispr_mosa(decomposed: pd.DataFrame) -> Path:
    return _single_figP12_panel(
        decomposed, family="crisprcas9", scope_key="mosa_all", scope_label="MOSA expanded",
        x_col="r_rf_mosa_all", y_col="r_tabpfn_mosa_all",
        stem="single_figP12c_consistency_crispr_mosa",
    )


def single_figP12d_consistency_drug_mosa(decomposed: pd.DataFrame) -> Path:
    return _single_figP12_panel(
        decomposed, family="drugresponse", scope_key="mosa_all", scope_label="MOSA expanded",
        x_col="r_rf_mosa_all", y_col="r_tabpfn_mosa_all",
        stem="single_figP12d_consistency_drug_mosa",
    )


def render_all_singles(summary_long: pd.DataFrame,
                        per_target_long: pd.DataFrame,
                        decomposed: pd.DataFrame,
                        selected_targets: dict[str, list[str]]) -> list[Path]:
    paths: list[Path] = [
        single_figP1a_mean_r_crispr(summary_long, per_target_long),
        single_figP1b_mean_r_drug(summary_long, per_target_long),
        single_figP1c_pooled_crispr(summary_long, per_target_long),
        single_figP1d_pooled_drug(summary_long, per_target_long),
        single_figP1e_train_n_crispr(summary_long, per_target_long),
        single_figP1f_train_n_drug(summary_long, per_target_long),
        single_figP2a_paired_crispr(decomposed),
        single_figP2b_paired_drug(decomposed),
        single_figP3a_distribution(decomposed),
        single_figP3b_winrate(decomposed),
        single_figP4a_heatmap_crispr(summary_long),
        single_figP4b_heatmap_drug(summary_long),
        single_figP5_top_winners_crispr(decomposed),
        single_figP5_top_winners_drug(decomposed),
    ]
    paths.extend(render_singles_figP6_per_target(decomposed, selected_targets))
    paths.extend([
        single_figP7d_lever_summary(decomposed),
        single_figP8_strategy_crispr(decomposed),
        single_figP8_strategy_drug(decomposed),
        single_figP10a_scope(summary_long),
        single_figP12a_consistency_crispr_orig(decomposed),
        single_figP12b_consistency_drug_orig(decomposed),
        single_figP12c_consistency_crispr_mosa(decomposed),
        single_figP12d_consistency_drug_mosa(decomposed),
    ])
    return paths


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_all() -> None:
    summary_long = load_summary_long()
    per_target_long = load_per_target_long()
    decomposed = load_decomposed()

    p1 = figP1_headline_grid(summary_long, per_target_long)
    p2 = figP2_paired_scatter(decomposed)
    p3 = figP3_decomposition(decomposed)
    p4 = figP4_strategy_frame(summary_long)
    p5 = figP5_top_winners(decomposed)
    p6, selected = figP6_target_deepdives(decomposed)
    p7 = figP7_helps_most(decomposed)
    p8 = figP8_strategy_per_target(decomposed)
    try:
        p9, p9_summary = figP9_imputation_accuracy()
        p9_summary.to_csv(FIG_DIR / "prediction_imputation_accuracy.csv", index=False)
    except FileNotFoundError as exc:
        print(f"figP9 skipped: {exc}")
        p9 = None
    p10 = figP10_imputation_scope(summary_long)
    p11 = figP11_cnv_ablation()
    p12 = figP12_model_consistency(decomposed)

    table_paths = write_prediction_tables(
        summary_long, per_target_long, decomposed, selected_targets=selected,
    )

    for label, p in [
        ("figP1", p1), ("figP2", p2), ("figP3", p3), ("figP4", p4),
        ("figP5", p5), ("figP6", p6), ("figP7", p7), ("figP8", p8),
        ("figP9", p9), ("figP10", p10), ("figP11", p11), ("figP12", p12),
    ]:
        if p is not None:
            print(f"{label}: {p.relative_to(ROOT)}")

    print(f"Selected deep-dive targets: {selected}")
    print()
    print("Tables:")
    for k, v in table_paths.items():
        print(f"  {k}: {v.relative_to(ROOT)}")

    print()
    print("Single-panel figures:")
    for p in render_all_singles(summary_long, per_target_long, decomposed, selected):
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    run_all()
