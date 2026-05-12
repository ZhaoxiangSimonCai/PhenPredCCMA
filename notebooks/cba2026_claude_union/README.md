# CBA 2026 — SHAP + prediction analyses (publication-quality)

Companion to `docs/CBA_Abstract2026_v1.docx`. **Union variant** (`min_views_per_sample=1`, ~425 cell lines) of the +CNV MOSA run, timestamp `20260511_174623`. The ≥2-omic version lives in `notebooks/cba2026_claude/` and remains the tracker-selected canonical version until/unless this variant is promoted. Two parallel suites:

- **SHAP suite** (Figs 1–8): which features the model learns to use, mapped
  back to known biology. Outputs at `reports/cba2026_claude_union/shap_analysis/20260511_174623/`.
- **Prediction suite** (Figs P1–P11): how well TabPFN+MOSA predicts vs RF on
  original data, where the gain comes from, and whether MOSA imputation is
  itself faithful. Outputs at `reports/cba2026_claude_union/prediction_analysis/20260511_174623/`.

For talk-prep status, slide-mapping crib and cross-suite paths see
`docs/CBA2026_talk_prep.md`.

## Files

- `_plot_style.py` — copied from the `nature-plots` skill.
  - Okabe-Ito-derived colourblind-safe palette (`PALETTE`).
  - Three Nature type scales (`composite` / `column` / `full`).
  - `save_figure()` writes PNG @ 300 dpi *plus* PDF with Type-42 (TrueType)
    fonts so labels stay editable in Illustrator / Inkscape.
- `_shap_analysis.py` — every SHAP figure and table is built by a function
  in this module so the two SHAP notebooks stay short and the visual style
  is shared. Run directly (`python _shap_analysis.py`) to regenerate.
- `_prediction_analysis.py` — analogous module for the prediction suite.
  Loaders (`load_summary_long`, `load_per_target_long`, `load_decomposed`),
  figure builders (`figP1`–`figP11`), table writer (`write_prediction_tables`),
  and slide-friendly singles. Run directly (`python _prediction_analysis.py`)
  for one-shot regeneration.
- `01_global_shap_landscape.ipynb` — global landscape figures (Figs 1, 2,
  3, 8) and copy-number tables.
- `02_target_level_shap_performance.ipynb` — target-level SHAP profile vs
  selected TabPFN performance (Figs 4, 5, 6, 7) and follow-up tables.
- `03_headline_performance.ipynb` — Figs P1–P4: aggregate model and
  augmentation comparison driving the abstract's headline numbers.
- `04_per_target_prediction.ipynb` — Figs P5–P8: per-target winners,
  selected-target deep-dives (MDM2, MDM4, top gainers), and lever
  decomposition.
- `05_mosa_imputation_quality.ipynb` — Figs P9–P11: imputation accuracy
  on held-out cells, imputation scope, and CNV-vs-no-CNV ablation
  (`20260313_162348` reference).

## Figure inventory — SHAP suite

Composite figures live directly in `reports/cba2026_claude_union/shap_analysis/<TIMESTAMP>/`. Single-panel slide-friendly versions of every panel live alongside in `…/singles/` (see below).

| File | Replaces | What it shows |
| --- | --- | --- |
| `fig1_global_landscape.{png,pdf}` | original 1+2 | Composite: omic-layer mean \|SHAP\|, within-family share, top-25 features per family. |
| `fig2_global_feature_heatmap.{png,pdf}` | original 3 | Top-feature × family heatmap with a dedicated omic-layer swatch column. |
| `fig3_target_omic_composition.{png,pdf}` | original 4 | Per-target distribution of top-200 SHAP share by omic layer (violin + strip + median bar). |
| `fig4_performance_vs_profile.{png,pdf}` | original 5 | Four-panel diagnostic: Pearson r distribution, CNV-share vs r scatter, omic-entropy vs Δr scatter, dominant-omic boxplot. |
| `fig5_selected_crispr_top_features.{png,pdf}` | original 6 | Top-18 SHAP features per requested CRISPR target. TP53 is requested but absent from the SHAP export, so the panel is backfilled with the top-Δr CRISPR target (PSMA2). |
| `fig6_selected_crispr_omic_composition.{png,pdf}` | original 7 | Heatmap of top-200 SHAP omic-layer share for the same selected targets. |
| `fig7_cnv_gain_volcano.{png,pdf}` | **new** | ΔPearson r vs CNV share of top-200 SHAP, callout-labelled top-10 gainers per family. |
| `fig8_shap_concentration.{png,pdf}` | **new** | Cumulative SHAP share vs feature rank (Lorenz-style), per-family median ± 10–90 % band. |

### Single-panel folder (`singles/`)

Every panel of every composite is also written out as its own standalone PNG + PDF, sized for slides / single-figure inserts. Type scale switches from `composite` to `column` so the typography stays correct when a single panel is shown larger than its slot in the composite.

| File (in `singles/`) | Source panel |
| --- | --- |
| `single_fig1a_omic_absolute.{png,pdf}` | Fig 1a — omic-layer absolute SHAP |
| `single_fig1b_omic_share.{png,pdf}` | Fig 1b — within-family share |
| `single_fig1c_top_features_crispr.{png,pdf}` | Fig 1c — CRISPR top-25 |
| `single_fig1d_top_features_drug.{png,pdf}` | Fig 1d — drug top-25 |
| `single_fig2_global_feature_heatmap.{png,pdf}` | Fig 2 — feature-union heatmap |
| `single_fig3a_composition_crispr.{png,pdf}` | Fig 3a — CRISPR violin+strip |
| `single_fig3b_composition_drug.{png,pdf}` | Fig 3b — drug violin+strip |
| `single_fig4a_performance_distribution.{png,pdf}` | Fig 4a — performance histogram |
| `single_fig4b_cnv_share_scatter.{png,pdf}` | Fig 4b — CNV share vs r |
| `single_fig4c_entropy_gain_scatter.{png,pdf}` | Fig 4c — entropy vs Δr |
| `single_fig4d_dominant_omic_box.{png,pdf}` | Fig 4d — dominant omic boxplot |
| `single_fig5_top_features_<TARGET>.{png,pdf}` | Fig 5 — one file per plotted CRISPR target (MDM2, MDM4, PSMA2) |
| `single_fig6_selected_crispr_omic_composition.{png,pdf}` | Fig 6 — selected CRISPR omic share |
| `single_fig7a_volcano_crispr.{png,pdf}` | Fig 7a — CRISPR CNV-gain volcano |
| `single_fig7b_volcano_drug.{png,pdf}` | Fig 7b — drug CNV-gain volcano |
| `single_fig8a_concentration_crispr.{png,pdf}` | Fig 8a — CRISPR concentration curve |
| `single_fig8b_concentration_drug.{png,pdf}` | Fig 8b — drug concentration curve |

## Table inventory — SHAP suite

- `selected_crispr_top_features.csv` — long-format dump of the bars in Fig 5.
- `target_level_shap_performance_summary.csv` — per-target SHAP profile + selected TabPFN performance + Δ vs RF baseline. Drives Figs 4, 7.
- `targets_high_copynumber_shap_share.csv` — top 25 per-family CNV-share targets.
- `targets_top_gain_loss_vs_rf_baseline.csv` — top 20 gain / loss targets per family relative to the RF `overlap × original` baseline.
- `table_global_copynumber_feature_ranking.csv`, `table_copynumber_feature_frequency_top200.csv`, `table_target_copynumber_share_top200.csv` — same content as in the original folder, regenerated through the shared module so there is a single source of truth.

## Figure inventory — prediction suite

Composites at `reports/cba2026_claude_union/prediction_analysis/<TIMESTAMP>/`. Single-panel slide-friendly exports under `…/singles/`.

| File | What it shows |
| --- | --- |
| `figP1_headline_performance.{png,pdf}` | 8-panel grid — mean per-target Pearson r (95 % bootstrap CI), mean R², pooled Pearson r, train_n. Three variants × two frames × {TabPFN, RF}, faceted by family. The abstract's headline numbers live here. |
| `figP2_paired_scatter.{png,pdf}` | Per-target paired RF↔TabPFN Pearson r scatter (CRISPR / drug). y=x diagonal, n_above + Wilcoxon p annotated. |
| `figP3_decomposition.{png,pdf}` | ΔPearson r = (TabPFN+MOSA − RF baseline). Distribution, cumulative win-rate, MOSA-only contribution per model, model-only contribution per data state. |
| `figP4_strategy_frame.{png,pdf}` | Heatmap of mean Pearson r over (variant × {model × frame}) and the matching cohort-size bar. |
| `figP5_top_winners.{png,pdf}` | Top-25 ΔPearson r gainers per family. Bars = ΔPearson r; ticks = absolute Pearson r at RF baseline (black) and TabPFN+MOSA (focus colour). |
| `figP6_target_deepdives.{png,pdf}` | y_true vs y_pred for MDM2, MDM4 (TP53 biomarkers in the abstract) + top CRISPR gainer + top three drug gainers. RF baseline overlaid in grey. |
| `figP7_helps_most.{png,pdf}` | Scatter set: ΔPearson r vs RF baseline r, vs n_test, MOSA-effect × model-effect quadrant, plus a per-lever bootstrap-mean summary. |
| `figP8_strategy_per_target.{png,pdf}` | Per-target slope plot across {original, MOSA gaps only, MOSA all} at expanded sample frame; faint per-target lines + per-model median trajectory. |
| `figP9_imputation_accuracy.{png,pdf}` | Held-out CCMA cells, measured vs MOSA-imputed scatter per omic, Pearson r and n in each panel. |
| `figP10_imputation_scope.{png,pdf}` | Cohort coverage per omic before/after MOSA + a performance-vs-cohort-size scatter. |
| `figP11_cnv_ablation.{png,pdf}` | CNV-vs-no-CNV MOSA — bar of mean Pearson r at the headline condition + paired per-target scatter. Compares `20260313_162348` (no-CNV) against `20260511_174623` (with-CNV). |

### Single-panel folder (`prediction_analysis/<TIMESTAMP>/singles/`)

| File (in `singles/`) | Source panel |
| --- | --- |
| `single_figP1{a,b}_mean_r_{crispr,drug}.{png,pdf}` | Fig P1 — mean per-target Pearson r per family |
| `single_figP1{c,d}_pooled_{crispr,drug}.{png,pdf}` | Fig P1 — pooled Pearson r per family |
| `single_figP1{e,f}_train_n_{crispr,drug}.{png,pdf}` | Fig P1 — cohort size per family |
| `single_figP2{a,b}_paired_{crispr,drug}.{png,pdf}` | Fig P2 — per-family paired scatter |
| `single_figP3a_distribution.{png,pdf}` | Fig P3 — ΔPearson r distribution |
| `single_figP3b_winrate.{png,pdf}` | Fig P3 — cumulative win-rate |
| `single_figP4{a,b}_heatmap_{crispr,drug}.{png,pdf}` | Fig P4 — variant × frame heatmap |
| `single_figP5_top_winners_{crispr,drug}.{png,pdf}` | Fig P5 — top-25 gainers per family |
| `single_figP6_deepdive_{family}_{target}.{png,pdf}` | Fig P6 — one file per deep-dive target (MDM2, MDM4, PSMA2, KN_93, SR_4835, Tozasertib_VX_680_MK_0457) |
| `single_figP7d_lever_summary.{png,pdf}` | Fig P7d — per-lever bootstrap-mean ΔPearson r |
| `single_figP8_strategy_{crispr,drug}.{png,pdf}` | Fig P8 — augmentation slope per family |
| `single_figP10a_scope.{png,pdf}` | Fig P10a — per-omic cohort coverage |

## Table inventory — prediction suite

- `prediction_summary_by_condition.csv` — per (family × model × frame × variant): n_targets, mean Pearson r, 95 % bootstrap CI.
- `prediction_wilcoxon_summary.csv` — paired Wilcoxon p-values for the four key comparisons (TabPFN+MOSA vs RF baseline, RF MOSA vs RF original, TabPFN MOSA vs TabPFN original, TabPFN vs RF on MOSA data) per family.
- `prediction_per_target_decomposed.csv` — per-target wide frame with the four levers (RF/orig, RF/MOSA, TabPFN/orig, TabPFN/MOSA) plus all Δ derivations. Drives Figs P2, P3, P5–P8.
- `prediction_top_gainers.csv` — top-25 ΔPearson r gainers and losers per family relative to the RF baseline.
- `prediction_selected_target_residuals.csv` — long-format y_true / y_pred for the Fig P6 deep-dive panels.
- `prediction_imputation_accuracy.csv` — per-omic Pearson r between measured and MOSA-imputed values on held-out cells.
- `prediction_imputation_scope.csv` — per-omic cohort coverage (measured vs after-MOSA cell counts).

## Visual conventions

- **Focus / reference framing**: CRISPR-Cas9 in deep blue (`PALETTE["new"]`), drug response in orange (`PALETTE["lost"]`). Brick-red (`PALETTE["highlight"]`) for significance markers and outlined points only.
- **Per-omic-layer colour mapping** (used wherever multiple layers are shown together):
  transcriptomics — Okabe blue · methylation — green · drug response — vermillion · CRISPR-Cas9 — pink · conditionals — grey · copy number — orange.
- **Legends** are placed outside the data area when they would otherwise land on it (Fig 1, Fig 2, Fig 5).
- **Callout columns** (Fig 7) keep all annotation text out of the scatter cloud — labels stack at evenly-spaced axes-relative positions and connect to markers via thin leader lines.
- **Rasterisation** is enabled on every dense scatter (Figs 4, 7) so PDFs stay small while text remains vector-editable.

## Standards-compliance verification

```
python -c "
import re; from pathlib import Path; from PIL import Image
d = Path('reports/cba2026_claude_union/shap_analysis/20260511_174623')
for png in sorted(d.glob('fig*.png')):
    print(png.name, Image.open(png).info.get('dpi'))
for pdf in sorted(d.glob('fig*.pdf')):
    subs = set(s.decode() for s in re.findall(rb'/Subtype\s*/(\w+)', pdf.read_bytes()))
    print(pdf.name, subs, 'Type3' in subs)
"
```
Every PNG reports ≈300 dpi; every PDF reports `{Type0, CIDFontType2}` (Type-42), never `Type3`.

## Reproducing the report

```
cd notebooks/cba2026_claude_union
python _shap_analysis.py            # one-shot regeneration
# or open either notebook for an interactive walk-through.
```
