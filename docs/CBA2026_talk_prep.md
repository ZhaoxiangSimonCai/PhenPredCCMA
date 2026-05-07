# CBA 2026 — talk preparation notes

Companion to the abstract `docs/CBA_Abstract2026_v1.docx`. Lives in `docs/`
because it tracks the talk as a whole, not any single notebook folder.

## Abstract — quick reference

**Title**: *Augmenting the Childhood Cancer Cell Map through Multi-Omic Data
Synthesis and Predictive Modelling.*

**Authors**: Zhaoxiang Cai, Dingyin Sun, Vincent Senyang Xue, Ron Firestein,
Claire Xin Sun.

**Headline numbers** (mean per-target Pearson r):

| Comparison | CRISPR | Drug response |
| --- | --- | --- |
| RF + original (baseline) | 0.15 | 0.23 |
| TabPFN + MOSA-augmented (headline) | 0.21 | 0.32 |
| **Δ over RF baseline** | **+41.6 %** | **+37.8 %** |
| MOSA also lifts RF (vs RF + original) | **+29.9 %** | **+14.7 %** |

SHAP recovered known biomarkers, including TP53 mutation status driving
MDM2 / MDM4 dependency.

## Selected MOSA training run

`20260505_131645` — MOSA + CNV training. Selected over `20260313_162348`
(no-CNV) because adding copy-number lifts both modalities (validated in
Fig P11). All notebooks in `notebooks/cba2026_claude/` and all reports
under `reports/cba2026_claude/` reference this timestamp.

## Status — what's done

### SHAP suite (already on disk)
Built in `notebooks/cba2026_claude/`. See `notebooks/cba2026_claude/README.md`
for full inventory. Outputs: `reports/cba2026_claude/shap_analysis/20260505_131645/`.

| Fig | What it shows |
| --- | --- |
| Fig 1 | Global landscape: omic-layer mean \|SHAP\|, within-family share, top-25 features per family. |
| Fig 2 | Top-feature × family heatmap (omic swatch column on the left). |
| Fig 3 | Per-target distribution of top-200 SHAP share by omic layer. |
| Fig 4 | Performance vs SHAP profile (4-panel diagnostic). |
| Fig 5 | Top-18 SHAP features per requested CRISPR target (MDM2, MDM4, top-Δr backfill). |
| Fig 6 | Heatmap of top-200 SHAP omic share for the same selected targets. |
| Fig 7 | ΔPearson r vs CNV share volcano with callout-labelled top-10 gainers. |
| Fig 8 | Cumulative SHAP share vs feature rank (Lorenz). |

### Prediction suite (this work)
Built in `notebooks/cba2026_claude/`. Outputs land at
`reports/cba2026_claude/prediction_analysis/20260505_131645/` with
single-panel exports under `singles/`.

| Fig | Notebook | What it shows |
| --- | --- | --- |
| P1 | 03 | Headline grid: mean Pearson r, R², pooled Pearson r, train_n by 3 variants × 2 frames × {TabPFN, RF} per family. |
| P2 | 03 | Per-target paired RF↔TabPFN Pearson r scatter (CRISPR / drug). |
| P3 | 03 | ΔPearson r decomposition: distribution, win-rate curve, MOSA-only and model-only contributions. |
| P4 | 03 | Variant × frame heatmap of mean Pearson r and the matching train_n cohort-expansion bar. |
| P5 | 04 | Top-25 ΔPearson r gainers per family (RF baseline r as point overlay). |
| P6 | 04 | y_true vs y_pred deep-dives for MDM2, MDM4 + top CRISPR gainer + top three drug gainers. |
| P7 | 04 | Where TabPFN + MOSA helps most: vs baseline r, vs target activity, vs n_test, plus a model-vs-data Δ quadrant. |
| P8 | 04 | Per-target slope across {original, mosa_nan_only, mosa_all}. |
| P9 | 05 | MOSA imputation accuracy on held-out cells, predicted vs measured per omic. |
| P10 | 05 | Imputation scope — cells × omics fill heatmap and per-omic n_imputed bars. |
| P11 | 05 | CNV-vs-no-CNV MOSA ablation (`20260505_131645` vs `20260313_162348`). |

## Slide-mapping crib (10–12 slide outline for a 15-min talk)

1. **Problem** — paediatric cancer mortality + CCMA coverage gap (no figure; just stats from abstract).
2. **MOSA in one slide** — diagram + Fig P10 (imputation scope) showing what MOSA fills in.
3. **Imputation faithfulness** — Fig P9 (imputation accuracy across omics).
4. **Headline result** — Fig P1 (or its single P1a) — the 0.21 vs 0.15 / 0.32 vs 0.23 contrast.
5. **Decomposition** — Fig P3 (or singles P3a + P3b) — disentangle MOSA effect from model effect.
6. **Per-target evidence** — Fig P2 (paired scatter) showing the joint distribution shifts above the diagonal.
7. **Top winners** — Fig P5 (top-25 gainers per family), call out abstract-named targets.
8. **Worked example** — Fig P6 panels for MDM2 / MDM4 (TabPFN tightens up the residual story).
9. **What does the model "see"?** — pivot to SHAP: Fig 1 + Fig 2 (omic-layer landscape).
10. **Biology check** — Fig 5 (top features for MDM2 / MDM4) — TP53 features dominate.
11. **CNV pays its way** — Fig 7 (CNV-gain volcano) and / or Fig P11 (CNV ablation).
12. **Take-home** — generative augmentation × foundation models = practical framework for paediatric functional genomics. Conference-style summary slide.

If running tight: drop slide 11 (CNV detail) and merge slide 4 + 5.
If running long: add Fig P7 (regime-where-it-wins) between slides 6 and 7, and Fig 8 (SHAP concentration / Lorenz) before slide 11.

## Paths — single source of truth

| Purpose | Path |
| --- | --- |
| Selected aggregate comparison | `reports/model_comparison_cnv_mosa_only/feature_augmentation/20260505_131645/summary_model_comparison.csv` |
| Per-target comparison | `…/per_target_model_comparison.csv` |
| Per-target combined (long) | `…/combined_per_target.csv` |
| Per-condition combined (long) | `…/combined_summary.csv` |
| TabPFN per-run | `reports/tabpfn_cnv_mosa_only/feature_augmentation/20260505_131645/{family}/{frame}/{variant}/` |
| RF per-run | `reports/random_forest_cnv_mosa_only/feature_augmentation/20260505_131645/{family}/{frame}/{variant}/` |
| MOSA imputed matrices | `reports/vae/files/20260505_131645_imputed_{omic}{,_test_all,_test_nans_only,_train_all,_train_nans_only,_cvtest}.csv.gz` |
| Original measured CCMA splits | `data/clines/ccma_processed/{omic}_ccma{,_mosa_test,_mosa_train,_overlap_test,_overlap_train}.csv` |
| CNV-vs-no-CNV ablation reference | `reports/model_comparison/feature_augmentation/20260313_162348/` |
| SHAP analysis outputs | `reports/cba2026_claude/shap_analysis/20260505_131645/` |
| Prediction analysis outputs | `reports/cba2026_claude/prediction_analysis/20260505_131645/` |
| Talk asset folder (composites) | `reports/cba2026_claude/{shap,prediction}_analysis/20260505_131645/fig*.{png,pdf}` |
| Talk asset folder (singles) | `…/singles/single_*.{png,pdf}` |

## Reproducing the prediction suite

```
cd notebooks/cba2026_claude
python _prediction_analysis.py    # one-shot regeneration of every figure + table
# or open notebooks 03 / 04 / 05 for an interactive walk-through.
```

Same standards-compliance check as the SHAP suite (300 dpi PNG, Type-42 PDF):

```
python -c "
import re; from pathlib import Path; from PIL import Image
d = Path('reports/cba2026_claude/prediction_analysis/20260505_131645')
for png in sorted(d.glob('fig*.png')):
    print(png.name, Image.open(png).info.get('dpi'))
for pdf in sorted(d.glob('fig*.pdf')):
    subs = set(s.decode() for s in re.findall(rb'/Subtype\s*/(\w+)', pdf.read_bytes()))
    print(pdf.name, subs, 'Type3' in subs)
"
```
