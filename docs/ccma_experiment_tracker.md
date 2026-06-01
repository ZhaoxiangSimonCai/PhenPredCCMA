# CCMA Experiment Tracker

Last updated: 2026-05-18

## Current Selected Version

Use the **union variant** (`min_views_per_sample=1`, ~425 cell lines) of the copy-number-augmented `feature_augmentation` downstream experiment from MOSA timestamp `20260511_174623`.

2026-05-17 confirmation: the CCMA-paper-style per-target correlation feature-selection follow-up (`reports/*_cnv_mosa_only_union_ftsel/...`, runbook `docs/ccma_runs/feature_selection_experiment.md`) was completed and **not promoted**. TabPFN deltas on the selected combo (`expanded/mosa_all`) were within noise (drug +0.0018, CRISPR −0.0025). RF gains were larger (drug +0.0315, CRISPR +0.0197) but RF is not the selected downstream model. The original block-variance pipeline stays as the selected version.

Selected final model/setting:

- Model: standard TabPFN, not the finetuned TabPFN branch
- Downstream experiment: `feature_augmentation`
- Selected sample frame and variant: `expanded` / `mosa_all`
- Upstream MOSA run: `20260511_174623`, `ccma_scratch_cnv_union`
- Sample-inclusion criterion: `min_views_per_sample = 1` (union of all CCMA cell lines with at least one omic; ~425 augmented cell lines, MOSA-train 421 / test 24 unchanged from the prior holdout)
- Added upstream view: `copynumber`, preprocessed from `data/CCMA/cnv.csv`
- Downstream predictor blocks: unchanged from the selected non-CNV workflow; CNV is used through MOSA imputation, not as a direct downstream predictor block
- Baseline used in the final comparison: random forest `overlap` / `original`
- Metric reported in the final figure: per-target macro mean Pearson r
- Target panel: 500 drug-response targets and 500 CRISPR-Cas12 targets
- Selected feature count: 2000
- Seed: 42

Primary artifacts:

- Upstream MOSA config: `reports/vae/configs/hyperparameters_ccma_scratch_cnv_union.json` (`min_views_per_sample=1`, `dataname=ccma_scratch_cnv_union`)
- Upstream MOSA config (archived): `reports/vae/configs/history/20260511_174623_hyperparameters.json`
- Upstream MOSA outputs: `reports/vae/files/20260511_174623_*`
- CNV preprocessing summary: `data/clines/ccma_processed/copynumber_ccma_preprocess_summary.json`
- Selected TabPFN outputs:
  - `reports/tabpfn_cnv_mosa_only_union/feature_augmentation/20260511_174623/drugresponse/expanded/mosa_all/`
  - `reports/tabpfn_cnv_mosa_only_union/feature_augmentation/20260511_174623/crisprcas9/expanded/mosa_all/`
- RF baseline outputs:
  - `reports/random_forest_cnv_mosa_only_union/feature_augmentation/20260511_174623/drugresponse/overlap/original/`
  - `reports/random_forest_cnv_mosa_only_union/feature_augmentation/20260511_174623/crisprcas9/overlap/original/`
- Cross-model summary for the selected union comparison: `reports/model_comparison_cnv_mosa_only_union/feature_augmentation/20260511_174623/summary_model_comparison.csv`
- Cross-model per-target data: `reports/model_comparison_cnv_mosa_only_union/feature_augmentation/20260511_174623/combined_per_target.csv`
- Aggregate comparison plot: `reports/model_comparison_cnv_mosa_only_union/feature_augmentation/20260511_174623/aggregate_model_comparison.png`
- Notebooks (figure generation): `notebooks/cba2026_claude_union/`
- Runbook: `docs/ccma_runs/union_variant_experiment.md`

## Current Selected Results

These are the headline values for the selected union variant (≥1 omic). The "Previous selected TabPFN r" column refers to the prior ≥2-omic CNV run (`20260505_131645`); the "Delta vs previous" column is union − ≥2.

| Target family | RF overlap/original baseline | RF expanded/mosa_all | TabPFN overlap/original | Selected TabPFN expanded/mosa_all | Selected gain vs RF baseline | Per-target improvement vs RF baseline | Previous selected TabPFN r | Delta vs previous |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Drug Response | 0.2346 | 0.2856 | 0.2207 | 0.3481 | +0.1135 (+48.4%) | 74.4% improved; median delta r = 0.1145 | 0.3194 | +0.0287 |
| CRISPR-Cas12 | 0.1510 | 0.1973 | 0.1436 | 0.2195 | +0.0685 (+45.4%) | 65.2% improved; median delta r = 0.0675 | 0.2124 | +0.0071 |

Selected TabPFN run details:

| Target family | Train n | Test n | Targets | Test observations | Test Pearson r | Test R2 | Test RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Drug Response | 244 | 24 | 500 | 11951 | 0.3481 | 0.0513 | 0.1672 |
| CRISPR-Cas12 | 140 | 24 | 500 | 11932 | 0.2195 | 0.0024 | 0.6472 |

## Copy Number / CNV Status

Copy number is used in the current selected upstream MOSA run (`20260511_174623`).

CNV preprocessing:

- Raw input: `data/CCMA/cnv.csv`
- Processed matrix: `data/clines/ccma_processed/copynumber_ccma.csv`
- Split matrices:
  - `data/clines/ccma_processed/copynumber_ccma_mosa_train.csv`
  - `data/clines/ccma_processed/copynumber_ccma_mosa_test.csv`
- Feature filtering: curated `cancer_driver=True` genes from `data/clines/mutations_summary_20230202.csv`
- Raw CNV genes: 42,416
- Retained CNV genes: 606
- CNV samples retained: 322

Encoding used:

| `cnvkit__copy_no` | MOSA `copynumber` state |
| ---: | ---: |
| 0 | -2 |
| 1 | -1 |
| 2 or missing event | 0 |
| 3 | 1 |
| >=4 | 2 |

The selected downstream feature-augmentation configs still use the same direct predictor blocks as the previous selected run:

- Drug Response predictors: `transcriptomics`, `methylation`, `crisprcas9`, `mutations`
- CRISPR-Cas12 predictors: `transcriptomics`, `methylation`, `drugresponse`, `mutations`

So CNV enters the selected workflow through MOSA's learned/imputed views, not as a direct TabPFN/RF raw predictor.

## Previous Selected Version

The previous selected version was the ≥2-omic CNV `feature_augmentation` downstream experiment from MOSA timestamp `20260505_131645`.

Key artifacts:

- Upstream MOSA config: `reports/vae/configs/hyperparameters_ccma_scratch_cnv.json` (`min_views_per_sample=2`, `dataname=ccma_scratch_cnv`)
- Upstream MOSA config (archived): `reports/vae/configs/history/20260505_131645_hyperparameters.json`
- Upstream MOSA outputs: `reports/vae/files/20260505_131645_*`
- Cross-model summary: `reports/model_comparison_cnv_mosa_only/feature_augmentation/20260505_131645/summary_model_comparison.csv`
- Aggregate comparison plot: `reports/model_comparison_cnv_mosa_only/feature_augmentation/20260505_131645/aggregate_model_comparison.png`

Previous selected values:

| Target family | Selected TabPFN expanded/mosa_all r | RF overlap/original baseline | Notes |
| --- | ---: | ---: | --- |
| Drug Response | 0.3194 | 0.2346 | Same upstream views as the union variant; differs only in `min_views_per_sample=2` (vs 1). Train n=232 (vs 244 under union). |
| CRISPR-Cas12 | 0.2124 | 0.1510 | Same upstream views as the union variant; train n=132 (vs 140 under union). |

That upstream MOSA config used `crisprcas9`, `drugresponse`, `transcriptomics`, `methylation`, and `copynumber`, with `mutations` as labels/conditionals — identical to the current selected union variant. Promotion to the union variant on 2026-05-11 was driven by the +0.0287 (drug) and +0.0071 (CRISPR) Pearson r improvements at `expanded/mosa_all`.

Earlier non-CNV selected version (chronologically older): MOSA `20260313_162348`. Selected TabPFN `expanded/mosa_all` r: drug response `0.3231`, CRISPR-Cas12 `0.2139`. Artifacts under `reports/{vae/files,tabpfn,random_forest,model_comparison}/.../20260313_162348/`. That upstream config used four views (no `copynumber`).

## Other Experiment Branches

These were run or documented, but they are not the current selected version.

| Branch | Location | Status | Notes |
| --- | --- | --- | --- |
| Standard TabPFN + union CNV MOSA feature augmentation | `reports/tabpfn_cnv_mosa_only_union/feature_augmentation/20260511_174623/` | **Current selected downstream model** | Selected setting is `expanded/mosa_all`. Union variant (`min_views_per_sample=1`). |
| Random forest + union CNV MOSA feature augmentation | `reports/random_forest_cnv_mosa_only_union/feature_augmentation/20260511_174623/` | **Current comparator** | Used for RF baseline and RF expanded/mosa_all bars in the selected version. |
| Cross-model + union CNV feature comparison | `reports/model_comparison_cnv_mosa_only_union/feature_augmentation/20260511_174623/` | **Current selected comparison artifact** | Contains the current selected summary and plot. |
| Standard TabPFN + union CNV MOSA feature augmentation, per-target corr FS | `reports/tabpfn_cnv_mosa_only_union_ftsel/feature_augmentation/20260511_174623/` | Not selected (2026-05-17 follow-up) | CCMA-paper-style `vc=0.01` + top-N=500 per-target Pearson r FS. Selected combo (`expanded/mosa_all`) TabPFN r: drug 0.3498 (Δ +0.0018 vs no-FS baseline), CRISPR 0.2170 (Δ −0.0025). Flat for TabPFN on the selected combo. |
| Random forest + union CNV MOSA feature augmentation, per-target corr FS | `reports/random_forest_cnv_mosa_only_union_ftsel/feature_augmentation/20260511_174623/` | Not selected (2026-05-17 follow-up) | RF gains from per-target FS more than TabPFN. Selected combo RF r: drug 0.3171 (Δ +0.0315), CRISPR 0.2170 (Δ +0.0197). On `expanded/mosa_all` RF is now essentially tied with TabPFN on CRISPR. |
| Cross-model + union CNV feature comparison, per-target corr FS | `reports/model_comparison_cnv_mosa_only_union_ftsel/feature_augmentation/20260511_174623/` | Not selected (2026-05-17 follow-up) | Headline summary CSV + aggregate plot for the per-target FS branch. |
| Standard TabPFN + union CNV MOSA feature augmentation, omics-only + CNV + clinical | `reports/tabpfn_cnv_mosa_only_union_omicsclin/feature_augmentation/20260511_174623/` | Not selected (2026-05-18 follow-up) | New predictor design: drops cross-phenotype (drug↔CRISPR), adds copy number and clinical metadata (age, sex, tumor-class dummies). Selected combo (`expanded/mosa_all`) TabPFN r: drug 0.3473 (Δ −0.0008 vs cross-phenotype baseline 0.3481), CRISPR 0.2126 (Δ −0.0070 vs 0.2195). |
| Random forest + union CNV MOSA feature augmentation, omics-only + CNV + clinical | `reports/random_forest_cnv_mosa_only_union_omicsclin/feature_augmentation/20260511_174623/` | Not selected (2026-05-18 follow-up) | RF selected combo (`expanded/mosa_all`) r: drug 0.2813 (Δ −0.0043 vs 0.2856), CRISPR 0.1908 (Δ −0.0065 vs 0.1973). |
| Cross-model + union CNV feature comparison, omics-only + CNV + clinical | `reports/model_comparison_cnv_mosa_only_union_omicsclin/feature_augmentation/20260511_174623/` | Not selected (2026-05-18 follow-up) | Headline summary CSV + aggregate plot for the omicsclin branch. |
| Figure set, omics-only + CNV + clinical | `reports/cba2026_claude_union_omicsclin/{shap_analysis,prediction_analysis}/20260511_174623/` | Companion to the omicsclin branch | Regenerated via `notebooks/cba2026_claude_union_omicsclin/0?_*.ipynb` plus direct calls to `_pathway_enrichment.run_all()` and `figP12_model_consistency`. Mirrors baseline `reports/cba2026_claude_union/` parity. `fig1_setup_illustration.{png,svg}` not regenerated (hand-drawn). |
| Standard TabPFN + ≥2 CNV MOSA feature augmentation | `reports/tabpfn_cnv_mosa_only/feature_augmentation/20260505_131645/` | Previous selected downstream model (superseded 2026-05-11) | Same upstream views as union; `min_views_per_sample=2`. |
| Random forest + ≥2 CNV MOSA feature augmentation | `reports/random_forest_cnv_mosa_only/feature_augmentation/20260505_131645/` | Previous comparator | RF rows for the ≥2 version. |
| Cross-model + ≥2 CNV feature comparison | `reports/model_comparison_cnv_mosa_only/feature_augmentation/20260505_131645/` | Previous selected comparison artifact | Containing the headline values for the ≥2 variant. |
| Standard TabPFN non-CNV feature augmentation | `reports/tabpfn/feature_augmentation/20260313_162348/` | Older non-CNV selected (pre-CNV era) | Values match the original final figures. |
| Random forest non-CNV feature augmentation | `reports/random_forest/feature_augmentation/20260313_162348/` | Older non-CNV comparator | Used for the pre-CNV RF baseline and RF expanded/mosa_all bars. |
| Standard TabPFN pseudo-label augmentation | `reports/tabpfn/pseudolabel_augmentation/20260313_162348/` | Not selected | Pseudo-label aggregate values do not match the selected final setting. |
| Random forest pseudo-label augmentation | `reports/random_forest/pseudolabel_augmentation/20260313_162348/` | Not selected | Used for pseudo-label comparison only. |
| Finetuned TabPFN default checkpoint | `reports/tabpfn_finetune/default_ckpt/` | Not selected | Close to the prior standard TabPFN values on the non-CNV run, but not selected. |

## Reproduction Command References

Command runbooks:

- CNV runbook: `docs/ccma_runs/cnv_added_mosa_experiment.md`
- Union variant (≥1 omic) runbook: `docs/ccma_runs/union_variant_experiment.md`
- Per-target correlation FS runbook (CCMA-paper style): `docs/ccma_runs/feature_selection_experiment.md`
- Omics-only + CNV + clinical runbook: `docs/ccma_runs/omics_clinical_experiment.md`
- TabPFN runs: `docs/ccma_runs/tabpfn_experiment_commands.md`
- Random forest and model comparison: `docs/ccma_runs/random_forest_model_comparison_commands.md`

Key environment for the current selected run:

```bash
PY=/home/scai/anaconda3/envs/mosa/bin/python
TS=20260511_174623
REPORTS_STD=reports/tabpfn_cnv_mosa_only_union
RF_REPORTS=reports/random_forest_cnv_mosa_only_union
MODEL_COMPARE_REPORTS=reports/model_comparison_cnv_mosa_only_union
```
