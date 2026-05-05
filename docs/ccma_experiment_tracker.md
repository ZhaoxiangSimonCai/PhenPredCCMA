# CCMA Experiment Tracker

Last updated: 2026-05-05

## Current Selected Version

Use the copy-number-augmented `feature_augmentation` downstream experiment from MOSA timestamp `20260505_131645`.

Selected final model/setting:

- Model: standard TabPFN, not the finetuned TabPFN branch
- Downstream experiment: `feature_augmentation`
- Selected sample frame and variant: `expanded` / `mosa_all`
- Upstream MOSA run: `20260505_131645`, `ccma_scratch_cnv`
- Added upstream view: `copynumber`, preprocessed from `data/CCMA/cnv.csv`
- Downstream predictor blocks: unchanged from the selected non-CNV workflow; CNV is used through MOSA imputation, not as a direct downstream predictor block
- Baseline used in the final comparison: random forest `overlap` / `original`
- Metric reported in the final figure: per-target macro mean Pearson r
- Target panel: 500 drug-response targets and 500 CRISPR-Cas9 targets
- Selected feature count: 2000
- Seed: 42

Primary artifacts:

- Upstream MOSA config: `reports/vae/configs/history/20260505_131645_hyperparameters.json`
- Upstream MOSA outputs: `reports/vae/files/20260505_131645_*`
- CNV preprocessing summary: `data/clines/ccma_processed/copynumber_ccma_preprocess_summary.json`
- Selected TabPFN outputs:
  - `reports/tabpfn_cnv_mosa_only/feature_augmentation/20260505_131645/drugresponse/expanded/mosa_all/`
  - `reports/tabpfn_cnv_mosa_only/feature_augmentation/20260505_131645/crisprcas9/expanded/mosa_all/`
- RF baseline outputs:
  - `reports/random_forest_cnv_mosa_only/feature_augmentation/20260505_131645/drugresponse/overlap/original/`
  - `reports/random_forest_cnv_mosa_only/feature_augmentation/20260505_131645/crisprcas9/overlap/original/`
- Cross-model summary for the selected CNV comparison: `reports/model_comparison_cnv_mosa_only/feature_augmentation/20260505_131645/summary_model_comparison.csv`
- Cross-model per-target data: `reports/model_comparison_cnv_mosa_only/feature_augmentation/20260505_131645/combined_per_target.csv`
- Aggregate comparison plot: `reports/model_comparison_cnv_mosa_only/feature_augmentation/20260505_131645/aggregate_model_comparison.png`

## Current Selected Results

These are the headline values for the selected +CNV version.

| Target family | RF overlap/original baseline | RF expanded/mosa_all | TabPFN overlap/original | Selected TabPFN expanded/mosa_all | Selected gain vs RF baseline | Per-target improvement vs RF baseline | Previous selected TabPFN r | Delta vs previous |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Drug Response | 0.2346 | 0.2626 | 0.2207 | 0.3194 | +0.0849 (+36.2%) | 70.0% improved; median delta r = 0.0985 | 0.3231 | -0.0037 |
| CRISPR-Cas9 | 0.1510 | 0.1819 | 0.1436 | 0.2124 | +0.0613 (+40.6%) | 64.0% improved; median delta r = 0.0668 | 0.2139 | -0.0015 |

Selected TabPFN run details:

| Target family | Train n | Test n | Targets | Test observations | Test Pearson r | Test R2 | Test RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Drug Response | 232 | 24 | 500 | 11951 | 0.3194 | 0.0344 | 0.1688 |
| CRISPR-Cas9 | 132 | 24 | 500 | 11932 | 0.2124 | -0.0019 | 0.6490 |

## Copy Number / CNV Status

Copy number is used in the current selected upstream MOSA run.

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
- CRISPR-Cas9 predictors: `transcriptomics`, `methylation`, `drugresponse`, `mutations`

So CNV enters the selected workflow through MOSA's learned/imputed views, not as a direct TabPFN/RF raw predictor.

## Previous Selected Version

The previous selected version was the non-CNV `feature_augmentation` downstream experiment from MOSA timestamp `20260313_162348`.

Key artifacts:

- Upstream MOSA config: `reports/vae/configs/history/20260313_162348_hyperparameters.json`
- Upstream MOSA outputs: `reports/vae/files/20260313_162348_*`
- Cross-model summary: `reports/model_comparison/feature_augmentation/20260313_162348/summary_model_comparison.csv`
- Aggregate comparison plot: `reports/model_comparison/feature_augmentation/20260313_162348/aggregate_model_comparison.png`

Previous selected values:

| Target family | Selected TabPFN expanded/mosa_all r | RF overlap/original baseline | Notes |
| --- | ---: | ---: | --- |
| Drug Response | 0.3231 | 0.2346 | Matched the original final figures before the CNV rerun. |
| CRISPR-Cas9 | 0.2139 | 0.1510 | Matched the original final figures before the CNV rerun. |

That upstream MOSA config used `crisprcas9`, `drugresponse`, `transcriptomics`, and `methylation`; it used `mutations` as labels/conditionals and did not include `copynumber`.

## Other Experiment Branches

These were run or documented, but they are not the current selected version.

| Branch | Location | Status | Notes |
| --- | --- | --- | --- |
| Standard TabPFN + CNV MOSA feature augmentation | `reports/tabpfn_cnv_mosa_only/feature_augmentation/20260505_131645/` | Current selected downstream model | Selected setting is `expanded/mosa_all`. |
| Random forest + CNV MOSA feature augmentation | `reports/random_forest_cnv_mosa_only/feature_augmentation/20260505_131645/` | Current comparator | Used for RF baseline and RF expanded/mosa_all bars. |
| Cross-model + CNV feature comparison | `reports/model_comparison_cnv_mosa_only/feature_augmentation/20260505_131645/` | Current selected comparison artifact | Contains the current selected summary and plot. |
| Standard TabPFN non-CNV feature augmentation | `reports/tabpfn/feature_augmentation/20260313_162348/` | Previous selected version | Values match the original final figures. |
| Random forest non-CNV feature augmentation | `reports/random_forest/feature_augmentation/20260313_162348/` | Previous comparator | Used for the previous RF baseline and RF expanded/mosa_all bars. |
| Standard TabPFN pseudo-label augmentation | `reports/tabpfn/pseudolabel_augmentation/20260313_162348/` | Not selected | Pseudo-label aggregate values do not match the selected final setting. |
| Random forest pseudo-label augmentation | `reports/random_forest/pseudolabel_augmentation/20260313_162348/` | Not selected | Used for pseudo-label comparison only. |
| Finetuned TabPFN default checkpoint | `reports/tabpfn_finetune/default_ckpt/` | Not selected | Close to the previous selected standard TabPFN values, but not selected. |

## Reproduction Command References

Command runbooks:

- CNV runbook: `docs/ccma_runs/cnv_added_mosa_experiment.md`
- TabPFN runs: `docs/ccma_runs/tabpfn_experiment_commands.md`
- Random forest and model comparison: `docs/ccma_runs/random_forest_model_comparison_commands.md`

Key environment for the current selected run:

```bash
PY=/home/scai/anaconda3/envs/mosa/bin/python
TS=20260505_131645
REPORTS_STD=reports/tabpfn_cnv_mosa_only
RF_REPORTS=reports/random_forest_cnv_mosa_only
MODEL_COMPARE_REPORTS=reports/model_comparison_cnv_mosa_only
```
