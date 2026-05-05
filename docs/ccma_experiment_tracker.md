# CCMA Experiment Tracker

Last updated: 2026-05-05

## Current selected version

Use the `feature_augmentation` downstream experiment from MOSA timestamp `20260313_162348`.

Selected final model/setting:

- Model: standard TabPFN, not the finetuned TabPFN branch
- Downstream experiment: `feature_augmentation`
- Selected sample frame and variant: `expanded` / `mosa_all`
- Upstream MOSA run: `20260313_162348`, `ccma_scratch`
- Baseline used in the final comparison: random forest `overlap` / `original`
- Metric reported in the final figure: per-target macro mean Pearson r
- Target panel: 500 drug-response targets and 500 CRISPR-Cas9 targets
- Selected feature count: 2000
- Seed: 42

Primary artifacts:

- Upstream MOSA config: `reports/vae/configs/history/20260313_162348_hyperparameters.json`
- Upstream MOSA outputs: `reports/vae/files/20260313_162348_*`
- Selected TabPFN outputs:
  - `reports/tabpfn/feature_augmentation/20260313_162348/drugresponse/expanded/mosa_all/`
  - `reports/tabpfn/feature_augmentation/20260313_162348/crisprcas9/expanded/mosa_all/`
- RF baseline outputs:
  - `reports/random_forest/feature_augmentation/20260313_162348/drugresponse/overlap/original/`
  - `reports/random_forest/feature_augmentation/20260313_162348/crisprcas9/overlap/original/`
- Cross-model summary matching the final figure: `reports/model_comparison/feature_augmentation/20260313_162348/summary_model_comparison.csv`
- Cross-model per-target data: `reports/model_comparison/feature_augmentation/20260313_162348/combined_per_target.csv`
- Aggregate comparison plot: `reports/model_comparison/feature_augmentation/20260313_162348/aggregate_model_comparison.png`

## Evidence from the final results

These values match the attached final result figures after rounding.

| Target family | RF overlap/original baseline | RF expanded/mosa_all | TabPFN overlap/original | Selected TabPFN expanded/mosa_all | Selected gain vs RF baseline | Per-target improvement vs RF baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Drug Response | 0.2346 | 0.2690 | 0.2207 | 0.3231 | +0.0886 (+37.8%) | 70.0% improved; median delta r = 0.0905 |
| CRISPR-Cas9 | 0.1510 | 0.1962 | 0.1436 | 0.2139 | +0.0629 (+41.6%) | 63.8% improved; median delta r = 0.0558 |

Selected TabPFN run details:

| Target family | Train n | Test n | Targets | Test observations | Test Pearson r | Test R2 | Test RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Drug Response | 231 | 24 | 500 | 11951 | 0.3231 | 0.0283 | 0.1689 |
| CRISPR-Cas9 | 132 | 24 | 500 | 11932 | 0.2139 | -0.0027 | 0.6490 |

## Copy number / CNV status

Copy number data was not used in the selected previous round.

The selected upstream MOSA config `reports/vae/configs/history/20260313_162348_hyperparameters.json` contains these datasets only:

- `crisprcas9`
- `drugresponse`
- `transcriptomics`
- `methylation`

It uses `mutations` as labels/conditionals, but there is no `copynumber` or `cnv` dataset in either `datasets` or `holdout_datasets`.

The selected downstream feature-augmentation configs also exclude CNV:

- Drug Response predictors: `transcriptomics`, `methylation`, `crisprcas9`, `mutations`
- CRISPR-Cas9 predictors: `transcriptomics`, `methylation`, `drugresponse`, `mutations`

There are no selected `20260313_162348_imputed_copynumber*` files under `reports/vae/files/`.

## Other experiment branches

These were run or documented, but they are not the final selected version represented by the attached figures.

| Branch | Location | Status | Notes |
| --- | --- | --- | --- |
| Standard TabPFN feature augmentation | `reports/tabpfn/feature_augmentation/20260313_162348/` | Selected | Values match the final figures. |
| Random forest feature augmentation | `reports/random_forest/feature_augmentation/20260313_162348/` | Baseline/comparator | Used for RF baseline and RF expanded/mosa_all bars. |
| Cross-model feature comparison | `reports/model_comparison/feature_augmentation/20260313_162348/` | Selected comparison artifact | Contains the summary and plot matching the final figures. |
| Standard TabPFN pseudo-label augmentation | `reports/tabpfn/pseudolabel_augmentation/20260313_162348/` | Not selected | Pseudo-label aggregate values do not match the final figure. |
| Random forest pseudo-label augmentation | `reports/random_forest/pseudolabel_augmentation/20260313_162348/` | Not selected | Used for pseudo-label comparison only. |
| Finetuned TabPFN default checkpoint | `reports/tabpfn_finetune/default_ckpt/` | Not selected | Close to the final feature-augmentation values, but the attached final figure matches the standard TabPFN report under `reports/tabpfn/`. |

## Reproduction command references

Command runbooks:

- TabPFN runs: `docs/ccma_runs/tabpfn_experiment_commands.md`
- Random forest and model comparison: `docs/ccma_runs/random_forest_model_comparison_commands.md`

Key environment from the runbooks:

```bash
PY=/home/scai/anaconda3/envs/mosa/bin/python
TS=20260313_162348
REPORTS_STD=reports/tabpfn
RF_REPORTS=reports/random_forest
MODEL_COMPARE_REPORTS=reports/model_comparison
```
