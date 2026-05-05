# CCMA Worklog

Last updated: 2026-05-05

This worklog complements [ccma_experiment_tracker.md](ccma_experiment_tracker.md). The tracker records the currently selected setting and headline results; this file records what has been done in time order and what remains pending.

Date evidence is based mainly on artifact timestamps under `reports/`, supported by runbooks under `docs/ccma_runs/`. Treat these dates as filesystem artifact times, not necessarily exact job start times.

## Completed Work

| Date | Work | Status and evidence |
| --- | --- | --- |
| 2026-02-25 to 2026-02-26 | Earlier CCMA MOSA transfer/scratch experiments were documented. | `docs/ccma_runs/checkpoint_journal.md` records transfer and scratch runs for `20260225_133435`, `20260225_223235`, `20260226_024535`, and `20260226_031500`. The current `reports/vae/files/` directory does not contain those older model artifacts, so they are treated as historical/documented work rather than current reproducible outputs in this checkout. |
| 2026-03-13 | Non-CNV MOSA scratch run was generated. | MOSA timestamp `20260313_162348` produced upstream artifacts under `reports/vae/files/`, internal benchmark outputs under `reports/vae/internal/`, loss plots under `reports/vae/losses/`, and saved config `reports/vae/configs/history/20260313_162348_hyperparameters.json`. This became the previous selected upstream run before the CNV rerun. |
| 2026-03-14 | Standard TabPFN downstream experiments were run for the non-CNV MOSA timestamp. | Standard TabPFN feature-augmentation and pseudo-label experiments completed for `crisprcas9` and `drugresponse` under `reports/tabpfn/feature_augmentation/20260313_162348/` and `reports/tabpfn/pseudolabel_augmentation/20260313_162348/`. |
| 2026-03-15 | Random-forest baselines and cross-model comparisons were generated for the non-CNV MOSA timestamp. | Random-forest outputs were written under `reports/random_forest/`, and cross-model summaries/plots under `reports/model_comparison/`. The previous selected result was standard TabPFN `feature_augmentation`, `expanded/mosa_all`, with macro mean Pearson r `0.3231` for drug response and `0.2139` for CRISPR-Cas9. |
| 2026-03-16 to 2026-03-19 | Finetuned TabPFN default-checkpoint branch was run. | Outputs under `reports/tabpfn_finetune/default_ckpt/` include feature-augmentation and pseudo-label comparisons for timestamp `20260313_162348`. These results were close to the previous selected standard TabPFN values but were not selected. |
| 2026-05-05 | Experiment tracking and CNV rerun planning were added. | `docs/ccma_experiment_tracker.md` records the selected setting and CNV status. `docs/ccma_runs/cnv_added_mosa_experiment.md` documents the workflow for adding copy-number data. |
| 2026-05-05 | CNV preprocessing was completed. | `data/clines/ccma_processed/copynumber_ccma*.csv` files were generated. `data/clines/ccma_processed/copynumber_ccma_preprocess_summary.json` reports `322` samples, `606` cancer-driver genes after filtering, and the conversion from absolute copy number to MOSA states: `0 -> -2`, `1 -> -1`, `2 or missing event -> 0`, `3 -> 1`, and `>=4 -> 2`. |
| 2026-05-05 | CNV MOSA scratch run completed. | MOSA timestamp `20260505_131645` completed with `copynumber` as a fifth input view. Outputs include `reports/vae/files/20260505_131645_model.pt`, `reports/vae/files/20260505_131645_imputed_copynumber*.csv.gz`, internal benchmark files under `reports/vae/internal/`, loss plots under `reports/vae/losses/`, and saved config `reports/vae/configs/history/20260505_131645_hyperparameters.json`. |
| 2026-05-05 | Downstream CNV TabPFN, random forest, and cross-model comparison completed. | TabPFN outputs completed under `reports/tabpfn_cnv_mosa_only/feature_augmentation/20260505_131645/`; random-forest outputs completed under `reports/random_forest_cnv_mosa_only/feature_augmentation/20260505_131645/`; final comparison artifacts completed under `reports/model_comparison_cnv_mosa_only/feature_augmentation/20260505_131645/`. |
| 2026-05-05 | +CNV version was selected as the current version. | The current selected setting is standard TabPFN `feature_augmentation`, `expanded/mosa_all`, using MOSA timestamp `20260505_131645`. Macro mean Pearson r is `0.3194` for drug response and `0.2124` for CRISPR-Cas9. This supersedes the previous non-CNV selected run `20260313_162348`. |

## Pending Next Steps

| Priority | Next step | Current status | Completion criteria |
| --- | --- | --- | --- |
| 1 | Complete SHAP calculation on drug and CRISPR for the current selected +CNV run. | The repo has the SHAP runner (`PhenPred/vae/RunCCMAShap.py`) and SHAP notebooks, but no completed SHAP result files are present under `reports/` beyond `reports/vae/files/shap_target_genes.txt`. | Generate SHAP value tables, feature rankings, omic rankings, and plots for `crisprcas9` and `drugresponse` using the selected CNV MOSA timestamp `20260505_131645`. |
| 2 | Analyze improved and degraded predictions for the selected +CNV run. | The selected CNV comparison is complete under `reports/model_comparison_cnv_mosa_only/feature_augmentation/20260505_131645/`. A quick macro comparison to the non-CNV run shows +CNV is slightly lower on the selected TabPFN Pearson r values, but it is now the selected version. | Summarize per-target deltas, improved-target fractions, top improved/degraded targets, and drug-vs-CRISPR patterns for the selected +CNV run. |
| 3 | Analyze SHAP values. | SHAP outputs are pending, so interpretation cannot be completed yet. | Summarize feature-level and omic-layer SHAP rankings, target-level attribution patterns, and links between SHAP patterns and improved or degraded predictions. |

## Useful References

- Current selected setting: `docs/ccma_experiment_tracker.md`
- CNV experiment runbook: `docs/ccma_runs/cnv_added_mosa_experiment.md`
- Standard TabPFN commands: `docs/ccma_runs/tabpfn_experiment_commands.md`
- Random-forest and model-comparison commands: `docs/ccma_runs/random_forest_model_comparison_commands.md`
- SHAP runbook: `docs/ccma_runs/ccma_analysis_commands.md`
