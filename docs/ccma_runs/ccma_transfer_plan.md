# CCMA Transfer Learning Plan for MOSA (CRISPR + RNA + Mutation, Cross-Omic SHAP)

## Summary
Implement a CCMA-specific training path that loads a pretrained MOSA checkpoint (`.pt + hyperparameters.json`), fine-tunes on CCMA with available omics only (`crisprcas9`, `transcriptomics`, `mutations`), disables unavailable conditionals (`labels=[]`, `use_conditionals=false`), runs internal-only evaluation, and produces cross-omic SHAP rankings for a 100-gene CRISPR target panel. Methylation integration is a second phase with harmonization first.

## Locked Decisions
1. Transfer source: checkpoint + original hyperparameters JSON.
2. Initial omics: `CRISPR + RNA + mutation`.
3. Missing metadata strategy: disable conditionals (no zero-filled tissue/growth covariates).
4. Benchmark scope: internal-only.
5. SHAP scope: targeted CRISPR panel (100 genes), cross-omic mode (exclude CRISPR self-attribution).

## Implementation Spec

1. Add a CCMA dataset class and dataset factory
- Create `DatasetCCMA.py`.
- Behavior: load only configured `datasets`; transpose all except `crisprcas9`/`copynumber` exactly like current logic; apply CRISPR scaling; build `features_mask`; standardize; build `views`.
- Build `samples` from overlap across configured views with `min_views_per_sample` (default `2`), without DepMap SIDM mapping assumptions.
- Build placeholder `samplesheet` with `Unknown` values so downstream code is stable even without tissue/growth files.
- Build labels as a constant column when `labels=[]`.
- Keep view naming consistent (`mutations`, `transcriptomics`, `crisprcas9`) so SHAP and reporting reuse existing naming patterns.
- Add dataset factory selection in `Main.py`: `depmap23q2 | depmap24q4 | ccma`.

2. Make hyperparameter parsing robust for variable omics sets
- Update `Hypers.py`.
- Validate `len(view_loss_recon_type) == len(datasets)` and `len(view_loss_weights) == len(datasets)`.
- Fix default weights bug to use `len(datasets)` (not nonexistent `views` key).
- Add schema keys:
1. `dataset_class`
2. `benchmark_mode` (`full|internal|none`)
3. `transfer_checkpoint`
4. `transfer_hypers_json`
5. `transfer_mode` (`shared_views_partial`)
6. `transfer_freeze_epochs`
7. `min_views_per_sample`
8. `shap_target_view`
9. `shap_target_gene_count`
10. `shap_target_genes_file`
11. `shap_cross_omic_only`
- Keep backward compatibility: missing keys preserve current behavior.

3. Add partial shared-view transfer loading
- Extend `Train.py` with `load_pretrained_shared_views(...)` called from `initialize_model()`.
- Load checkpoint state dict and pretrained hyperparameters JSON.
- Map view-specific encoder/decoder weights by view name, not index position, using source/target dataset key order from both configs.
- Load only shape-compatible tensors for shared modules; reinitialize unmatched tensors (expected for changed joint input dimensionality and removed views).
- Support checkpoints with or without `module.` prefix.
- Emit a transfer report: loaded keys, skipped keys, loaded views.
- Optional freeze loaded view encoders/decoders for `transfer_freeze_epochs` (default `0`).

4. Refactor main runtime modes
- Update `Main.py`.
- Dataset instantiation via factory; stratified CV only when tissue labels are actually available.
- `benchmark_mode=none`: train/save only.
- `benchmark_mode=internal`: skip MOFA/MOVE/mixOmics/drug/proteomics/full CRISPR benchmarks; run internal benchmark module only.
- `benchmark_mode=full`: retain existing current path unchanged for DepMap workflows.

5. Add internal benchmark module (no external validation dependencies)
- Create `BenchmarkInternal.py`.
- Compute and save:
1. CV holdout observed-entry metrics per view: RMSE, MAE, Pearson, Spearman.
2. Per-view feature-level correlation summaries (median/IQR).
3. CRISPR skew preservation (original vs CV-predicted skew).
- Save outputs under `reports/vae/internal/`.
- Call from `Main.py` when `benchmark_mode=internal`.

6. Extend SHAP for targeted CRISPR and cross-omic attribution
- Update `Model.py` and `Train.py`.
- Add support for selecting output indices when `return_for_shap` targets a view, so SHAP can run on 100 CRISPR genes instead of all genes.
- Add target panel selector:
1. If `shap_target_genes_file` provided, use intersected genes.
2. Else auto-select top 100 CRISPR genes by essentiality prevalence and variance.
- Implement `shap_cross_omic_only=true` mode:
1. Hold CRISPR input view at baseline (feature mean) during SHAP input construction.
2. Exclude `crisprcas9_*` input features from final ranking tables.
- Add postprocessing outputs:
1. Feature-level SHAP ranking CSV.
2. Omic-layer SHAP ranking CSV.
3. Omic-layer bar plot for study reporting.

7. Add CCMA transfer config template
- Add `hyperparameters_ccma_transfer.json`.
- Defaults:
1. `dataset_class="ccma"`
2. `datasets={crisprcas9, transcriptomics, mutations}`
3. `labels=[]`
4. `use_conditionals=false`
5. `benchmark_mode="internal"`
6. `skip_cv=false`, `n_folds=3`
7. `transfer_mode="shared_views_partial"`
8. `shap_target_view="crisprcas9"`, `shap_target_gene_count=100`, `shap_cross_omic_only=true`

8. Phase 2 methylation integration (comparison run)
- Extend `preprocess_CCMA.ipynb` with a probe-to-gene harmonization workflow (EPIC probe annotation to gene-level aggregate matrix).
- Output `methylation_ccma_genelevel.csv` in `ccma_processed`.
- Run a second fine-tune with methylation added; compare against phase-1 using same internal metrics and SHAP omic-layer ranking.

## Public API and Interface Changes
1. New dataset class: `CLinesDatasetCCMA`.
2. New `Main.py` config keys listed above (`dataset_class`, `benchmark_mode`, transfer keys, SHAP keys).
3. New transfer method in `CLinesTrain`: partial shared-view checkpoint initialization.
4. SHAP APIs in `CLinesTrain` accept output target subset and cross-omic mode.
5. New benchmark entry point: `BenchmarkInternal`.

## Test Cases and Scenarios
1. Dataset load smoke test: instantiate CCMA dataset with 3 views and confirm non-zero sample count and expected view names.
2. Hyperparameter validation test: mismatched `view_loss_*` lengths fail fast with clear error.
3. Transfer mapping test: with synthetic source/target configs, verify only shared-view encoder/decoder keys are loaded.
4. End-to-end 1-epoch dry run: CCMA config runs training + prediction + model save without DepMap metadata files.
5. Internal benchmark test: outputs CSV/plots for all configured views.
6. SHAP test: CRISPR target panel size equals 100; omic ranking file excludes `crisprcas9_*` features in cross-omic mode.
7. Backward compatibility test: existing DepMap config with `benchmark_mode=full` preserves current outputs.

## Assumptions and Defaults
1. Pretrained checkpoint and matching hyperparameters JSON are available and readable.
2. Initial transfer run uses MOSA-compatible architecture settings (same model family as checkpoint).
3. No CCMA tissue/growth mapping file is available in phase 1.
4. At least two CCMA omic views are enabled per run.
5. Methylation is deferred until gene-level harmonization is produced.
