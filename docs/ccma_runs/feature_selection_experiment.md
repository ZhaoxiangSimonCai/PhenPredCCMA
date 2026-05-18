# CCMA Per-Target Feature Selection Experiment

Date added: 2026-05-17

Status: **completed, not promoted**. Per-target FS gives TabPFN essentially flat results on the selected combo (`expanded/mosa_all`); the original union pipeline remains the **Current Selected Version** in `docs/ccma_experiment_tracker.md`. See "Decision" below.

Purpose: re-run TabPFN and Random Forest on top of the union MOSA timestamp `20260511_174623` with the per-target feature-selection method from the CCMA paper (cdsr_models, `random_forest.R`), and compare against the current selected union variant (per-block variance prefilter with per-modality quotas).

The selection method (faithful to `cdsrmodels::random_forest`):

1. Variance prefilter per block: drop columns with NaN-aware variance ≤ `vc` (paper default `vc = 0.01`).
2. Inside the per-target loop, on the rows with a finite training label for target `j`: rank surviving features by `|Pearson r(X[:, k], y_j)|`, take the top `n` (paper default `n = 500`).
3. Fit and predict using only those top-`n` features.

No per-modality quotas; features compete in one global pool. This is on top of the union variant (`min_views_per_sample = 1`, MOSA `20260511_174623`).

## Experiment ID

```bash
EXPERIMENT=ccma_union_per_target_corr_n500
```

## 0) Preconditions

```bash
cd /mnt/scratch/scai/PhenPredCCMA
PY=/home/scai/anaconda3/envs/mosa/bin/python
TS=20260511_174623
MODEL_DEFAULT=/home/scai/scratch/PredCRISPRCCMA/tabpfn/models/tabpfn-v2.5-regressor-v2.5_default.ckpt
mkdir -p docs/ccma_runs/logs
```

Sanity check before launching:

```bash
test -f reports/vae/files/${TS}_imputed_crisprcas9_test_all.csv.gz
test -f reports/vae/files/${TS}_imputed_drugresponse_test_all.csv.gz
test -f data/clines/ccma_processed/crisprcas9_ccma_mosa_train.csv
```

## 1) Report roots

```bash
REPORTS_STD=reports/tabpfn_cnv_mosa_only_union_ftsel
RF_REPORTS=reports/random_forest_cnv_mosa_only_union_ftsel
MODEL_COMPARE_REPORTS=reports/model_comparison_cnv_mosa_only_union_ftsel
```

These mirror the current selected union roots with an `_ftsel` suffix so the existing union outputs stay untouched.

## 2) TabPFN sweep

```bash
for FAMILY in crisprcas9 drugresponse; do
  LOG="docs/ccma_runs/logs/ftsel_tabpfn_${FAMILY}_$(date +%Y%m%d_%H%M%S).log"
  set -o pipefail
  "$PY" tabpfn/run_feature_augmentation.py \
    --target-family "$FAMILY" \
    --sample-frame both \
    --variant all \
    --mosa-timestamp "$TS" \
    --ccma-dir data/clines/ccma_processed \
    --mosa-files-dir reports/vae/files \
    --out-dir "$REPORTS_STD/feature_augmentation" \
    --feature-selection-mode per_target_corr \
    --corr-top-n 500 \
    --variance-cutoff 0.01 \
    --tabpfn-estimator-mode standard \
    --tabpfn-n-estimators 8 \
    --tabpfn-fit-mode fit_preprocessors \
    --tabpfn-model-path "$MODEL_DEFAULT" \
    --device cuda \
    --gpu-id 0 \
    --log-every-targets 100 \
    2>&1 | tee "$LOG"
  echo "exit=$? log=$LOG"
done
```

Per `(family, frame, variant)` combo TabPFN writes:

- `target_fit_diagnostics.csv`: adds `n_features_used` column (≈500 for fit-status targets).
- `selected_features_per_target.csv.gz`: long-format `(target, rank, block, feature)` audit log.
- `variance_filter_summary.json`: per-block counts before and after the variance cutoff, plus `corr_top_n`.
- `test_metrics_summary.json`, `test_metrics_per_target.csv`, `test_predictions_wide.csv.gz`, `test_truth_wide.csv.gz`, `test_prediction_records.csv.gz`, `selected_features.json`, `config_used.json`, `split_indices.npz`.

## 3) Random forest sweep

```bash
for FAMILY in crisprcas9 drugresponse; do
  LOG="docs/ccma_runs/logs/ftsel_rf_${FAMILY}_$(date +%Y%m%d_%H%M%S).log"
  set -o pipefail
  "$PY" random_forest/run_feature_augmentation.py \
    --target-family "$FAMILY" \
    --sample-frame both \
    --variant all \
    --mosa-timestamp "$TS" \
    --ccma-dir data/clines/ccma_processed \
    --mosa-files-dir reports/vae/files \
    --out-dir "$RF_REPORTS/feature_augmentation" \
    --feature-selection-mode per_target_corr \
    --corr-top-n 500 \
    --variance-cutoff 0.01 \
    --rf-n-estimators 300 \
    --rf-max-features sqrt \
    --rf-n-jobs -1 \
    --log-every-targets 100 \
    2>&1 | tee "$LOG"
  echo "exit=$? log=$LOG"
done
```

## 4) Cross-model comparison

```bash
"$PY" tabpfn/plot_experiment_comparison.py \
  --reports-root "$REPORTS_STD" \
  --experiment-name feature_augmentation \
  --mosa-timestamp "$TS"

"$PY" tabpfn/plot_experiment_comparison.py \
  --reports-root "$RF_REPORTS" \
  --experiment-name feature_augmentation \
  --mosa-timestamp "$TS"

"$PY" model_comparison/plot_model_comparison.py \
  --experiment-name feature_augmentation \
  --mosa-timestamp "$TS" \
  --tabpfn-root "$REPORTS_STD" \
  --random-forest-root "$RF_REPORTS" \
  --out-dir "$MODEL_COMPARE_REPORTS"
```

Headline readout:

```bash
"$PY" - <<'PY'
import os, pandas as pd
ts = os.environ.get("TS", "20260511_174623")
path = f"reports/model_comparison_cnv_mosa_only_union_ftsel/feature_augmentation/{ts}/summary_model_comparison.csv"
df = pd.read_csv(path)
sel = df[
    (df["model_name_tabpfn"] == "tabpfn")
    & (df["sample_frame"] == "expanded")
    & (df["variant"] == "mosa_all")
]
print(sel[["target_family", "test_pearsonr_tabpfn", "test_pearsonr_random_forest", "delta_test_pearsonr_tabpfn_minus_random_forest"]])
PY
```

Compare against the current selected union values: drug `0.3481`, CRISPR `0.2195` (from `reports/model_comparison_cnv_mosa_only_union/feature_augmentation/20260511_174623/summary_model_comparison.csv`).

## 5) Implementation notes

- New code lives in `tabpfn/feature_selection.py` (`fit_variance_only_selector`, `select_per_target_corr_indices`), `tabpfn/experiment_core.py` (per-target FS hook in `fit_predict_per_target`), and `random_forest/model_core.py` (mirror hook in `fit_predict_per_target_rf`).
- CLI flags `--feature-selection-mode`, `--corr-top-n`, `--variance-cutoff` are wired through both runners. Default mode is `block_variance`; the existing union pipeline is unchanged.
- Variance filter is computed on the raw, NaN-aware blocks (matches the existing `_nan_variance_score` semantics). The preprocessor (NaN-fill + per-block standardisation for continuous modalities) still runs on the full pre-selection feature space; only column subsetting differs.
- The variance cutoff `0.01` is the CCMA / cdsr_models default. In our concatenated multi-omic setting it leaves transcriptomics and methylation largely intact and may filter most low-frequency mutations. Per-target correlation does the rest of the work; see `variance_filter_summary.json` per combo.

## 6) Decision (2026-05-17)

| Target family / combo | Baseline (no FS) | Per-target corr FS | Δ |
| --- | ---: | ---: | ---: |
| TabPFN drug `expanded/mosa_all` | 0.3481 | 0.3498 | +0.0018 |
| TabPFN CRISPR `expanded/mosa_all` | 0.2195 | 0.2170 | −0.0025 |
| RF drug `expanded/mosa_all` | 0.2856 | 0.3171 | +0.0315 |
| RF CRISPR `expanded/mosa_all` | 0.1973 | 0.2170 | +0.0197 |

Outcome: TabPFN gains on the selected combo are within noise (Δ ≈ ±0.002); RF benefits more substantially but RF is not the selected downstream model. Original union pipeline (per-block variance prefilter with per-modality quotas, no per-target FS) stays as the **Current Selected Version**.

Artifacts under `reports/{tabpfn,random_forest,model_comparison}_cnv_mosa_only_union_ftsel/feature_augmentation/20260511_174623/` are kept for the record but no further downstream work (SHAP, notebook re-execution, figure regeneration) will be done against them.

## 7) What is NOT changed

- No edits to `data/clines/ccma_processed/`.
- No edits to MOSA / `PhenPred/vae/*.py`.
- No changes to existing `reports/tabpfn_cnv_mosa_only_union/...` or `reports/random_forest_cnv_mosa_only_union/...` outputs. All new artifacts go under the `_ftsel`-suffixed roots.
- `notebooks/cba2026_claude_union/` continues to point at the current selected (non-FS) union artifacts.
