# CCMA MOSA + Copy Number Experiment

Date added: 2026-05-05

Purpose: rerun the selected CCMA MOSA workflow with copy number data from `data/CCMA/cnv.csv`, then rerun the selected downstream `feature_augmentation` comparison against the new MOSA timestamp.

## Experiment ID

```bash
EXPERIMENT=ccma_scratch_cnv
```

## CNV Data Check

`data/CCMA/cnv.csv` needs preprocessing before MOSA can use it.

Observed structure:

- Long annotated CNV table, not a MOSA-ready matrix.
- 2,946,236 rows, 84 columns.
- 322 unique sample IDs in column `ID`.
- 42,416 unique `gene_name` values.
- 15,129 genes overlap the DepMap transcriptomics reference, but the default preprocessing is stricter: it keeps the 606 curated `cancer_driver=True` genes from `data/clines/mutations_summary_20230202.csv`.
- All 119 existing CCMA holdout split IDs are present: 95 train and 24 test.
- All rows have `is_significant_copy_no=TRUE` and `to_remove=FALSE`.
- `cnvkit__copy_no` is absolute copy number, so it is converted to the repo's discrete `copynumber` state.

The preprocessing now lives in `notebooks/preprocess_CCMA.ipynb` under **Copy number preprocessing**. It writes the filtered `copynumber_ccma.csv`, and the existing split/export cell now writes the matching train/test files.

Encoding used:


| `cnvkit__copy_no`  | MOSA `copynumber` state |
| ------------------ | ----------------------- |
| 0                  | -2                      |
| 1                  | -1                      |
| 2 or missing event | 0                       |
| 3                  | 1                       |
| >=4                | 2                       |


## 1) Preprocess CNV

Open and run `notebooks/preprocess_CCMA.ipynb` through the copy-number preprocessing and deterministic split/export cells.

Expected outputs:

```bash
ls -lh data/clines/ccma_processed/copynumber_ccma*.csv
cat data/clines/ccma_processed/copynumber_ccma_preprocess_summary.json
ls -lh reports/vae/ccma_multiomic_map.png
```

## 2) MOSA Config

The CNV config has already been created:

```bash
reports/vae/configs/hyperparameters_ccma_scratch_cnv.json
```

This config adds `copynumber` as a fifth MOSA view but leaves `filter_features` aligned with the previous seven-omic configs: `transcriptomics`, `crisprcas9`, and `methylation` only. The CNV feature filtering happens in the notebook before MOSA sees the matrix.

If `align_to_reference_features` is later toggled on for a transfer-style run, `reference_feature_views` already includes `copynumber`.

Sanity check:

```bash
grep -n "copynumber\|dataname\|view_loss" reports/vae/configs/hyperparameters_ccma_scratch_cnv.json
```

## 3) Run MOSA With CNV

Run from the repo root:

```bash
cd /home/scai/scratch/PhenPredCCMA
PY=/home/scai/anaconda3/envs/mosa/bin/python
mkdir -p docs/ccma_runs/logs

LOG="docs/ccma_runs/logs/run_ccma_scratch_cnv_$(date +%Y%m%d_%H%M%S).log"
set -o pipefail
"$PY" -m PhenPred.vae.Main \
  --hypers-json reports/vae/configs/hyperparameters_ccma_scratch_cnv.json \
  2>&1 | tee "$LOG"
echo "exit_code=$? log=$LOG"
```

After the run, set `TS` to the new MOSA timestamp printed in the log or present in `reports/vae/files/`.

```bash
export TS=<new_mosa_timestamp>
ls -lh reports/vae/files/${TS}_model.pt
ls -lh reports/vae/configs/history/${TS}_hyperparameters.json
ls -lh reports/vae/files/${TS}_imputed_copynumber*.csv.gz
```

## 4) Run The Selected Downstream Comparison

This keeps the selected downstream version unchanged: standard TabPFN `feature_augmentation`, `expanded/mosa_all` selected, with RF comparison. CNV is used by MOSA as an added view, and the downstream predictor blocks remain the same as the selected run.

```bash
REPORTS_STD=reports/tabpfn_cnv_mosa_only
RF_REPORTS=reports/random_forest_cnv_mosa_only
MODEL_COMPARE_REPORTS=reports/model_comparison_cnv_mosa_only
MODEL_DEFAULT=/home/scai/scratch/PredCRISPRCCMA/tabpfn/models/tabpfn-v2.5-regressor-v2.5_default.ckpt
```

TabPFN:

```bash
for FAMILY in crisprcas9 drugresponse; do
  "$PY" tabpfn/run_feature_augmentation.py \
    --target-family "$FAMILY" \
    --sample-frame both \
    --variant all \
    --mosa-timestamp "$TS" \
    --ccma-dir data/clines/ccma_processed \
    --mosa-files-dir reports/vae/files \
    --out-dir "$REPORTS_STD/feature_augmentation" \
    --max-features 2000 \
    --min-features-per-modality 100 \
    --tabpfn-estimator-mode standard \
    --tabpfn-n-estimators 8 \
    --tabpfn-fit-mode fit_preprocessors \
    --tabpfn-model-path "$MODEL_DEFAULT" \
    --device cuda \
    --gpu-id 0 \
    --log-every-targets 100
done
```

Random forest baseline:

```bash
test -n "${TS:-}" || { echo "Set TS to the CNV MOSA timestamp first"; exit 1; }

for FAMILY in crisprcas9 drugresponse; do
  "$PY" random_forest/run_feature_augmentation.py \
    --target-family "$FAMILY" \
    --sample-frame both \
    --variant all \
    --mosa-timestamp "$TS" \
    --ccma-dir data/clines/ccma_processed \
    --mosa-files-dir reports/vae/files \
    --out-dir "$RF_REPORTS/feature_augmentation" \
    --max-features 2000 \
    --min-features-per-modality 100 \
    --rf-n-estimators 300 \
    --rf-max-features sqrt \
    --rf-n-jobs -1 \
    --log-every-targets 100
done
```

Aggregate and compare:

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

Note: `tabpfn/plot_experiment_comparison.py` is the shared aggregation plotter for both TabPFN and random forest report layouts. If it errors with `No test_metrics_summary.json files found`, the corresponding training loop has not produced metrics for that exact `TS`; rerun that loop with the same CNV MOSA timestamp.

Expected final comparison artifact:

```bash
reports/model_comparison_cnv_mosa_only/feature_augmentation/${TS}/summary_model_comparison.csv
reports/model_comparison_cnv_mosa_only/feature_augmentation/${TS}/aggregate_model_comparison.png
```

## 5) Tracking Results

The +CNV run is now selected as the current project version. The selected downstream setting remains standard TabPFN `feature_augmentation`, `expanded/mosa_all`.


| Date | MOSA timestamp | Downstream output root | Drug Response selected r | CRISPR selected r | Notes |
| --- | --- | --- | ---: | ---: | --- |
| 2026-05-05 | `20260505_131645` | `reports/model_comparison_cnv_mosa_only/feature_augmentation/20260505_131645/` | 0.3194 | 0.2124 | CNV added to MOSA as fifth view; downstream predictor blocks unchanged. Selected as current version. |

Quick comparison against the previous non-CNV selected run `20260313_162348`:

| Target family | Previous selected r | +CNV selected r | Delta |
| --- | ---: | ---: | ---: |
| Drug Response | 0.3231 | 0.3194 | -0.0037 |
| CRISPR-Cas12 | 0.2139 | 0.2124 | -0.0015 |


Useful extraction command after comparison:

```bash
"$PY" - <<'PY'
import os
import pandas as pd

ts = os.environ["TS"]
path = f"reports/model_comparison_cnv_mosa_only/feature_augmentation/{ts}/summary_model_comparison.csv"
df = pd.read_csv(path)
sel = df[
    (df["model_name_tabpfn"] == "tabpfn")
    & (df["sample_frame"] == "expanded")
    & (df["variant"] == "mosa_all")
]
print(sel[["target_family", "test_pearsonr_tabpfn", "test_pearsonr_random_forest", "delta_test_pearsonr_tabpfn_minus_random_forest"]])
PY
```

## 6) Run SHAP For The Selected +CNV MOSA Run

The selected upstream MOSA timestamp for SHAP is the current tracker selection:

```bash
cd /home/scai/scratch/PhenPredCCMA
PY=/home/scai/anaconda3/envs/mosa/bin/python
TS=20260505_131645
mkdir -p docs/ccma_runs/logs
```

Sanity checks before launching SHAP:

```bash
ls -lh reports/vae/files/${TS}_model.pt
ls -lh reports/vae/configs/history/${TS}_hyperparameters.json
ls -lh reports/vae/files/${TS}_imputed_copynumber.csv.gz
```

Current status before this stage: no `${TS}_shap_*` outputs were present under `reports/vae/files/`, so SHAP still needs to be generated for the selected +CNV run.

CRISPR-Cas12 SHAP:

```bash
LOG="docs/ccma_runs/logs/run_ccma_shap_${TS}_crisprcas9_$(date +%Y%m%d_%H%M%S).log"
set -o pipefail
"$PY" -m PhenPred.vae.RunCCMAShap \
  --timestamp "$TS" \
  --explain-target crisprcas9 \
  --all-samples \
  --multi-gpu-shap \
  --n-samples 50 \
  --seed 42 \
  2>&1 | tee "$LOG"
echo "exit_code=$? log=$LOG"
```

Drug-response SHAP:

```bash
LOG="docs/ccma_runs/logs/run_ccma_shap_${TS}_drugresponse_$(date +%Y%m%d_%H%M%S).log"
set -o pipefail
"$PY" -m PhenPred.vae.RunCCMAShap \
  --timestamp "$TS" \
  --explain-target drugresponse \
  --all-samples \
  --multi-gpu-shap \
  --n-samples 50 \
  --seed 42 \
  2>&1 | tee "$LOG"
echo "exit_code=$? log=$LOG"
```

If memory becomes limiting, rerun the same target with explicit chunking or a smaller SHAP batch:

```bash
"$PY" -m PhenPred.vae.RunCCMAShap \
  --timestamp "$TS" \
  --explain-target crisprcas9 \
  --shap-batch-size 64 \
  --target-chunk-size 1 \
  --multi-gpu-shap \
  --n-samples 50 \
  --seed 42
```

The current runner writes aggregate mean-absolute SHAP outputs with a `_mean_abs` suffix. Check CRISPR-Cas12 outputs:

```bash
ls -lh reports/vae/files/${TS}_shap_values_crisprcas9_mean_abs.csv.gz
ls -lh reports/vae/files/${TS}_shap_feature_ranking_crisprcas9_mean_abs.csv
ls -lh reports/vae/files/${TS}_shap_omic_ranking_crisprcas9_mean_abs.csv
ls -lh reports/vae/files/${TS}_shap_values_top_features_crisprcas9_mean_abs.feather
ls -lh reports/vae/${TS}_shap_omic_ranking_crisprcas9_mean_abs.png
ls -lh reports/vae/${TS}_shap_omic_ranking_crisprcas9_mean_abs.pdf
```

Check drug-response outputs:

```bash
ls -lh reports/vae/files/${TS}_shap_values_drugresponse_mean_abs.csv.gz
ls -lh reports/vae/files/${TS}_shap_feature_ranking_drugresponse_mean_abs.csv
ls -lh reports/vae/files/${TS}_shap_omic_ranking_drugresponse_mean_abs.csv
ls -lh reports/vae/files/${TS}_shap_values_top_features_drugresponse_mean_abs.feather
ls -lh reports/vae/${TS}_shap_omic_ranking_drugresponse_mean_abs.png
ls -lh reports/vae/${TS}_shap_omic_ranking_drugresponse_mean_abs.pdf
```

After the CRISPR-Cas12 outputs exist, open the existing notebook and set the timestamp to the selected +CNV run:

```bash
jupyter lab notebooks/shap_analysis_ccma_crispr.ipynb
```

In the notebook:

```python
TIMESTAMP = "20260505_131645"
```

## Optional: Direct CNV As A Downstream Predictor

The commands above test whether adding CNV to MOSA improves the selected downstream workflow while keeping the downstream predictor set unchanged.

If the next question is whether the TabPFN/RF downstream models should directly consume CNV as an additional predictor block, update `tabpfn/experiment_core.py` before running step 4:

- Add `"copynumber": "copynumber"` to `RAW_VIEW_FILE_STEMS`.
- Add `"copynumber"` to `RAW_ROWS_ARE_SAMPLES`.
- Add `"copynumber"` to both entries in `PREDICTOR_BLOCKS`.
- Keep `copynumber` out of `CONTINUOUS_BLOCKS` so it is not standardized.

Use a separate output root, for example `reports/tabpfn_cnv_direct`, so MOSA-only-CNV results and direct-CNV-predictor results stay distinct.
