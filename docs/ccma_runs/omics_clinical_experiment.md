# CCMA Omics-Only + CNV + Clinical Experiment

Date added: 2026-05-17

Purpose: re-run TabPFN and Random Forest on top of MOSA `20260511_174623` with the **downstream X redefined**:

- Cross-phenotype predictors removed (drug-response no longer sees CRISPR essentiality, and vice versa).
- `copynumber` added as a direct downstream predictor block (previously only consumed by MOSA upstream).
- `clinical` (age, sex, tumor class one-hots) added as a direct downstream predictor block.

Plus the standing SHAP runs for `crisprcas9` and `drugresponse` targets against MOSA `20260511_174623`. MOSA itself is unchanged.

Selected union version (cross-phenotype, no CNV / clinical downstream) stays in `reports/{tabpfn,random_forest,model_comparison}_cnv_mosa_only_union/...`; this branch sits under the `_omicsclin` suffix.

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
test -f reports/vae/files/${TS}_imputed_copynumber_train_all.csv.gz
test -f data/clines/ccma_processed/clinical_ccma_mosa_train.csv
grep -A1 '"crisprcas9"' tabpfn/experiment_core.py | grep copynumber   # confirm new PREDICTOR_BLOCKS
```

## 1) Build clinical CSVs

```bash
"$PY" scripts/build_clinical_features.py \
  --meta-csv /home/scai/scratch/PhenPredCCMA/data/CCMA/CCMA_meta.csv \
  --ccma-dir data/clines/ccma_processed \
  --min-class-count 5
```

Writes:

```
data/clines/ccma_processed/clinical_ccma_overlap_train.csv
data/clines/ccma_processed/clinical_ccma_overlap_test.csv
data/clines/ccma_processed/clinical_ccma_mosa_train.csv
data/clines/ccma_processed/clinical_ccma_mosa_test.csv
data/clines/ccma_processed/clinical_ccma_preprocess_summary.json
```

Encoding (see `scripts/build_clinical_features.py`):

- `age_years` continuous; NaNs passed through (downstream preprocessor median-fills).
- `sex_is_male` binary (1=male, 0=female); `*_inferred` collapsed to base values; NaN where missing.
- `class__<name>` one-hot per tumor class with ≥5 training samples (currently 14 classes). All rarer labels collapse into `class__other`. Samples with NaN class get all-NaN class dummies (preprocessor fills with median = 0 → "no class signal").

Total clinical features written: 17 (`age_years`, `sex_is_male`, 14 kept-class dummies, `class__other`).

## 2) Report roots

```bash
REPORTS_STD=reports/tabpfn_cnv_mosa_only_union_omicsclin
RF_REPORTS=reports/random_forest_cnv_mosa_only_union_omicsclin
MODEL_COMPARE_REPORTS=reports/model_comparison_cnv_mosa_only_union_omicsclin
```

## 3) TabPFN sweep

```bash
for FAMILY in crisprcas9 drugresponse; do
  LOG="docs/ccma_runs/logs/omicsclin_tabpfn_${FAMILY}_$(date +%Y%m%d_%H%M%S).log"
  set -o pipefail
  "$PY" tabpfn/run_feature_augmentation.py \
    --target-family "$FAMILY" --sample-frame both --variant all \
    --mosa-timestamp "$TS" \
    --ccma-dir data/clines/ccma_processed --mosa-files-dir reports/vae/files \
    --out-dir "$REPORTS_STD/feature_augmentation" \
    --max-features 2000 --min-features-per-modality 100 \
    --tabpfn-estimator-mode standard --tabpfn-n-estimators 8 \
    --tabpfn-fit-mode fit_preprocessors \
    --tabpfn-model-path "$MODEL_DEFAULT" \
    --device cuda --gpu-id 0 --log-every-targets 100 \
    2>&1 | tee "$LOG"
done
```

## 4) Random forest sweep

```bash
for FAMILY in crisprcas9 drugresponse; do
  LOG="docs/ccma_runs/logs/omicsclin_rf_${FAMILY}_$(date +%Y%m%d_%H%M%S).log"
  set -o pipefail
  "$PY" random_forest/run_feature_augmentation.py \
    --target-family "$FAMILY" --sample-frame both --variant all \
    --mosa-timestamp "$TS" \
    --ccma-dir data/clines/ccma_processed --mosa-files-dir reports/vae/files \
    --out-dir "$RF_REPORTS/feature_augmentation" \
    --max-features 2000 --min-features-per-modality 100 \
    --rf-n-estimators 300 --rf-max-features sqrt --rf-n-jobs -1 \
    --log-every-targets 100 \
    2>&1 | tee "$LOG"
done
```

## 5) Cross-model comparison

```bash
"$PY" tabpfn/plot_experiment_comparison.py \
  --reports-root "$REPORTS_STD" --experiment-name feature_augmentation --mosa-timestamp "$TS"
"$PY" tabpfn/plot_experiment_comparison.py \
  --reports-root "$RF_REPORTS" --experiment-name feature_augmentation --mosa-timestamp "$TS"
"$PY" model_comparison/plot_model_comparison.py \
  --experiment-name feature_augmentation --mosa-timestamp "$TS" \
  --tabpfn-root "$REPORTS_STD" --random-forest-root "$RF_REPORTS" \
  --out-dir "$MODEL_COMPARE_REPORTS"
```

Headline readout:

```bash
"$PY" - <<'PY'
import os, pandas as pd
ts = os.environ.get("TS", "20260511_174623")
df = pd.read_csv(f"reports/model_comparison_cnv_mosa_only_union_omicsclin/feature_augmentation/{ts}/summary_model_comparison.csv")
sel = df[(df["model_name_tabpfn"]=="tabpfn") & (df["sample_frame"]=="expanded") & (df["variant"]=="mosa_all")]
print(sel[["target_family","test_pearsonr_tabpfn","test_pearsonr_random_forest","delta_test_pearsonr_tabpfn_minus_random_forest"]])
PY
```

Compare against the current selected union values: drug `0.3481`, CRISPR `0.2195`.

## 6) SHAP

SHAP runs against MOSA upstream `20260511_174623` and is independent of the new `PREDICTOR_BLOCKS`. Two target views:

```bash
"$PY" -m PhenPred.vae.RunCCMAShap --timestamp "$TS" --explain-target crisprcas9 --all-samples --n-samples 50 --seed 42
"$PY" -m PhenPred.vae.RunCCMAShap --timestamp "$TS" --explain-target drugresponse --all-samples --n-samples 50 --seed 42
```

Outputs under `reports/vae/files/`:

- `${TS}_shap_values_{target}.csv.gz`
- `${TS}_shap_feature_ranking_{target}.csv`
- `${TS}_shap_omic_ranking_{target}.csv`
- `${TS}_shap_values_top_features_{target}.feather`

SHAP universe is the MOSA training views (transcriptomics, methylation, drugresponse, crisprcas9, copynumber). Clinical is not part of MOSA, so it does not appear in SHAP outputs.

## 7) Implementation notes

- `tabpfn/experiment_core.py:39-55`: `PREDICTOR_BLOCKS` updated to omics-only-plus-CNV-plus-clinical for both target families; `RAW_VIEW_FILE_STEMS` gains `copynumber` and `clinical`; `RAW_ROWS_ARE_SAMPLES` gains `copynumber` and `clinical` (their CSVs are already samples-by-features); new `NON_MOSA_IMPUTED_VIEWS = {"mutations", "clinical"}` constant.
- `tabpfn/run_feature_augmentation.py` and `random_forest/run_feature_augmentation.py`: `load_predictor_view` routes any view in `NON_MOSA_IMPUTED_VIEWS` through the raw CSV path. The sample-intersection step now skips mutations *and* clinical (renamed `non_mutation_views` → `mosa_imputed_views`), so the train-set size is set by MOSA-imputed views only. Mutations and clinical NaN-pad as before.
- Feature selection: unchanged (`fit_feature_selector`, block-variance + per-modality quota). With 5 blocks and `--min-features-per-modality 100`, clinical (17 features total) ships in full; copynumber gets ≥100; transcriptomics + methylation share the remaining budget.

## 8) What is NOT changed

- MOSA training: model `20260511_174623` stays. CCMA_meta was not part of MOSA's input views and is not retrofitted.
- Notebooks: `notebooks/cba2026_claude_union/` continues to point at the current selected (cross-phenotype) outputs until / unless this branch is promoted.
- Existing `_union` and `_union_ftsel` report roots are untouched.
