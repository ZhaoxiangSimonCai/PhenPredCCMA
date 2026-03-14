# TabPFN Experiment Commands

Run these commands from the repository root: `/home/scai/scratch/PhenPredCCMA`

The feature-augmentation and pseudo-label augmentation workflows now live in separate scripts. Both runners lock the target set to the exact 500 CRISPR genes or 500 drugs present in the MOSA outputs for the chosen timestamp, and both write richer test outputs including R2, Pearson's r, RMSE, wide prediction/truth matrices, and long-form prediction records.

## Environment

```bash
PY=/home/scai/anaconda3/envs/mosa/bin/python
TS=20260313_162348
MODEL=/home/scai/scratch/PredCRISPRCCMA/tabpfn/models/tabpfn-v2.5-regressor-v2.5_default.ckpt
REPORTS=reports/tabpfn
```

## Feature Augmentation Smoke Check

This reruns the existing experiment explicitly as feature augmentation: real labels, with original or MOSA-imputed predictor features.

```bash
for FAMILY in crisprcas9 drugresponse; do
  $PY tabpfn/run_feature_augmentation.py \
    --target-family "$FAMILY" \
    --sample-frame both \
    --variant all \
    --mosa-timestamp "$TS" \
    --ccma-dir data/clines/ccma_processed \
    --mosa-files-dir reports/vae/files \
    --out-dir "$REPORTS/feature_augmentation" \
    --target-limit 5 \
    --max-features 300 \
    --min-features-per-modality 25 \
    --tabpfn-n-estimators 1 \
    --tabpfn-fit-mode fit_preprocessors \
    --tabpfn-model-path "$MODEL" \
    --device cpu \
    --log-every-targets 5
done
```

Aggregate the feature-augmentation comparison outputs:

```bash
$PY tabpfn/plot_experiment_comparison.py \
  --reports-root "$REPORTS" \
  --experiment-name feature_augmentation \
  --mosa-timestamp "$TS"
```

## Feature Augmentation Full Runs

```bash
for FAMILY in crisprcas9 drugresponse; do
  $PY tabpfn/run_feature_augmentation.py \
    --target-family "$FAMILY" \
    --sample-frame both \
    --variant all \
    --mosa-timestamp "$TS" \
    --ccma-dir data/clines/ccma_processed \
    --mosa-files-dir reports/vae/files \
    --out-dir "$REPORTS/feature_augmentation" \
    --max-features 2000 \
    --min-features-per-modality 100 \
    --tabpfn-n-estimators 8 \
    --tabpfn-fit-mode fit_preprocessors \
    --tabpfn-model-path "$MODEL" \
    --device cuda \
    --gpu-id 0 \
    --log-every-targets 100
done
```

## Pseudo-Label Augmentation Smoke Check

This new experiment uses MOSA target matrices as train labels while keeping the held-out test labels real. Predictor features come from the raw expanded CCMA matrices with training-only imputation in preprocessing.

```bash
for FAMILY in crisprcas9 drugresponse; do
  $PY tabpfn/run_pseudolabel_augmentation.py \
    --target-family "$FAMILY" \
    --variant all \
    --mosa-timestamp "$TS" \
    --ccma-dir data/clines/ccma_processed \
    --mosa-files-dir reports/vae/files \
    --out-dir "$REPORTS/pseudolabel_augmentation" \
    --target-limit 5 \
    --max-features 300 \
    --min-features-per-modality 25 \
    --tabpfn-n-estimators 1 \
    --tabpfn-fit-mode fit_preprocessors \
    --tabpfn-model-path "$MODEL" \
    --device cpu \
    --log-every-targets 5
done
```

Aggregate the pseudo-label comparison outputs:

```bash
$PY tabpfn/plot_experiment_comparison.py \
  --reports-root "$REPORTS" \
  --experiment-name pseudolabel_augmentation \
  --mosa-timestamp "$TS"
```

## Pseudo-Label Augmentation Full Runs

```bash
for FAMILY in crisprcas9 drugresponse; do
  $PY tabpfn/run_pseudolabel_augmentation.py \
    --target-family "$FAMILY" \
    --variant all \
    --mosa-timestamp "$TS" \
    --ccma-dir data/clines/ccma_processed \
    --mosa-files-dir reports/vae/files \
    --out-dir "$REPORTS/pseudolabel_augmentation" \
    --max-features 2000 \
    --min-features-per-modality 100 \
    --tabpfn-n-estimators 8 \
    --tabpfn-fit-mode fit_preprocessors \
    --tabpfn-model-path "$MODEL" \
    --device cuda \
    --gpu-id 0 \
    --log-every-targets 100
done
```

## Output Layout

Feature augmentation runs are written under:

```bash
reports/tabpfn/feature_augmentation/${TS}/<target_family>/<sample_frame>/<variant>/
```

Pseudo-label augmentation runs are written under:

```bash
reports/tabpfn/pseudolabel_augmentation/${TS}/<target_family>/<variant>/
```

Each run writes:

```bash
test_metrics_summary.json
test_metrics_per_target.csv
test_predictions_wide.csv.gz
test_truth_wide.csv.gz
test_prediction_records.csv.gz
target_fit_diagnostics.csv
selected_features.json
config_used.json
split_indices.npz
```

Aggregated comparison outputs are written under:

```bash
reports/tabpfn/<experiment_name>/${TS}/comparison/
```
