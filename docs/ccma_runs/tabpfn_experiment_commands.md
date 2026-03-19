# TabPFN Experiment Commands

Run these commands from the repository root: `/home/scai/scratch/PhenPredCCMA`

The TabPFN runners now support two estimator modes:
- `--tabpfn-estimator-mode standard`: uses `TabPFNRegressor`
- `--tabpfn-estimator-mode finetune`: uses `FinetunedTabPFNRegressor`

Important:
- `--tabpfn-fit-mode` is only used in `standard` mode.
- `--tabpfn-model-path` accepts one or more checkpoints.
- Use a separate `--out-dir` root per configuration so different checkpoints or finetune runs do not overwrite each other.
- Both workflows still lock the target set to the exact 500 CRISPR genes or 500 drugs present in the MOSA outputs for the chosen timestamp.

## Environment

```bash
PY=/home/scai/anaconda3/envs/mosa/bin/python
TS=20260313_162348
MODEL_DEFAULT=/home/scai/scratch/PredCRISPRCCMA/tabpfn/models/tabpfn-v2.5-regressor-v2.5_default.ckpt
MODEL_SMALL=/home/scai/scratch/PredCRISPRCCMA/tabpfn/models/tabpfn-v2.5-regressor-v2.5_small-samples.ckpt
REPORTS_STD=reports/tabpfn
REPORTS_FT_DEFAULT=reports/tabpfn_finetune/default_ckpt
REPORTS_FT_SMALL=reports/tabpfn_finetune/small_samples_ckpt
REPORTS_FT_ENSEMBLE=reports/tabpfn_finetune/default_plus_small
```

## Standard Feature-Augmentation Full Runs

```bash
for FAMILY in crisprcas9 drugresponse; do
  $PY tabpfn/run_feature_augmentation.py \
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

## Standard Pseudo-Label Full Runs

```bash
for FAMILY in crisprcas9 drugresponse; do
  $PY tabpfn/run_pseudolabel_augmentation.py \
    --target-family "$FAMILY" \
    --variant all \
    --mosa-timestamp "$TS" \
    --ccma-dir data/clines/ccma_processed \
    --mosa-files-dir reports/vae/files \
    --out-dir "$REPORTS_STD/pseudolabel_augmentation" \
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

## Finetune Smoke Check

This is the safest first pass before a full finetune run.

```bash
for FAMILY in crisprcas9 drugresponse; do
  $PY tabpfn/run_feature_augmentation.py \
    --target-family "$FAMILY" \
    --sample-frame both \
    --variant all \
    --mosa-timestamp "$TS" \
    --ccma-dir data/clines/ccma_processed \
    --mosa-files-dir reports/vae/files \
    --out-dir reports/tabpfn_finetune_smoke/default_ckpt/feature_augmentation \
    --target-limit 5 \
    --max-features 300 \
    --min-features-per-modality 25 \
    --tabpfn-estimator-mode finetune \
    --tabpfn-model-path "$MODEL_DEFAULT" \
    --tabpfn-n-estimators 4 \
    --tabpfn-finetune-epochs 5 \
    --tabpfn-finetune-n-estimators 1 \
    --tabpfn-finetune-n-estimators-validation 1 \
    --tabpfn-finetune-n-estimators-final-inference 4 \
    --device cuda \
    --gpu-id 0 \
    --log-every-targets 5
 done
```

## Finetune Feature-Augmentation Full Runs

Using the default TabPFN checkpoint:

```bash
for FAMILY in crisprcas9 drugresponse; do
  $PY tabpfn/run_feature_augmentation.py \
    --target-family "$FAMILY" \
    --sample-frame both \
    --variant all \
    --mosa-timestamp "$TS" \
    --ccma-dir data/clines/ccma_processed \
    --mosa-files-dir reports/vae/files \
    --out-dir "$REPORTS_FT_DEFAULT/feature_augmentation" \
    --max-features 2000 \
    --min-features-per-modality 100 \
    --tabpfn-estimator-mode finetune \
    --tabpfn-model-path "$MODEL_DEFAULT" \
    --tabpfn-n-estimators 8 \
    --tabpfn-finetune-epochs 30 \
    --tabpfn-finetune-learning-rate 1e-5 \
    --tabpfn-finetune-n-estimators 2 \
    --tabpfn-finetune-n-estimators-validation 2 \
    --tabpfn-finetune-n-estimators-final-inference 8 \
    --device cuda \
    --gpu-id 0 \
    --log-every-targets 100
 done
```

Using the small-samples checkpoint:

```bash
for FAMILY in crisprcas9 drugresponse; do
  $PY tabpfn/run_feature_augmentation.py \
    --target-family "$FAMILY" \
    --sample-frame both \
    --variant all \
    --mosa-timestamp "$TS" \
    --ccma-dir data/clines/ccma_processed \
    --mosa-files-dir reports/vae/files \
    --out-dir "$REPORTS_FT_SMALL/feature_augmentation" \
    --max-features 2000 \
    --min-features-per-modality 100 \
    --tabpfn-estimator-mode finetune \
    --tabpfn-model-path "$MODEL_SMALL" \
    --tabpfn-n-estimators 8 \
    --tabpfn-finetune-epochs 30 \
    --tabpfn-finetune-learning-rate 1e-5 \
    --tabpfn-finetune-n-estimators 2 \
    --tabpfn-finetune-n-estimators-validation 2 \
    --tabpfn-finetune-n-estimators-final-inference 8 \
    --device cuda \
    --gpu-id 0 \
    --log-every-targets 100
 done
```

## Finetune Pseudo-Label Full Runs

Using the default TabPFN checkpoint:

```bash
for FAMILY in crisprcas9 drugresponse; do
  $PY tabpfn/run_pseudolabel_augmentation.py \
    --target-family "$FAMILY" \
    --variant all \
    --mosa-timestamp "$TS" \
    --ccma-dir data/clines/ccma_processed \
    --mosa-files-dir reports/vae/files \
    --out-dir "$REPORTS_FT_DEFAULT/pseudolabel_augmentation" \
    --max-features 2000 \
    --min-features-per-modality 100 \
    --tabpfn-estimator-mode finetune \
    --tabpfn-model-path "$MODEL_DEFAULT" \
    --tabpfn-n-estimators 8 \
    --tabpfn-finetune-epochs 30 \
    --tabpfn-finetune-learning-rate 1e-5 \
    --tabpfn-finetune-n-estimators 2 \
    --tabpfn-finetune-n-estimators-validation 2 \
    --tabpfn-finetune-n-estimators-final-inference 8 \
    --device cuda \
    --gpu-id 0 \
    --log-every-targets 100
 done
```

Using the small-samples checkpoint:

```bash
for FAMILY in crisprcas9 drugresponse; do
  $PY tabpfn/run_pseudolabel_augmentation.py \
    --target-family "$FAMILY" \
    --variant all \
    --mosa-timestamp "$TS" \
    --ccma-dir data/clines/ccma_processed \
    --mosa-files-dir reports/vae/files \
    --out-dir "$REPORTS_FT_SMALL/pseudolabel_augmentation" \
    --max-features 2000 \
    --min-features-per-modality 100 \
    --tabpfn-estimator-mode finetune \
    --tabpfn-model-path "$MODEL_SMALL" \
    --tabpfn-n-estimators 8 \
    --tabpfn-finetune-epochs 30 \
    --tabpfn-finetune-learning-rate 1e-5 \
    --tabpfn-finetune-n-estimators 2 \
    --tabpfn-finetune-n-estimators-validation 2 \
    --tabpfn-finetune-n-estimators-final-inference 8 \
    --device cuda \
    --gpu-id 0 \
    --log-every-targets 100
 done
```

## Optional Multi-Checkpoint Finetune Run

If you want to pass both local checkpoints as a checkpoint ensemble:

```bash
for FAMILY in crisprcas9 drugresponse; do
  $PY tabpfn/run_feature_augmentation.py \
    --target-family "$FAMILY" \
    --sample-frame both \
    --variant all \
    --mosa-timestamp "$TS" \
    --ccma-dir data/clines/ccma_processed \
    --mosa-files-dir reports/vae/files \
    --out-dir "$REPORTS_FT_ENSEMBLE/feature_augmentation" \
    --max-features 2000 \
    --min-features-per-modality 100 \
    --tabpfn-estimator-mode finetune \
    --tabpfn-model-path "$MODEL_DEFAULT" "$MODEL_SMALL" \
    --tabpfn-n-estimators 8 \
    --tabpfn-finetune-epochs 30 \
    --tabpfn-finetune-n-estimators 2 \
    --tabpfn-finetune-n-estimators-validation 2 \
    --tabpfn-finetune-n-estimators-final-inference 8 \
    --device cuda \
    --gpu-id 0 \
    --log-every-targets 100
 done
```

## Plot Aggregation

Standard runs:

```bash
$PY tabpfn/plot_experiment_comparison.py \
  --reports-root "$REPORTS_STD" \
  --experiment-name feature_augmentation \
  --mosa-timestamp "$TS"

$PY tabpfn/plot_experiment_comparison.py \
  --reports-root "$REPORTS_STD" \
  --experiment-name pseudolabel_augmentation \
  --mosa-timestamp "$TS"
```

Finetuned default-checkpoint runs:

```bash
$PY tabpfn/plot_experiment_comparison.py \
  --reports-root "$REPORTS_FT_DEFAULT" \
  --experiment-name feature_augmentation \
  --mosa-timestamp "$TS"

$PY tabpfn/plot_experiment_comparison.py \
  --reports-root "$REPORTS_FT_DEFAULT" \
  --experiment-name pseudolabel_augmentation \
  --mosa-timestamp "$TS"
```

Finetuned small-samples runs:

```bash
$PY tabpfn/plot_experiment_comparison.py \
  --reports-root "$REPORTS_FT_SMALL" \
  --experiment-name feature_augmentation \
  --mosa-timestamp "$TS"

$PY tabpfn/plot_experiment_comparison.py \
  --reports-root "$REPORTS_FT_SMALL" \
  --experiment-name pseudolabel_augmentation \
  --mosa-timestamp "$TS"
```

## Output Layout

Feature augmentation runs are written under:

```bash
<reports-root>/feature_augmentation/${TS}/<target_family>/<sample_frame>/<variant>/
```

Pseudo-label augmentation runs are written under:

```bash
<reports-root>/pseudolabel_augmentation/${TS}/<target_family>/<variant>/
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
