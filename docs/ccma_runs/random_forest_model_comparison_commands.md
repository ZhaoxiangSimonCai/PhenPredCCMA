# Random Forest And Model Comparison Commands

Run these commands from the repository root: `/home/scai/scratch/PhenPredCCMA`

Use [tabpfn_experiment_commands.md](/home/scai/scratch/PhenPredCCMA/docs/ccma_runs/tabpfn_experiment_commands.md) for the TabPFN runs. This file adds the random-forest baseline and the TabPFN-versus-random-forest comparison step.

## Environment

```bash
PY=/home/scai/anaconda3/envs/mosa/bin/python
TS=20260313_162348
RF_REPORTS=reports/random_forest
TABPFN_REPORTS=reports/tabpfn
MODEL_COMPARE_REPORTS=reports/model_comparison
```

## Random Forest Feature Augmentation Smoke Check

```bash
for FAMILY in crisprcas9 drugresponse; do
  $PY random_forest/run_feature_augmentation.py \
    --target-family "$FAMILY" \
    --sample-frame both \
    --variant all \
    --mosa-timestamp "$TS" \
    --ccma-dir data/clines/ccma_processed \
    --mosa-files-dir reports/vae/files \
    --out-dir "$RF_REPORTS/feature_augmentation" \
    --target-limit 5 \
    --max-features 300 \
    --min-features-per-modality 25 \
    --rf-n-estimators 100 \
    --rf-max-features sqrt \
    --rf-n-jobs -1 \
    --log-every-targets 5
done
```

Aggregate the RF feature-augmentation outputs:

```bash
$PY tabpfn/plot_experiment_comparison.py \
  --reports-root "$RF_REPORTS" \
  --experiment-name feature_augmentation \
  --mosa-timestamp "$TS"
```

## Random Forest Pseudo-Label Augmentation Smoke Check

```bash
for FAMILY in crisprcas9 drugresponse; do
  $PY random_forest/run_pseudolabel_augmentation.py \
    --target-family "$FAMILY" \
    --variant all \
    --mosa-timestamp "$TS" \
    --ccma-dir data/clines/ccma_processed \
    --mosa-files-dir reports/vae/files \
    --out-dir "$RF_REPORTS/pseudolabel_augmentation" \
    --target-limit 5 \
    --max-features 300 \
    --min-features-per-modality 25 \
    --rf-n-estimators 100 \
    --rf-max-features sqrt \
    --rf-n-jobs -1 \
    --log-every-targets 5
done
```

Aggregate the RF pseudo-label outputs:

```bash
$PY tabpfn/plot_experiment_comparison.py \
  --reports-root "$RF_REPORTS" \
  --experiment-name pseudolabel_augmentation \
  --mosa-timestamp "$TS"
```

## Random Forest Full Runs

Feature augmentation:

```bash
for FAMILY in crisprcas9 drugresponse; do
  $PY random_forest/run_feature_augmentation.py \
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

Pseudo-label augmentation:

```bash
for FAMILY in crisprcas9 drugresponse; do
  $PY random_forest/run_pseudolabel_augmentation.py \
    --target-family "$FAMILY" \
    --variant all \
    --mosa-timestamp "$TS" \
    --ccma-dir data/clines/ccma_processed \
    --mosa-files-dir reports/vae/files \
    --out-dir "$RF_REPORTS/pseudolabel_augmentation" \
    --max-features 2000 \
    --min-features-per-modality 100 \
    --rf-n-estimators 300 \
    --rf-max-features sqrt \
    --rf-n-jobs -1 \
    --log-every-targets 100
done
```

## TabPFN Versus Random Forest Comparison

Feature augmentation:

```bash
$PY model_comparison/plot_model_comparison.py \
  --experiment-name feature_augmentation \
  --mosa-timestamp "$TS" \
  --tabpfn-root "$TABPFN_REPORTS" \
  --random-forest-root "$RF_REPORTS" \
  --out-dir "$MODEL_COMPARE_REPORTS"
```

Pseudo-label augmentation:

```bash
$PY model_comparison/plot_model_comparison.py \
  --experiment-name pseudolabel_augmentation \
  --mosa-timestamp "$TS" \
  --tabpfn-root "$TABPFN_REPORTS" \
  --random-forest-root "$RF_REPORTS" \
  --out-dir "$MODEL_COMPARE_REPORTS"
```

## Output Layout

Random-forest runs are written under:

```bash
reports/random_forest/<experiment_name>/${TS}/...
```

Cross-model comparison outputs are written under:

```bash
reports/model_comparison/<experiment_name>/${TS}/
```
