# TabPFN MOSA Comparison Commands

Run these commands from the repository root: `/home/scai/scratch/PhenPredCCMA`

The runner locks the target set to the exact 500 CRISPR genes or 500 drugs present in the MOSA outputs for the chosen timestamp, so the original and MOSA variants stay on the same target definition.

## Environment

```bash
PY=/home/scai/anaconda3/envs/mosa/bin/python
TS=20260313_162348
MODEL=/home/scai/scratch/PredCRISPRCCMA/tabpfn/models/tabpfn-v2.5-regressor-v2.5_default.ckpt
OUT=reports/tabpfn
```

## Smoke Check

This runs all 12 combinations with a small target subset and low-cost TabPFN settings. There is no CV pass; each run fits once on the training split and evaluates on the held-out test split.

```bash
for FAMILY in crisprcas9 drugresponse; do
  $PY tabpfn/run_experiment.py \
    --target-family "$FAMILY" \
    --sample-frame both \
    --variant all \
    --mosa-timestamp "$TS" \
    --ccma-dir data/clines/ccma_processed \
    --mosa-files-dir reports/vae/files \
    --out-dir "$OUT" \
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

Build comparison tables and plots from the smoke runs:

```bash
$PY tabpfn/plot_comparison.py \
  --reports-root "$OUT" \
  --mosa-timestamp "$TS"
```

## Full Runs

This runs all combinations for both target families with the default feature budget.

```bash
for FAMILY in crisprcas9 drugresponse; do
  $PY tabpfn/run_experiment.py \
    --target-family "$FAMILY" \
    --sample-frame both \
    --variant all \
    --mosa-timestamp "$TS" \
    --ccma-dir data/clines/ccma_processed \
    --mosa-files-dir reports/vae/files \
    --out-dir "$OUT" \
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

Build the final comparison tables and plots:

```bash
$PY tabpfn/plot_comparison.py \
  --reports-root "$OUT" \
  --mosa-timestamp "$TS"
```

## Outputs

Per-run outputs are written under:

```bash
reports/tabpfn/${TS}/<target_family>/<sample_frame>/<variant>/
```

Each run writes `metrics_test.json`, `metrics_test_per_target.csv`, `target_fit_diagnostics.csv`, `selected_features.json`, `config_used.json`, `pred_test.csv.gz`, and `split_indices.npz`.

Combined comparison outputs are written under:

```bash
reports/tabpfn/${TS}/comparison/
```
