# CCMA MOSA Transfer/Scratch + SHAP Command Runbook

## 1) Environment and paths

```bash
cd /mnt/hackathon-team2-8bf1s/scai/PhenPred
PY=/home/ubuntu/miniconda3/envs/mosa/bin/python
mkdir -p docs/ccma_runs/logs
```

Config layout note:

- Training command is unchanged: `python -m PhenPred.vae.Main`
- Default tracked configs live in `reports/vae/configs/`
- Timestamped saved configs live in `reports/vae/configs/history/`
- Runtime outputs remain in `reports/vae/files/`

## 2) Quick sanity checks

```bash
test -f reports/vae/configs/hyperparameters_ccma_transfer.json && echo "ok: transfer config"
test -f docs/ccma_runs/hyperparameters_ccma_scratch.json && echo "ok: scratch config"
test -f models/mosa_pretrained_20231023_092657.pt && echo "ok: pretrained checkpoint"
test -f /mnt/hackathon-team2-8bf1s/scai/PhenPred/data/clines/ccma_processed/methylation_ccma.csv && echo "ok: CCMA methylation"
```

Note:

- With mutations moved to conditionals (not an input view), effective sample count depends on `min_views_per_sample` and overlap between CRISPR/transcriptomics/methylation.
- If sample count is too low, reduce `"min_views_per_sample"` in the active config (for example, `docs/ccma_runs/hyperparameters_ccma_scratch.json`).

## 3) Run transfer learning + internal benchmark

```bash
LOG="docs/ccma_runs/logs/run_ccma_transfer_$(date +%Y%m%d_%H%M%S).log"
set -o pipefail
"$PY" -m PhenPred.vae.Main \
  --hypers-json reports/vae/configs/hyperparameters_ccma_transfer.json \
  2>&1 | tee "$LOG"
echo "exit_code=$? log=$LOG"
```

## 4) Run CCMA from scratch (no checkpoint loading)

```bash
LOG="docs/ccma_runs/logs/run_ccma_scratch_$(date +%Y%m%d_%H%M%S).log"
set -o pipefail
"$PY" -m PhenPred.vae.Main \
  --hypers-json docs/ccma_runs/hyperparameters_ccma_scratch.json \
  2>&1 | tee "$LOG"
echo "exit_code=$? log=$LOG"
```

## 5) Select run timestamp for SHAP

Use the training timestamp from logs/files, for example:

```bash
TS=20260225_133435
TS=20260225_223235
```

Optional check:

```bash
ls -lh reports/vae/files/${TS}_model.pt
ls -lh reports/vae/configs/history/${TS}_hyperparameters.json
```

## 6) Run SHAP (CRISPR target for CCMA)

```bash
"$PY" -m PhenPred.vae.RunCCMAShap \
  --timestamp "$TS" \
  --explain-target crisprcas9 \
  --all-samples \
  --multi-gpu-shap \
  --n-samples 50 \
  --seed 42
```

Optional: skip top-200 feather export

```bash
"$PY" -m PhenPred.vae.RunCCMAShap \
  --timestamp "$TS" \
  --explain-target crisprcas9 \
  --n-samples 50 \
  --seed 42 \
  --skip-top200
```

Optional: custom SHAP batch size (instead of `--all-samples`)

```bash
"$PY" -m PhenPred.vae.RunCCMAShap \
  --timestamp "$TS" \
  --explain-target crisprcas9 \
  --shap-batch-size 64 \
  --multi-gpu-shap \
  --n-samples 50 \
  --seed 42
```

## 7) Check SHAP outputs

```bash
ls -lh reports/vae/files/${TS}_shap_values_crisprcas9.csv.gz
ls -lh reports/vae/files/${TS}_shap_feature_ranking_crisprcas9.csv
ls -lh reports/vae/files/${TS}_shap_omic_ranking_crisprcas9.csv
ls -lh reports/vae/files/${TS}_explanation_crisprcas9.pkl
```

## 8) Open notebook for SHAP analysis

```bash
jupyter lab notebooks/shap_analysis_ccma_crispr.ipynb
```

In the notebook, set:

```python
TIMESTAMP = "20260225_042500"  # replace with your run timestamp
```
