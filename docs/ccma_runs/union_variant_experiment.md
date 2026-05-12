# CCMA MOSA Union-Variant Experiment

Date added: 2026-05-11

Purpose: rerun the MOSA + downstream pipeline with the sample-inclusion criterion relaxed from **≥2 omics per sample** (currently selected, MOSA timestamp `20260505_131645`, ~340 augmented cell lines) to the **full union ≥1 omic** (~425 augmented cell lines), without disturbing the current selected version. The current version remains the tracker-selected canonical run until this variant either replaces it or is rejected.

## Experiment ID

```bash
EXPERIMENT=ccma_scratch_cnv_union
```

## Why nothing else changes

The `≥2`-vs-`union` switch is a single config field, `min_views_per_sample`, read by `PhenPred/vae/DatasetCCMA.py:_samples_union()`. The per-view `*_mosa_train.csv` / `*_mosa_test.csv` files under `data/clines/ccma_processed/` are already permissive (each contains all samples in that view minus the 24-sample test holdout); the filter is applied at MOSA load time, not baked into the files. The 24-sample test holdout is reused unchanged for direct comparability with the ≥2 run.

## 0) Preconditions

```bash
cd /mnt/scratch/scai/PhenPredCCMA
PY=/home/scai/anaconda3/envs/mosa/bin/python
mkdir -p docs/ccma_runs/logs
```

Sanity checks before launching:

```bash
test -f reports/vae/configs/hyperparameters_ccma_scratch_cnv_union.json
grep '"min_views_per_sample"' reports/vae/configs/hyperparameters_ccma_scratch_cnv_union.json   # expect 1
grep '"dataname"' reports/vae/configs/hyperparameters_ccma_scratch_cnv_union.json               # expect "ccma_scratch_cnv_union"
ls -lh data/clines/ccma_processed/*_ccma_mosa_train.csv | wc -l                                  # expect 6
```

## 1) Run MOSA (union)

```bash
LOG="docs/ccma_runs/logs/run_ccma_scratch_cnv_union_$(date +%Y%m%d_%H%M%S).log"
set -o pipefail
"$PY" -m PhenPred.vae.Main \
  --hypers-json reports/vae/configs/hyperparameters_ccma_scratch_cnv_union.json \
  2>&1 | tee "$LOG"
echo "exit_code=$? log=$LOG"
```

Once the run finishes, capture the new timestamp from the log or from `reports/vae/files/`:

```bash
export TS_UNION=<new_mosa_timestamp>
ls -lh reports/vae/files/${TS_UNION}_model.pt
ls -lh reports/vae/configs/history/${TS_UNION}_hyperparameters.json
ls -lh reports/vae/files/${TS_UNION}_imputed_copynumber*.csv.gz
grep '"min_views_per_sample"' reports/vae/configs/history/${TS_UNION}_hyperparameters.json   # confirm 1
```

Sample-count check (the headline correctness check for this run):

```bash
grep -Ei "samples|_samples_union|min_views" "$LOG" | head -40
```

Expected: MOSA-train sample count near **~401** (≈ 425 union total − 24 test holdouts). If it lands near the ~316 of the ≥2 run, the config did not take effect — re-check the JSON and the launch command.

## 2) Run the downstream comparison into `_union` report roots

```bash
test -n "${TS_UNION:-}" || { echo "Set TS_UNION first"; exit 1; }

REPORTS_STD=reports/tabpfn_cnv_mosa_only_union
RF_REPORTS=reports/random_forest_cnv_mosa_only_union
MODEL_COMPARE_REPORTS=reports/model_comparison_cnv_mosa_only_union
MODEL_DEFAULT=/home/scai/scratch/PredCRISPRCCMA/tabpfn/models/tabpfn-v2.5-regressor-v2.5_default.ckpt
```

TabPFN (`feature_augmentation`, both `crisprcas9` and `drugresponse`):

```bash
for FAMILY in crisprcas9 drugresponse; do
  "$PY" tabpfn/run_feature_augmentation.py \
    --target-family "$FAMILY" \
    --sample-frame both \
    --variant all \
    --mosa-timestamp "$TS_UNION" \
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
for FAMILY in crisprcas9 drugresponse; do
  "$PY" random_forest/run_feature_augmentation.py \
    --target-family "$FAMILY" \
    --sample-frame both \
    --variant all \
    --mosa-timestamp "$TS_UNION" \
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
  --mosa-timestamp "$TS_UNION"

"$PY" tabpfn/plot_experiment_comparison.py \
  --reports-root "$RF_REPORTS" \
  --experiment-name feature_augmentation \
  --mosa-timestamp "$TS_UNION"

"$PY" model_comparison/plot_model_comparison.py \
  --experiment-name feature_augmentation \
  --mosa-timestamp "$TS_UNION" \
  --tabpfn-root "$REPORTS_STD" \
  --random-forest-root "$RF_REPORTS" \
  --out-dir "$MODEL_COMPARE_REPORTS"
```

Expected final comparison artifacts:

```bash
reports/model_comparison_cnv_mosa_only_union/feature_augmentation/${TS_UNION}/summary_model_comparison.csv
reports/model_comparison_cnv_mosa_only_union/feature_augmentation/${TS_UNION}/aggregate_model_comparison.png
```

Quick headline-r extraction (mirror of `cnv_added_mosa_experiment.md` §5):

```bash
"$PY" - <<'PY'
import os
import pandas as pd

ts = os.environ["TS_UNION"]
path = f"reports/model_comparison_cnv_mosa_only_union/feature_augmentation/{ts}/summary_model_comparison.csv"
df = pd.read_csv(path)
sel = df[
    (df["model_name_tabpfn"] == "tabpfn")
    & (df["sample_frame"] == "expanded")
    & (df["variant"] == "mosa_all")
]
print(sel[["target_family", "test_pearsonr_tabpfn", "test_pearsonr_random_forest", "delta_test_pearsonr_tabpfn_minus_random_forest"]])
PY
```

## 3) (Optional) SHAP for the union run

Run only after step 2 shows the union variant is competitive enough to be worth interpreting. Substitute `$TS_UNION` for the timestamp in `docs/ccma_runs/cnv_added_mosa_experiment.md` §6 — the commands are otherwise identical.

## 4) Figure notebooks (union duplicates)

```bash
cp -r notebooks/cba2026_claude notebooks/cba2026_claude_union
```

In each notebook and helper module inside `notebooks/cba2026_claude_union/`:

- Set `TIMESTAMP = "<TS_UNION>"`.
- Replace report roots `tabpfn_cnv_mosa_only` → `tabpfn_cnv_mosa_only_union`, `random_forest_cnv_mosa_only` → `random_forest_cnv_mosa_only_union`, `model_comparison_cnv_mosa_only` → `model_comparison_cnv_mosa_only_union`.
- Redirect figure outputs to `figure/union/` (or the corresponding `_union`-suffixed equivalent of whatever the original notebook writes to).

Files affected (confirmed by grep for `20260505_131645`):

- `notebooks/cba2026_claude_union/01_global_shap_landscape.ipynb`
- `notebooks/cba2026_claude_union/02_target_level_shap_performance.ipynb`
- `notebooks/cba2026_claude_union/03_headline_performance.ipynb`
- `notebooks/cba2026_claude_union/04_per_target_prediction.ipynb`
- `notebooks/cba2026_claude_union/05_mosa_imputation_quality.ipynb`
- `notebooks/cba2026_claude_union/_prediction_analysis.py`
- `notebooks/cba2026_claude_union/_shap_analysis.py`
- `notebooks/cba2026_claude_union/README.md`

If `notebooks/cba2026/` is also live, mirror the same duplication.

## 5) Tracker / worklog updates

After the union run finishes:

- `docs/ccma_experiment_tracker.md`: add a row in **Other Experiment Branches** for the union variant. Leave **Current Selected Version** pointing at `20260505_131645` / ≥2 unless and until the union variant is promoted.
- `docs/ccma_worklog.md`: log the dated rerun.

## What is NOT changed

- No edits to `data/clines/ccma_processed/`.
- No edits to `PhenPred/vae/*.py` (default `min_views_per_sample=2` stays).
- No edits to `reports/vae/configs/hyperparameters_ccma_scratch_cnv.json`.
- All `reports/vae/files/20260505_131645_*` and the three existing `*_cnv_mosa_only` report trees stay in place.
- `notebooks/cba2026_claude/` continues to point at `20260505_131645`.

## Side-by-side comparison checkpoint

Once both runs have completed comparison artifacts, read off TabPFN `expanded/mosa_all` Pearson r for drug response and CRISPR from:

- Current selected (≥2): `reports/model_comparison_cnv_mosa_only/feature_augmentation/20260505_131645/summary_model_comparison.csv` — drug response `0.3194`, CRISPR `0.2124`.
- Union (this run): `reports/model_comparison_cnv_mosa_only_union/feature_augmentation/${TS_UNION}/summary_model_comparison.csv`.

The delta on those two numbers is the primary decision input for whether to promote the union variant.
