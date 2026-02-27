# CCMA Checkpoint Journal

Track each relevant checkpoint timestamp, what generated it, and how it was used.

## Entries


| Date Logged | Run Timestamp   | Checkpoint Path                              | Mode                       | Config                                                   | What Was Done                                                                                      |
| ----------- | --------------- | -------------------------------------------- | -------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| 2026-02-25  | 20231023_092657 | `models/mosa_pretrained_20231023_092657.pt`  | Source (DepMap pretrained) | `reports/vae/files/20231023_092657_hyperparameters.json` | Used as transfer initialization source for CCMA shared-view fine-tuning.                           |
| 2026-02-25  | 20260225_133435 | `reports/vae/files/20260225_133435_model.pt` | CCMA transfer run          | `reports/vae/files/hyperparameters_ccma_transfer.json`   | Trained on CCMA with transfer loading enabled; internal benchmark + CRISPR SHAP outputs generated. |
| 2026-02-25  | 20260225_223235 | `reports/vae/files/<TIMESTAMP>_model.pt`     | CCMA scratch run           | `docs/ccma_runs/hyperparameters_ccma_scratch.json`       | Planned baseline run with transfer loading disabled (`transfer_mode: null`).                       |
| 2026-02-26  | 20260226_024535 | `reports/vae/files/20260226_024535_model.pt` | CCMA transfer run (methylation) | `reports/vae/files/hyperparameters_ccma_transfer.json`   | CCMA transfer run including methylation data; internal benchmark and SHAP for CRISPR generated.    |
| 2026-02-26  | 20260226_031500 | `reports/vae/files/20260226_031500_model.pt` | CCMA scratch run (methylation) | `docs/ccma_runs/hyperparameters_ccma_scratch.json`       | CCMA baseline run including methylation data; trained from scratch without transfer loading.        |



## Update Template

Copy this row template after each new run:

```markdown
| YYYY-MM-DD | YYYYMMDD_HHMMSS | reports/vae/files/YYYYMMDD_HHMMSS_model.pt | ccma_transfer or ccma_scratch | path/to/config.json | short note on outputs/analysis |
```

