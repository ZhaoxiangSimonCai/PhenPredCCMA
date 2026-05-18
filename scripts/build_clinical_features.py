#!/usr/bin/env python3
"""Build clinical predictor CSVs from CCMA_meta.csv.

Produces four sample-by-feature CSVs that match the existing
`<view>_ccma_{overlap,mosa}_{train,test}.csv` convention, encoding:

- `age_years` (continuous, NaN passthrough; downstream preprocessor median-fills)
- `sex_is_male` (binary, NaN where original was missing; `_inferred` collapsed)
- `class__<name>` one-hot dummies; classes with `< min_class_count` train samples
  are grouped into `class__other`.

Class dummies are fit on the MOSA-train slice (largest training cohort) so the
same column set applies to overlap_{train,test} and mosa_{train,test}.

A `clinical_ccma_preprocess_summary.json` is written alongside.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def read_sample_ids_from_transcriptomics(path: Path) -> List[str]:
    df = pd.read_csv(path, index_col=0, nrows=0)
    return df.columns.astype(str).tolist()


def normalize_sex(value: object) -> float:
    if not isinstance(value, str):
        return float("nan")
    s = value.strip().lower()
    if s.endswith("_inferred"):
        s = s[: -len("_inferred")]
    if s == "male":
        return 1.0
    if s == "female":
        return 0.0
    return float("nan")


def build_clinical_frame(
    meta: pd.DataFrame,
    sample_ids: List[str],
    class_columns: List[str],
    rare_classes: set,
) -> pd.DataFrame:
    aligned = meta.reindex(index=sample_ids)

    out = pd.DataFrame(index=pd.Index(sample_ids, name="ModelID"))
    out["age_years"] = pd.to_numeric(aligned["diagnosis_age_years"], errors="coerce").astype(float)
    out["sex_is_male"] = aligned["patient_sex"].map(normalize_sex).astype(float)

    raw_class = aligned["ccma_class"].astype("string")
    for col in class_columns:
        if col == "class__other":
            mask = raw_class.isin(rare_classes)
            mask_na = raw_class.isna()
            out[col] = np.where(mask_na, np.nan, mask.astype(float))
        else:
            class_name = col[len("class__"):]
            mask_na = raw_class.isna()
            out[col] = np.where(mask_na, np.nan, (raw_class == class_name).astype(float))

    out = out.astype(float)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meta-csv",
        type=str,
        default="/home/scai/scratch/PhenPredCCMA/data/CCMA/CCMA_meta.csv",
        help="Path to CCMA_meta.csv (must contain columns ID, ccma_class, diagnosis_age_years, patient_sex).",
    )
    parser.add_argument(
        "--ccma-dir",
        type=str,
        default="data/clines/ccma_processed",
        help="Directory containing the existing per-view CSVs. Clinical files are written here.",
    )
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=5,
        help="Tumor-class labels with fewer than this many training samples are collapsed into class__other.",
    )
    args = parser.parse_args()

    meta_csv = Path(args.meta_csv).expanduser().resolve()
    ccma_dir = Path(args.ccma_dir).expanduser().resolve()

    if not meta_csv.is_file():
        raise FileNotFoundError(f"meta csv not found: {meta_csv}")
    if not ccma_dir.is_dir():
        raise FileNotFoundError(f"ccma_dir not found: {ccma_dir}")

    meta = pd.read_csv(meta_csv)
    meta["ID"] = meta["ID"].astype(str)
    if meta["ID"].duplicated().any():
        dupes = meta.loc[meta["ID"].duplicated(keep=False), "ID"].tolist()
        raise AssertionError(f"Duplicate IDs in meta: {dupes[:10]}")
    meta = meta.set_index("ID")

    transcriptomics_paths = {
        ("overlap", "train"): ccma_dir / "transcriptomics_ccma_overlap_train.csv",
        ("overlap", "test"): ccma_dir / "transcriptomics_ccma_overlap_test.csv",
        ("mosa", "train"): ccma_dir / "transcriptomics_ccma_mosa_train.csv",
        ("mosa", "test"): ccma_dir / "transcriptomics_ccma_mosa_test.csv",
    }
    for path in transcriptomics_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Expected sample-ID source file missing: {path}")

    sample_ids_by_slice: Dict[tuple, List[str]] = {
        slice_key: read_sample_ids_from_transcriptomics(path)
        for slice_key, path in transcriptomics_paths.items()
    }

    mosa_train_ids = sample_ids_by_slice[("mosa", "train")]
    mosa_train_classes = meta["ccma_class"].reindex(mosa_train_ids).dropna()
    class_counts = mosa_train_classes.value_counts()
    rare_classes = set(class_counts[class_counts < int(args.min_class_count)].index.tolist())
    kept_classes = [c for c in class_counts.index if c not in rare_classes]

    class_columns = [f"class__{name}" for name in sorted(kept_classes)]
    if rare_classes:
        class_columns.append("class__other")

    written: Dict[str, dict] = {}
    for (frame_token, split), sample_ids in sample_ids_by_slice.items():
        df = build_clinical_frame(
            meta=meta,
            sample_ids=sample_ids,
            class_columns=class_columns,
            rare_classes=rare_classes,
        )
        out_path = ccma_dir / f"clinical_ccma_{frame_token}_{split}.csv"
        df.to_csv(out_path)
        written[f"{frame_token}_{split}"] = {
            "path": str(out_path),
            "n_samples": int(df.shape[0]),
            "n_features": int(df.shape[1]),
            "n_samples_in_meta": int(meta.index.isin(sample_ids).sum()),
            "n_samples_missing_meta": int(len([s for s in sample_ids if s not in meta.index])),
            "age_nan_count": int(df["age_years"].isna().sum()),
            "sex_nan_count": int(df["sex_is_male"].isna().sum()),
        }
        print(
            f"[clinical] wrote {out_path.name}: "
            f"{df.shape[0]} samples x {df.shape[1]} features, "
            f"{written[f'{frame_token}_{split}']['n_samples_in_meta']}/{df.shape[0]} in meta"
        )

    summary = {
        "meta_csv": str(meta_csv),
        "ccma_dir": str(ccma_dir),
        "min_class_count": int(args.min_class_count),
        "class_counts_on_mosa_train": class_counts.to_dict(),
        "kept_classes": sorted(kept_classes),
        "rare_classes_grouped_to_other": sorted(rare_classes),
        "feature_columns": ["age_years", "sex_is_male"] + class_columns,
        "outputs": written,
    }
    summary_path = ccma_dir / "clinical_ccma_preprocess_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[clinical] wrote summary {summary_path}")


if __name__ == "__main__":
    main()
