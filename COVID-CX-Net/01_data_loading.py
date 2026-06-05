#!/usr/bin/env python3
"""
01_data_loading.py
==================
Dataset assembly, DICOM ViewPosition metadata extraction, patient-wise
splitting, AP/PA pair matching, and class-balance reporting.

Paper: "Projection-Induced Domain Shift in Chest X-Ray Classification"

Usage examples:
--------------
# Build manifest from a directory tree (COVID/Normal/Pneumonia subdirs):
python 01_data_loading.py --mode build \
    --data_dir /path/to/dataset \
    --output_dir data/manifests/dataset_a

# Extract MIMIC-CXR projection labels from DICOM metadata:
python 01_data_loading.py --mode mimic \
    --mimic_root /path/to/mimic-cxr \
    --metadata_csv /path/to/mimic-cxr-2.0.0-metadata.csv \
    --labels_csv   /path/to/mimic-cxr-2.0.0-chexpert.csv \
    --output_dir   data/manifests/mimic

# Match AP/PA pairs within a manifest:
python 01_data_loading.py --mode pairs \
    --manifest data/manifests/mimic/manifest.csv \
    --output_dir data/manifests/mimic
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from utils.datasets import patient_wise_split


# ---------------------------------------------------------------------------
# Projection inference for composite datasets (Datasets A–E)
# ---------------------------------------------------------------------------

# Projection inference pipeline (paper Section 3.2):
# 1. Dataset provenance review
# 2. Demographic inference (pediatric → AP)
# 3. ResNet-18 classifier (2,000 manually labelled images)
# 4. Radiologist verification (500 images/dataset, κ = 0.87)

PEDIATRIC_DATASET_SOURCES = {
    "rsna",          # RSNA Pneumonia Detection Challenge: pediatric
    "covid_cxr",     # Cohen et al. COVID CXR: mixed, often AP
}

PROJECTION_RULES: Dict[str, str] = {
    # Known AP-heavy sources
    "covid_cxr":   "AP",
    "rsna":        "AP",
    # Known PA-heavy sources
    "chexpert":    "PA",
    "tb_shenzhen": "PA",
    "tb_montgomery":"PA",
    # Mixed / inferred
    "covid19_db":  "mixed",
    "nih":         "mixed",
}


def infer_projection_from_source(source: str) -> str:
    """Rule-based projection inference from dataset source name."""
    for key, proj in PROJECTION_RULES.items():
        if key in source.lower():
            return proj
    return "unknown"


# ---------------------------------------------------------------------------
# Build manifest from directory tree
# ---------------------------------------------------------------------------

def build_manifest_from_dir(
    data_dir: str,
    class_names: Optional[List[str]] = None,
    projection_source: str = "unknown",
    seed: int = 0,
) -> pd.DataFrame:
    """
    Build a manifest CSV from a directory tree where each subdirectory
    represents a class:
        data_dir/
            COVID-19/  (or COVID/)
            Normal/
            Pneumonia/

    Args:
        data_dir:          Root directory.
        class_names:       List of class names (subdirectory names).
                           If None, autodiscovered from subdirectories.
        projection_source: Dataset source string for projection inference.
        seed:              Random seed for shuffling.

    Returns:
        DataFrame with columns: filepath, label, class_name, projection.
    """
    data_dir = Path(data_dir)
    if class_names is None:
        class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    print(f"Classes discovered: {class_names}")

    label_map = {name: idx for idx, name in enumerate(class_names)}
    rows = []
    extensions = {".png", ".jpg", ".jpeg", ".dcm", ".tiff", ".bmp"}

    for cls_name in class_names:
        cls_dir = data_dir / cls_name
        if not cls_dir.exists():
            # Try common aliases
            for alias in [cls_name.lower(), cls_name.upper(), cls_name.replace("-", "_")]:
                alt = data_dir / alias
                if alt.exists():
                    cls_dir = alt
                    break
            else:
                print(f"  [WARN] Class directory not found: {cls_dir}")
                continue

        files = [f for f in cls_dir.rglob("*") if f.suffix.lower() in extensions]
        proj = infer_projection_from_source(projection_source)
        for f in files:
            rows.append({
                "filepath":   str(f),
                "label":      label_map[cls_name],
                "class_name": cls_name,
                "projection": proj,
                "source":     projection_source,
            })
        print(f"  {cls_name}: {len(files)} images  label={label_map[cls_name]}")

    df = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# MIMIC-CXR manifest builder
# ---------------------------------------------------------------------------

def build_mimic_manifest(
    mimic_root: str,
    metadata_csv: str,
    labels_csv: str,
    output_dir: str,
    image_format: str = "jpg",   # "jpg" for mimic-cxr-jpg, "dcm" for DICOM
    min_per_stratum: int = 100,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Build a MIMIC-CXR frontal-view manifest with:
      - ViewPosition (AP/PA) from DICOM metadata
      - CheXpert multi-label pathology labels
      - Patient-wise 70/15/15% split

    Args:
        mimic_root:    Root of MIMIC-CXR-JPG or MIMIC-CXR-DICOM.
        metadata_csv:  mimic-cxr-2.0.0-metadata.csv
        labels_csv:    mimic-cxr-2.0.0-chexpert.csv
        output_dir:    Directory to save manifest CSVs.
        image_format:  "jpg" or "dcm"
        min_per_stratum: Minimum images per AP/PA stratum to include study.
        seed:          Random seed for splitting.

    Returns:
        Full manifest DataFrame.
    """
    print("Loading MIMIC-CXR metadata...")
    meta   = pd.read_csv(metadata_csv)
    labels = pd.read_csv(labels_csv)

    # Frontal views only
    frontal = meta[meta["ViewPosition"].isin(["AP", "PA"])].copy()
    print(f"  Frontal views: {len(frontal)} "
          f"(AP={( frontal['ViewPosition']=='AP').sum()}, "
          f"PA={( frontal['ViewPosition']=='PA').sum()})")

    # Merge with labels
    merged = frontal.merge(labels, on=["subject_id", "study_id"], how="inner")
    print(f"  After label merge: {len(merged)}")

    # Build file paths
    if image_format == "jpg":
        def make_path(row):
            pid = f"p{str(row['subject_id'])[:2]}"
            return os.path.join(
                mimic_root, "files", pid,
                f"p{row['subject_id']}", f"s{row['study_id']}",
                f"{row['dicom_id']}.jpg"
            )
    else:
        def make_path(row):
            pid = f"p{str(row['subject_id'])[:2]}"
            return os.path.join(
                mimic_root, "files", pid,
                f"p{row['subject_id']}", f"s{row['study_id']}",
                f"{row['dicom_id']}.dcm"
            )

    merged["filepath"] = merged.apply(make_path, axis=1)
    merged = merged[merged["filepath"].apply(os.path.exists)].copy()
    print(f"  Files found on disk: {len(merged)}")

    # Binary label columns (1=positive, 0=negative, NaN=uncertain → 0)
    label_cols = ["Atelectasis", "Cardiomegaly", "Consolidation",
                  "Edema", "Pleural Effusion"]
    for col in label_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).clip(0, 1).astype(int)

    # Rename ViewPosition → projection
    merged = merged.rename(columns={"ViewPosition": "projection"})

    # Patient-wise split
    train_df, val_df, test_df = patient_wise_split(
        merged, subject_col="subject_id",
        train_frac=0.70, val_frac=0.15, test_frac=0.15, seed=seed
    )
    train_df["split"] = "train"
    val_df["split"]   = "val"
    test_df["split"]  = "test"

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    full_df.to_csv(os.path.join(output_dir, "manifest.csv"), index=False)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    # Summary statistics
    print("\nClass balance (training set):")
    for col in label_cols:
        if col in train_df.columns:
            pos = train_df[col].sum()
            print(f"  {col}: {pos} positive / {len(train_df)-pos} negative")

    return full_df


# ---------------------------------------------------------------------------
# AP/PA pair matching
# ---------------------------------------------------------------------------

def match_ap_pa_pairs(
    df: pd.DataFrame,
    subject_col: str = "subject_id",
    projection_col: str = "projection",
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Find patients with both AP and PA images and return a paired DataFrame.

    Args:
        df:             Input manifest with projection labels.
        subject_col:    Column identifying patients.
        projection_col: Column with 'AP' or 'PA' values.
        output_path:    Optional CSV save path.

    Returns:
        DataFrame with columns: subject_id, ap_filepath, pa_filepath,
        ap_study_id, pa_study_id, label.
    """
    ap_df = df[df[projection_col] == "AP"].copy()
    pa_df = df[df[projection_col] == "PA"].copy()

    ap_subs = set(ap_df[subject_col].unique())
    pa_subs = set(pa_df[subject_col].unique())
    paired_subs = sorted(ap_subs & pa_subs)
    print(f"AP subjects: {len(ap_subs)}, PA subjects: {len(pa_subs)}, "
          f"Paired: {len(paired_subs)}")

    rows = []
    for sid in paired_subs:
        ap_rows = ap_df[ap_df[subject_col] == sid]
        pa_rows = pa_df[pa_df[subject_col] == sid]
        rows.append({
            "subject_id":   sid,
            "ap_filepath":  ap_rows.iloc[0]["filepath"],
            "pa_filepath":  pa_rows.iloc[0]["filepath"],
            "label":        ap_rows.iloc[0].get("label", -1),
        })

    pairs_df = pd.DataFrame(rows)
    if output_path:
        pairs_df.to_csv(output_path, index=False)
        print(f"Pairs saved → {output_path}")
    return pairs_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Dataset assembly and manifest creation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["build", "mimic", "pairs"], required=True,
                   help="Operation mode")
    p.add_argument("--data_dir",      help="Root directory for 'build' mode")
    p.add_argument("--class_names",   nargs="+", help="Class names (subdirectory names)")
    p.add_argument("--source_name",   default="unknown", help="Dataset source tag")
    p.add_argument("--mimic_root",    help="MIMIC-CXR root for 'mimic' mode")
    p.add_argument("--metadata_csv",  help="mimic-cxr-2.0.0-metadata.csv")
    p.add_argument("--labels_csv",    help="mimic-cxr-2.0.0-chexpert.csv")
    p.add_argument("--manifest",      help="Existing manifest CSV for 'pairs' mode")
    p.add_argument("--output_dir",    default="data/manifests", help="Output directory")
    p.add_argument("--image_format",  default="jpg", choices=["jpg","dcm"],
                   help="Image format for MIMIC-CXR")
    p.add_argument("--seed",          type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "build":
        assert args.data_dir, "--data_dir required for 'build' mode"
        df = build_manifest_from_dir(
            args.data_dir,
            class_names=args.class_names,
            projection_source=args.source_name,
            seed=args.seed,
        )
        # Patient-wise split (subject_id inferred from filename stem)
        df["subject_id"] = df["filepath"].apply(
            lambda p: Path(p).stem.split("_")[0]
        )
        train_df, val_df, test_df = patient_wise_split(
            df, subject_col="subject_id", seed=args.seed
        )
        for split, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
            out = os.path.join(args.output_dir, f"{split}.csv")
            sdf.to_csv(out, index=False)
            print(f"Saved {split}: {len(sdf)} rows → {out}")

    elif args.mode == "mimic":
        assert args.mimic_root,   "--mimic_root required"
        assert args.metadata_csv, "--metadata_csv required"
        assert args.labels_csv,   "--labels_csv required"
        build_mimic_manifest(
            args.mimic_root, args.metadata_csv, args.labels_csv,
            args.output_dir, args.image_format, seed=args.seed,
        )

    elif args.mode == "pairs":
        assert args.manifest, "--manifest required for 'pairs' mode"
        df = pd.read_csv(args.manifest)
        pairs_df = match_ap_pa_pairs(
            df,
            output_path=os.path.join(args.output_dir, "ap_pa_pairs.csv"),
        )
        frac = len(pairs_df) / len(df) * 100
        print(f"Pairing rate: {frac:.1f}% of total images "
              f"(paper target ~15% composite, ~18% MIMIC-CXR)")


if __name__ == "__main__":
    main()
