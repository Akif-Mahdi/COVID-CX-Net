"""
utils/datasets.py
=================
Dataset classes, image transforms, and AP/PA pair utilities for
"Projection-Induced Domain Shift in Chest X-Ray Classification."
"""

from __future__ import annotations
import os
import glob
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Standard image transforms
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(
    split: str = "train",
    image_size: int = 224,
) -> transforms.Compose:
    """
    Return standard image transforms for train / val / test splits.

    Note: Horizontal flipping is DISABLED to preserve anatomical
    left/right orientation (paper Section 4, Supplementary Section 4).

    Args:
        split:      "train" | "val" | "test"
        image_size: Target spatial size (default 224).

    Returns:
        torchvision.transforms.Compose
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            # NO horizontal flip — preserves cardiac anatomy laterality
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


# ---------------------------------------------------------------------------
# Generic CXR dataset
# ---------------------------------------------------------------------------

class CXRDataset(Dataset):
    """
    Generic chest X-ray dataset.

    Expects a DataFrame with columns:
        - 'filepath'   : absolute path to image (PNG/JPG/DICOM)
        - 'label'      : integer class index
        - 'projection' : 'AP' or 'PA' (optional; used for VBS evaluation)

    Args:
        df:         pandas DataFrame with filepath, label, [projection].
        transform:  Image transform pipeline.
        return_meta: If True, also return a dict with filepath + projection.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[Callable] = None,
        return_meta: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.return_meta = return_meta

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["filepath"]

        # Load image
        if str(path).lower().endswith(".dcm"):
            image = self._load_dicom(path)
        else:
            image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(row["label"])
        if self.return_meta:
            meta = {
                "filepath":   str(path),
                "projection": str(row.get("projection", "unknown")),
            }
            return image, label, meta
        return image, label

    @staticmethod
    def _load_dicom(path: str) -> Image.Image:
        """Load a DICOM file and return a PIL RGB image."""
        try:
            import pydicom
        except ImportError:
            raise ImportError("Install pydicom: pip install pydicom")
        dcm = pydicom.dcmread(str(path))
        arr = dcm.pixel_array.astype(float)
        pi  = getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")
        if "MONOCHROME1" in pi:
            arr = arr.max() - arr
        arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")


# ---------------------------------------------------------------------------
# Paired AP/PA dataset (for view-consistency regularization)
# ---------------------------------------------------------------------------

class PairedCXRDataset(Dataset):
    """
    Dataset that returns matched AP/PA image pairs for a single patient.

    Used to compute the view-consistency loss L_view (Equation 2 in the paper).

    Args:
        df:        DataFrame with columns: filepath, label, projection, subject_id.
        transform: Image transform.

    Only patients with BOTH an AP and PA image in df are retained.
    """

    def __init__(self, df: pd.DataFrame, transform: Optional[Callable] = None):
        self.transform = transform

        # Find subjects with both AP and PA
        has_ap = set(df[df["projection"] == "AP"]["subject_id"].unique())
        has_pa = set(df[df["projection"] == "PA"]["subject_id"].unique())
        paired = sorted(has_ap & has_pa)

        self.pairs = []
        for sid in paired:
            sub = df[df["subject_id"] == sid]
            ap_rows = sub[sub["projection"] == "AP"]
            pa_rows = sub[sub["projection"] == "PA"]
            # Use the first AP and first PA for each subject
            ap_row = ap_rows.iloc[0]
            pa_row = pa_rows.iloc[0]
            self.pairs.append({
                "ap_path": ap_row["filepath"],
                "pa_path": pa_row["filepath"],
                "label":   int(ap_row["label"]),
            })

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        ap_img = CXRDataset._load_image_or_dicom(pair["ap_path"])
        pa_img = CXRDataset._load_image_or_dicom(pair["pa_path"])
        if self.transform:
            ap_img = self.transform(ap_img)
            pa_img = self.transform(pa_img)
        return ap_img, pa_img, pair["label"]


# ---------------------------------------------------------------------------
# DICOM metadata extraction for MIMIC-CXR
# ---------------------------------------------------------------------------

def extract_mimic_projection_labels(
    mimic_root: str,
    metadata_csv: str,
    output_csv: str,
) -> pd.DataFrame:
    """
    Extract ViewPosition from MIMIC-CXR DICOM metadata.

    Args:
        mimic_root:   Root directory of MIMIC-CXR DICOM files.
        metadata_csv: Path to mimic-cxr-2.0.0-metadata.csv.
        output_csv:   Path to save enriched DataFrame with projection labels.

    Returns:
        DataFrame with columns: subject_id, study_id, dicom_id,
        ViewPosition, filepath.
    """
    try:
        import pydicom
    except ImportError:
        raise ImportError("Install pydicom: pip install pydicom")

    meta = pd.read_csv(metadata_csv)
    frontal = meta[meta["ViewPosition"].isin(["AP", "PA"])].copy()

    # Build filepath
    def make_path(row):
        pid = f"p{str(row['subject_id'])[:2]}"
        pdir = f"p{row['subject_id']}"
        sdir = f"s{row['study_id']}"
        return os.path.join(mimic_root, "files", pid, pdir, sdir,
                            f"{row['dicom_id']}.dcm")

    frontal["filepath"] = frontal.apply(make_path, axis=1)
    frontal = frontal[frontal["filepath"].apply(os.path.exists)].copy()
    frontal.to_csv(output_csv, index=False)
    print(f"MIMIC-CXR frontal subset: {len(frontal)} DICOMs")
    print(f"  AP: {(frontal['ViewPosition']=='AP').sum()}")
    print(f"  PA: {(frontal['ViewPosition']=='PA').sum()}")
    return frontal


# ---------------------------------------------------------------------------
# Patient-wise stratified split
# ---------------------------------------------------------------------------

def patient_wise_split(
    df: pd.DataFrame,
    subject_col: str = "subject_id",
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    test_frac:  float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split df by unique subject IDs (not by row) to prevent data leakage.
    Used for MIMIC-CXR splits (paper Section 3.2).

    Args:
        df:          DataFrame containing subject_col.
        subject_col: Column name holding patient identifiers.
        train_frac:  Fraction for training (default 0.70).
        val_frac:    Fraction for validation (default 0.15).
        test_frac:   Fraction for test (default 0.15).
        seed:        Random seed (paper uses seed=0 for first run).

    Returns:
        (train_df, val_df, test_df) DataFrames.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        "Fractions must sum to 1.0"

    rng = np.random.default_rng(seed)
    subjects = np.array(sorted(df[subject_col].unique()))
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_subs = set(subjects[:n_train])
    val_subs   = set(subjects[n_train:n_train + n_val])
    test_subs  = set(subjects[n_train + n_val:])

    train_df = df[df[subject_col].isin(train_subs)].copy()
    val_df   = df[df[subject_col].isin(val_subs)].copy()
    test_df  = df[df[subject_col].isin(test_subs)].copy()

    print(f"Patient-wise split: "
          f"train={len(train_df)} ({len(train_subs)} patients), "
          f"val={len(val_df)} ({len(val_subs)} patients), "
          f"test={len(test_df)} ({len(test_subs)} patients)")
    return train_df, val_df, test_df
