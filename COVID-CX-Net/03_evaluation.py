#!/usr/bin/env python3
"""
03_evaluation.py
================
Projection-stratified evaluation: AP-only, PA-only, mixed accuracy, AUC,
macro-F1, VBS computation with bootstrap CI, cross-dataset transfer,
and per-pathology VBS for MIMIC-CXR multi-label task.

Paper: "Projection-Induced Domain Shift in Chest X-Ray Classification"
Results: Tables 3, 5, 6, 8, 9, 10

Usage:
------
# Within-dataset projection-stratified evaluation:
python 03_evaluation.py \
    --checkpoint checkpoints/best_seed0.pth \
    --test_csv   data/manifests/dataset_a/test.csv \
    --arch       covidcxnet \
    --num_classes 3 \
    --mode within

# Cross-dataset transfer evaluation (train on B, test on C):
python 03_evaluation.py \
    --checkpoint checkpoints/dataset_b/best_seed0.pth \
    --test_csv   data/manifests/dataset_c/test.csv \
    --arch       covidcxnet \
    --num_classes 3 \
    --mode cross \
    --train_name "B (85% PA)" \
    --test_name  "C (92% AP)"

# Per-pathology VBS (MIMIC-CXR multi-label):
python 03_evaluation.py \
    --checkpoint checkpoints/mimic/best_seed0.pth \
    --test_csv   data/manifests/mimic/test.csv \
    --arch       covidcxnet \
    --num_classes 5 \
    --mode mimic_pathology
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from models import build_model
from utils import (
    CXRDataset, get_transforms,
    compute_vbs, compute_vbs_within,
    bootstrap_ci, compute_metrics,
)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: str, device: torch.device):
    """Load model from checkpoint. Returns (model, metadata_dict)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch        = ckpt.get("arch", "covidcxnet")
    num_classes = ckpt.get("num_classes", 3)
    model = build_model(arch, num_classes, pretrained=False).to(device)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    df: pd.DataFrame,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    device: torch.device = None,
) -> dict:
    """
    Run model inference on a manifest DataFrame.

    Returns:
        dict with keys: y_true, y_pred, y_prob, meta (list of dicts).
    """
    if device is None:
        device = next(model.parameters()).device

    tf  = get_transforms("test", image_size)
    ds  = CXRDataset(df, transform=tf, return_meta=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    all_true, all_pred, all_prob, all_meta = [], [], [], []
    for images, labels, metas in loader:
        images = images.to(device)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1)
        all_true.extend(labels.tolist())
        all_pred.extend(logits.argmax(dim=1).cpu().tolist())
        all_prob.extend(probs.cpu().tolist())
        for i in range(len(labels)):
            all_meta.append({k: v[i] if isinstance(v, list) else v for k, v in metas.items()})

    return {
        "y_true": np.array(all_true),
        "y_pred": np.array(all_pred),
        "y_prob": np.array(all_prob),
        "meta":   all_meta,
    }


# ---------------------------------------------------------------------------
# Projection-stratified evaluation
# ---------------------------------------------------------------------------

def evaluate_projection_stratified(
    results: dict,
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """
    Compute accuracy / AUC / macro-F1 for AP-only, PA-only, and mixed subsets.
    Computes VBS_within = Acc(AP subset) - Acc(PA subset).

    Args:
        results:     Output of run_inference.
        df:          Manifest DataFrame (must have 'projection' column).
        n_bootstrap: Bootstrap iterations.
        ci:          Confidence level.
        seed:        Bootstrap seed.

    Returns:
        dict with per-stratum metrics and VBS_within.
    """
    y_true = results["y_true"]
    y_pred = results["y_pred"]
    y_prob = results["y_prob"]

    output = {}

    if "projection" not in df.columns:
        print("[WARN] No 'projection' column found; computing mixed metrics only.")
        m = compute_metrics(y_true, y_pred, y_prob)
        # Bootstrap CI for accuracy
        _, lo, hi = bootstrap_ci(y_true, y_pred,
                                  metric_fn=lambda yt, yp: (yt == yp).mean(),
                                  n_iter=n_bootstrap, ci=ci, seed=seed)
        output["mixed"] = {**m, "acc_ci": (lo, hi)}
        return output

    projections = df["projection"].values

    for subset_name, proj_val in [("AP-only", "AP"), ("PA-only", "PA"), ("Mixed", None)]:
        if proj_val is not None:
            mask = projections == proj_val
        else:
            mask = np.ones(len(y_true), dtype=bool)

        if mask.sum() == 0:
            print(f"  [WARN] No samples for {subset_name}")
            continue

        yt = y_true[mask]
        yp = y_pred[mask]
        yb = y_prob[mask]

        m = compute_metrics(yt, yp, yb)

        # Stratified bootstrap (stratify by projection + class)
        if proj_val is None:
            strat = projections
        else:
            strat = None

        _, acc_lo, acc_hi = bootstrap_ci(
            yt, yp,
            metric_fn=lambda a, b: (a == b).mean(),
            n_iter=n_bootstrap, ci=ci, seed=seed,
            stratify_by=strat[mask] if strat is not None else None,
        )
        output[subset_name] = {
            **m,
            "n": int(mask.sum()),
            "acc_ci": (acc_lo, acc_hi),
        }
        print(f"  {subset_name:10s} (n={mask.sum():6d}): "
              f"acc={m['accuracy']:.4f} [{acc_lo:.4f}–{acc_hi:.4f}]  "
              f"f1={m['macro_f1']:.4f}  "
              f"auc={m.get('auc', float('nan')):.4f}")

    # VBS_within
    if "AP-only" in output and "PA-only" in output:
        vbs = compute_vbs_within(
            y_true[projections == "AP"], y_pred[projections == "AP"],
            y_true[projections == "PA"], y_pred[projections == "PA"],
        )
        output["VBS_within"] = float(vbs)
        print(f"  VBS_within = {vbs*100:.2f}%")

    return output


# ---------------------------------------------------------------------------
# Per-pathology VBS (MIMIC-CXR multi-label)
# ---------------------------------------------------------------------------

def evaluate_per_pathology_vbs(
    model: torch.nn.Module,
    test_df: pd.DataFrame,
    pathology_cols: list,
    image_size: int = 224,
    batch_size: int = 32,
    device: torch.device = None,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Compute per-pathology VBS_within for the MIMIC-CXR multi-label task.
    Returns a DataFrame matching Table 10 of the paper.

    VBS_within per pathology = PA_AUC - AP_AUC.
    """
    if device is None:
        device = next(model.parameters()).device

    tf = get_transforms("test", image_size)

    rows = []
    for path_col in pathology_cols:
        if path_col not in test_df.columns:
            print(f"  [WARN] Column '{path_col}' not in test_df, skipping.")
            continue

        ap_df = test_df[test_df["projection"] == "AP"].copy()
        pa_df = test_df[test_df["projection"] == "PA"].copy()

        def get_binary_auc(subset_df, col):
            ds = CXRDataset(subset_df, transform=tf)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
            probs, y = [], []
            with torch.no_grad():
                for imgs, labels in loader:
                    logits = model(imgs.to(device))
                    p = torch.softmax(logits, dim=1)
                    probs.extend(p[:, 1].cpu().tolist())  # binary: positive class
                    y.extend(subset_df[col].values[:len(labels)])
            try:
                return roc_auc_score(y, probs)
            except ValueError:
                return float("nan")

        # For multi-label we use the dedicated pathology label columns
        ap_df_bin = ap_df[["filepath", path_col]].rename(columns={path_col: "label"})
        pa_df_bin = pa_df[["filepath", path_col]].rename(columns={path_col: "label"})

        # AUC for full mixed test set
        mixed_df_bin = test_df[["filepath", path_col]].rename(columns={path_col: "label"})

        auc_mixed = get_binary_auc(mixed_df_bin, "label")
        auc_ap    = get_binary_auc(ap_df_bin, "label")
        auc_pa    = get_binary_auc(pa_df_bin, "label")
        vbs       = auc_pa - auc_ap

        rows.append({
            "Pathology":   path_col,
            "AUC (Mixed)": round(auc_mixed, 3),
            "AUC (AP)":    round(auc_ap, 3),
            "AUC (PA)":    round(auc_pa, 3),
            "VBS_within":  round(vbs * 100, 1),
        })
        print(f"  {path_col:20s}: AUC(mixed)={auc_mixed:.3f}  "
              f"AUC(AP)={auc_ap:.3f}  AUC(PA)={auc_pa:.3f}  "
              f"VBS={vbs*100:.1f}%")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Projection-stratified evaluation and VBS computation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--test_csv",     required=True)
    p.add_argument("--arch",         default="covidcxnet")
    p.add_argument("--num_classes",  type=int, default=3)
    p.add_argument("--mode",
                   choices=["within", "cross", "mimic_pathology"],
                   default="within")
    p.add_argument("--train_name",   default="Train",
                   help="Label for train dataset (cross-dataset mode)")
    p.add_argument("--test_name",    default="Test",
                   help="Label for test dataset (cross-dataset mode)")
    p.add_argument("--pathology_cols", nargs="+",
                   default=["Atelectasis", "Cardiomegaly", "Consolidation",
                             "Edema", "Pleural Effusion"],
                   help="Pathology columns for MIMIC-CXR multi-label evaluation")
    p.add_argument("--image_size",   type=int, default=224)
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--bootstrap_n",  type=int, default=1000)
    p.add_argument("--out_dir",      default="results")
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--device",       default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else args.device
    )
    model, ckpt_meta = load_checkpoint(args.checkpoint, device)
    test_df = pd.read_csv(args.test_csv)

    print(f"\nEvaluating: {args.checkpoint}")
    print(f"Test set: {len(test_df)} images ({args.test_name})")

    if args.mode in ("within", "cross"):
        results = run_inference(model, test_df, args.image_size,
                                args.batch_size, args.num_workers, device)
        metrics = evaluate_projection_stratified(
            results, test_df,
            n_bootstrap=args.bootstrap_n, seed=args.seed
        )

        # Confusion matrix
        cm = confusion_matrix(results["y_true"], results["y_pred"])
        print(f"\nConfusion matrix:\n{cm}")

        suffix = f"{args.train_name}_to_{args.test_name}".replace(" ", "_")
        out_path = os.path.join(args.out_dir, f"eval_{suffix}_seed{args.seed}.json")
        with open(out_path, "w") as f:
            json.dump({
                "train":   args.train_name,
                "test":    args.test_name,
                "metrics": {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                             for k, v in metrics.items()},
                "confusion_matrix": cm.tolist(),
            }, f, indent=2, default=float)
        print(f"\nResults saved → {out_path}")

    elif args.mode == "mimic_pathology":
        print("\nPer-pathology VBS (MIMIC-CXR multi-label):")
        result_df = evaluate_per_pathology_vbs(
            model, test_df, args.pathology_cols,
            args.image_size, args.batch_size, device, args.seed
        )
        out_csv = os.path.join(args.out_dir, f"per_pathology_vbs_seed{args.seed}.csv")
        result_df.to_csv(out_csv, index=False)
        print(f"\n{result_df.to_string(index=False)}")
        print(f"\nSaved → {out_csv}")


if __name__ == "__main__":
    main()
