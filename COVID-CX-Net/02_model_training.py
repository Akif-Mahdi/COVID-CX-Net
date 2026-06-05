#!/usr/bin/env python3
"""
02_model_training.py
====================
Training loop for COVID-CX-Net (and baselines) with view-consistency
regularization (λ grid search), cosine LR decay, and early stopping.

Paper: "Projection-Induced Domain Shift in Chest X-Ray Classification"
Table 4: λ ∈ {0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30}
Optimal: λ = 0.1 (reduces VBS from 11.5% → 6.7%)

Usage:
------
# Train COVID-CX-Net with λ=0.1 (optimal):
python 02_model_training.py \
    --train_csv  data/manifests/dataset_a/train.csv \
    --val_csv    data/manifests/dataset_a/val.csv \
    --arch       covidcxnet \
    --num_classes 3 \
    --lambda_view 0.1 \
    --pairs_csv  data/manifests/dataset_a/ap_pa_pairs.csv \
    --out_dir    checkpoints/dataset_a_covidcxnet_l01 \
    --seed 0

# λ grid search (trains one model per λ):
python 02_model_training.py \
    --train_csv  data/manifests/dataset_a/train.csv \
    --val_csv    data/manifests/dataset_a/val.csv \
    --arch       covidcxnet \
    --lambda_grid 0.0 0.01 0.05 0.10 0.15 0.20 0.30 \
    --out_dir    checkpoints/gridsearch
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from models import build_model
from utils import (
    CXRDataset, PairedCXRDataset, get_transforms,
    ProjectionAwareLoss, compute_metrics,
)

SEEDS       = [0, 1, 2, 3, 4]       # 5 independent seeds (composite datasets)
SEEDS_MIMIC = list(range(10))        # 10 seeds for MIMIC-CXR


# ---------------------------------------------------------------------------
# Deterministic setup
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    import random, torch, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Training loop (single seed)
# ---------------------------------------------------------------------------

def train_one_seed(
    train_df:     pd.DataFrame,
    val_df:       pd.DataFrame,
    arch:         str,
    num_classes:  int,
    lambda_view:  float,
    pairs_df:     pd.DataFrame | None,
    out_dir:      str,
    seed:         int,
    epochs:       int = 50,
    patience:     int = 10,
    lr:           float = 1e-4,
    batch_size:   int = 32,
    image_size:   int = 224,
    num_workers:  int = 4,
    device:       str = "auto",
) -> dict:
    """
    Train a single model for one seed.

    Returns:
        dict with best validation accuracy, epoch, and checkpoint path.
    """
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"\n[Seed {seed}] arch={arch}, λ={lambda_view}, device={dev}")

    # ── Datasets ──────────────────────────────────────────────────────────
    tf_train = get_transforms("train", image_size)
    tf_val   = get_transforms("val",   image_size)

    train_ds = CXRDataset(train_df, transform=tf_train)
    val_ds   = CXRDataset(val_df,   transform=tf_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    # Paired loader for view-consistency loss
    pair_loader = None
    if pairs_df is not None and lambda_view > 0 and len(pairs_df) > 0:
        pair_ds = PairedCXRDataset(pairs_df, transform=tf_train)
        if len(pair_ds) > 0:
            pair_loader = DataLoader(pair_ds, batch_size=batch_size // 2,
                                     shuffle=True, num_workers=num_workers,
                                     pin_memory=True, drop_last=True)
            print(f"  Paired dataset: {len(pair_ds)} pairs")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(arch, num_classes, pretrained=True).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params/1e6:.2f}M total, {n_train/1e6:.2f}M trainable")

    # ── Optimizer and scheduler ───────────────────────────────────────────
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 1e-3
    )
    criterion = ProjectionAwareLoss(lambda_view=lambda_view, num_classes=num_classes)

    # ── Training ──────────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_epoch   = 0
    no_improve   = 0
    history      = []
    ckpt_path    = os.path.join(out_dir, f"best_seed{seed}.pth")

    pair_iter = iter(pair_loader) if pair_loader else None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for images, labels in train_loader:
            images = images.to(dev, non_blocking=True)
            labels = labels.to(dev, non_blocking=True)

            # Unpaired forward pass
            logits = model(images)

            # View-consistency: get embeddings for paired batch
            embed_ap = embed_pa = None
            if pair_loader is not None and lambda_view > 0:
                try:
                    ap_imgs, pa_imgs, _ = next(pair_iter)
                except StopIteration:
                    pair_iter = iter(pair_loader)
                    ap_imgs, pa_imgs, _ = next(pair_iter)
                ap_imgs = ap_imgs.to(dev, non_blocking=True)
                pa_imgs = pa_imgs.to(dev, non_blocking=True)
                # Get embeddings without gradient for view loss
                if hasattr(model, "embed"):
                    embed_ap = model.embed(ap_imgs)
                    embed_pa = model.embed(pa_imgs)

            loss, comps = criterion(logits, labels, embed_ap, embed_pa)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_preds, val_labels, val_probs = [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(dev)
                logits = model(images)
                probs  = torch.softmax(logits, dim=1)
                val_preds.extend(logits.argmax(dim=1).cpu().tolist())
                val_labels.extend(labels.tolist())
                val_probs.extend(probs.cpu().tolist())

        val_metrics = compute_metrics(
            np.array(val_labels), np.array(val_preds),
            y_prob=np.array(val_probs)
        )
        train_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

        row = {
            "epoch":    epoch,
            "train_loss": total_loss / len(train_loader),
            "train_acc":  train_acc,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": scheduler.get_last_lr()[0],
            "elapsed": time.time() - t0,
        }
        history.append(row)

        print(f"  Epoch {epoch:3d} | loss={row['train_loss']:.4f} | "
              f"train_acc={train_acc:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
              f"val_f1={val_metrics['macro_f1']:.4f} | "
              f"lr={row['lr']:.2e} | {row['elapsed']:.1f}s")

        # ── Early stopping ────────────────────────────────────────────────
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch   = epoch
            no_improve   = 0
            torch.save({
                "epoch":      epoch,
                "arch":       arch,
                "num_classes": num_classes,
                "lambda_view": lambda_view,
                "seed":        seed,
                "state_dict":  model.state_dict(),
                "val_metrics": val_metrics,
            }, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best val_acc={best_val_acc:.4f} at epoch {best_epoch})")
                break

    # Save training history
    pd.DataFrame(history).to_csv(
        os.path.join(out_dir, f"history_seed{seed}.csv"), index=False
    )

    result = {
        "arch":         arch,
        "lambda_view":  lambda_view,
        "seed":         seed,
        "best_val_acc": best_val_acc,
        "best_epoch":   best_epoch,
        "checkpoint":   ckpt_path,
    }
    with open(os.path.join(out_dir, f"result_seed{seed}.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------

def train_multi_seed(
    train_df, val_df, arch, num_classes, lambda_view,
    pairs_df, out_dir, seeds=SEEDS, **kwargs
) -> list:
    """Train the same configuration across multiple seeds."""
    results = []
    for seed in seeds:
        seed_dir = os.path.join(out_dir, f"seed_{seed}")
        r = train_one_seed(
            train_df=train_df, val_df=val_df, arch=arch,
            num_classes=num_classes, lambda_view=lambda_view,
            pairs_df=pairs_df, out_dir=seed_dir, seed=seed, **kwargs
        )
        results.append(r)

    accs = [r["best_val_acc"] for r in results]
    print(f"\n[{arch}, λ={lambda_view}] "
          f"Val acc: {np.mean(accs):.4f} ± {np.std(accs):.4f} "
          f"(n={len(accs)} seeds)")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train COVID-CX-Net with view-consistency regularization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train_csv",    required=True)
    p.add_argument("--val_csv",      required=True)
    p.add_argument("--pairs_csv",    default=None,
                   help="AP/PA pairs CSV for view-consistency loss")
    p.add_argument("--arch",         default="covidcxnet",
                   choices=["covidcxnet", "vgg16", "resnet50", "densenet121"])
    p.add_argument("--num_classes",  type=int, default=3)
    p.add_argument("--lambda_view",  type=float, default=0.1,
                   help="View-consistency regularization weight λ")
    p.add_argument("--lambda_grid",  type=float, nargs="+", default=None,
                   help="Run a λ grid search instead of single λ")
    p.add_argument("--out_dir",      default="checkpoints")
    p.add_argument("--epochs",       type=int, default=50)
    p.add_argument("--patience",     type=int, default=10)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--image_size",   type=int, default=224)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--seeds",        type=int, nargs="+", default=SEEDS)
    p.add_argument("--device",       default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    train_df = pd.read_csv(args.train_csv)
    val_df   = pd.read_csv(args.val_csv)
    pairs_df = pd.read_csv(args.pairs_csv) if args.pairs_csv else None

    kwargs = dict(
        arch=args.arch, num_classes=args.num_classes,
        pairs_df=pairs_df, epochs=args.epochs, patience=args.patience,
        lr=args.lr, batch_size=args.batch_size,
        image_size=args.image_size, num_workers=args.num_workers,
        device=args.device,
    )

    lambdas = args.lambda_grid if args.lambda_grid is not None else [args.lambda_view]
    all_results = {}

    for lam in lambdas:
        lam_dir = os.path.join(args.out_dir, f"lambda_{lam:.3f}")
        results = train_multi_seed(
            train_df=train_df, val_df=val_df,
            lambda_view=lam, out_dir=lam_dir,
            seeds=args.seeds, **kwargs
        )
        accs = [r["best_val_acc"] for r in results]
        all_results[lam] = {"mean": np.mean(accs), "std": np.std(accs)}

    if len(lambdas) > 1:
        print("\n=== Lambda Grid Search Summary ===")
        best_lam = max(all_results, key=lambda k: all_results[k]["mean"])
        for lam, stats in sorted(all_results.items()):
            marker = " ← BEST" if lam == best_lam else ""
            print(f"  λ={lam:.3f}:  val_acc = {stats['mean']:.4f} ± {stats['std']:.4f}{marker}")


if __name__ == "__main__":
    main()
