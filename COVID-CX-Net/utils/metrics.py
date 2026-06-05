"""
utils/metrics.py
================
View Bias Score (VBS), IoU, bootstrap CI, and related metrics used throughout
the paper "Projection-Induced Domain Shift in Chest X-Ray Classification."
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Sequence, Tuple, Union
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix


# ---------------------------------------------------------------------------
# View Bias Score
# ---------------------------------------------------------------------------

def compute_vbs(
    acc_same: float,
    acc_cross: float,
) -> float:
    """
    VBS = Acc_{P_same} − Acc_{P_cross}

    Args:
        acc_same:   Mean accuracy on same-projection evaluation pairs.
        acc_cross:  Mean accuracy on cross-projection evaluation pairs.

    Returns:
        VBS as a float (0–1 range; multiply by 100 for percentage).
    """
    return float(acc_same - acc_cross)


def compute_vbs_within(
    y_true_ap: np.ndarray,
    y_pred_ap: np.ndarray,
    y_true_pa: np.ndarray,
    y_pred_pa: np.ndarray,
) -> float:
    """
    VBS_within = Acc(f_AP, T_AP) − Acc(f_AP, T_PA)

    Measures how much accuracy drops when the same model is evaluated on
    the opposite projection distribution.

    Args:
        y_true_ap: Ground-truth labels for AP test subset.
        y_pred_ap: Model predictions for AP test subset.
        y_true_pa: Ground-truth labels for PA test subset.
        y_pred_pa: Model predictions for PA test subset.

    Returns:
        VBS_within float.
    """
    acc_ap = accuracy_score(y_true_ap, y_pred_ap)
    acc_pa = accuracy_score(y_true_pa, y_pred_pa)
    return float(acc_ap - acc_pa)


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_iter: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    stratify_by: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for any scalar metric.

    Args:
        y_true:       Ground-truth labels, shape (N,).
        y_pred:       Predictions, shape (N,) or (N, C).
        metric_fn:    Callable(y_true, y_pred) → float.
        n_iter:       Number of bootstrap iterations (default 1000).
        ci:           Confidence level (default 0.95).
        seed:         Random seed for reproducibility.
        stratify_by:  Optional array used for stratified resampling.
                      Must have the same length as y_true.
                      Stratification is performed jointly by pathology and
                      projection type (see paper Section 3.7).

    Returns:
        (mean, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []

    for _ in range(n_iter):
        if stratify_by is not None:
            # Stratified bootstrap: sample within each stratum
            idx = []
            for stratum in np.unique(stratify_by):
                mask = stratify_by == stratum
                stratum_idx = np.where(mask)[0]
                idx.append(rng.choice(stratum_idx, size=len(stratum_idx), replace=True))
            idx = np.concatenate(idx)
        else:
            idx = rng.choice(n, size=n, replace=True)

        try:
            score = metric_fn(y_true[idx], y_pred[idx] if y_pred.ndim == 1 else y_pred[idx])
            scores.append(score)
        except Exception:
            continue

    scores = np.array(scores)
    alpha = (1.0 - ci) / 2.0
    return float(scores.mean()), float(np.quantile(scores, alpha)), float(np.quantile(scores, 1 - alpha))


# ---------------------------------------------------------------------------
# IoU for Grad-CAM masks
# ---------------------------------------------------------------------------

def threshold_cam(cam: np.ndarray, tau: float = 0.5) -> np.ndarray:
    """
    Binarise a Grad-CAM heatmap at threshold τ (default 0.5, per paper Section 3.6).

    Args:
        cam: float32 array (H, W), values in [0, 1].
        tau: Binarisation threshold.

    Returns:
        Boolean mask (H, W).
    """
    if cam.min() < 0 or cam.max() > 1:
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam >= tau


def compute_iou(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> float:
    """
    Intersection-over-Union between two binary masks.
    Returns 0.0 when union is empty.
    """
    if mask_a.shape != mask_b.shape:
        from PIL import Image
        mb = Image.fromarray(mask_b.astype(np.uint8) * 255).resize(
            (mask_a.shape[1], mask_a.shape[0]), Image.NEAREST
        )
        mask_b = np.array(mb) > 127
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union > 0 else 0.0


def cardiac_region_mask(shape: tuple) -> np.ndarray:
    """
    Approximate anatomical cardiac region mask for a frontal CXR.
    Cardiac ROI: y ∈ [35%, 75%] × x ∈ [25%, 75%] (paper Section 3.6).
    """
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    mask[int(0.35*H):int(0.75*H), int(0.25*W):int(0.75*W)] = True
    return mask


def mediastinal_region_mask(shape: tuple) -> np.ndarray:
    """
    Approximate mediastinal region mask for a frontal CXR.
    Mediastinal ROI: y ∈ [15%, 80%] × x ∈ [35%, 65%] (paper Section 3.6).
    """
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    mask[int(0.15*H):int(0.80*H), int(0.35*W):int(0.65*W)] = True
    return mask


def compute_region_iou(
    cam: np.ndarray,
    region: str = "cardiac",
    tau: float = 0.5,
) -> float:
    """
    IoU between a thresholded Grad-CAM mask and an anatomical region mask.

    Args:
        cam:    float32 heatmap (H, W).
        region: "cardiac" or "mediastinal".
        tau:    Binarisation threshold (default 0.5).

    Returns:
        IoU float.
    """
    cam_mask = threshold_cam(cam, tau)
    if region == "cardiac":
        region_mask = cardiac_region_mask(cam.shape)
    elif region == "mediastinal":
        region_mask = mediastinal_region_mask(cam.shape)
    else:
        raise ValueError(f"Unknown region '{region}'. Choose: cardiac | mediastinal")
    return compute_iou(cam_mask, region_mask)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = "macro",
) -> dict:
    """
    Compute accuracy, macro-F1, and optionally macro-AUC.

    Args:
        y_true: Ground-truth integer labels (N,).
        y_pred: Predicted integer labels (N,).
        y_prob: Predicted probabilities (N, C) for AUC computation.
        average: Averaging strategy for multi-class ('macro' per paper).

    Returns:
        dict with keys: accuracy, macro_f1, auc (if y_prob provided).
    """
    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }
    if y_prob is not None:
        n_classes = y_prob.shape[1]
        try:
            if n_classes == 2:
                results["auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                results["auc"] = float(roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average=average
                ))
        except ValueError:
            results["auc"] = float("nan")
    return results
