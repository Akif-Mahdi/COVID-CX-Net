# COVID-CX-Net: Projection-Induced Domain Shift in Chest X-Ray Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9](https://img.shields.io/badge/Python-3.9.18-blue.svg)](https://www.python.org/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1.0-orange.svg)](https://pytorch.org/)
[![BSPC](https://img.shields.io/badge/Journal-BSPC-green.svg)](https://www.sciencedirect.com/journal/biomedical-signal-processing-and-control)

**Official implementation of:**

> **Projection-Induced Domain Shift in Chest X-Ray Classification: A Projection-Aware Evaluation and Regularization Framework**  
> Akif Mahdi, Jarin Alam Prity, M. Hasnat Kabir, Md. Ibne Shihab Shad, Ahmad Wasim Wardak, Nafees Nusrat Eysha  
> *Biomedical Signal Processing and Control (BSPC)*, 2026

---

## Overview

This repository demonstrates that AP/PA radiographic projection is a systematic and previously unquantified source of domain shift in chest X-ray AI. A model trained on outpatient PA films and deployed in an ICU (90% AP acquisitions) can suffer **>21% accuracy degradation** — enough to misclassify critically ill patients.

We introduce:
- **View Bias Score (VBS)** — the first computable pre-deployment screening metric for projection mismatch risk
- **View-Consistency Regularization** — reduces cross-dataset VBS from 11.5% → 6.7% with zero inference overhead
- **Projection-Aware Evaluation Protocol** — validated on 5 composite datasets + MIMIC-CXR (N≈112,000)

![Graphical Abstract](figures/graphical_abstract.png)

---

## Key Results

| Metric | Value |
|---|---|
| Max accuracy drop (PA→AP transfer) | **21.4%** |
| Adult-PA → Pediatric-AP VBS | **26.8%** |
| VBS reduced by view-consistency reg. | **11.5% → 6.7%** |
| MIMIC-CXR VBS_within (DICOM ground-truth) | **2.6%** |
| AP-trained cardiac attention (IoU) | **0.73 ± 0.08** |
| PA-trained mediastinal attention (IoU) | **0.68 ± 0.12** |

---

## Repository Structure

```
COVID-CX-Net/
├── 01_data_loading.py        # Dataset assembly, MIMIC-CXR DICOM extraction, AP/PA pairs
├── 02_model_training.py      # Training loop with view-consistency regularization + λ grid search
├── 03_evaluation.py          # Projection-stratified evaluation, VBS computation
├── 04_statistical_testing.py # McNemar's test, Bonferroni correction, bootstrap CI
├── 05_gradcam.py             # Grad-CAM generation, IoU quantification, figure export
│
├── models/
│   ├── __init__.py
│   └── covid_cx_net.py       # COVID-CX-Net, CheXNet, and baseline architectures
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py            # VBS, IoU, bootstrap CI, classification metrics
│   ├── datasets.py           # CXRDataset, PairedCXRDataset, MIMIC loader, transforms
│   └── view_consistency.py   # ViewConsistencyLoss, ProjectionAwareLoss
│
├── gradcam_pipeline/
│   ├── generate_figures.py   # Standalone figure generation script
│   ├── compute_iou.py        # Standalone IoU computation
│   └── README.md             # Pipeline usage guide
│
├── configs/
│   └── default_config.yaml   # All hyperparameters with paper-matched defaults
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Akif-Mahdi/COVID-CX-Net.git
cd COVID-CX-Net
pip install -r requirements.txt
```

> **GPU requirement:** NVIDIA GPU with ≥8 GB VRAM recommended.  
> Paper experiments used NVIDIA RTX A6000 (48 GB VRAM), CUDA 11.8.

---

## Datasets

### Composite Datasets A–E

| ID | Classes | Images | Projection | Sources |
|----|---------|--------|------------|---------|
| A | 3 | 10,500 | 62% AP / 38% PA | COVID-19 DB; TB; CheXpert |
| B | 3 | 1,800 | 15% AP / **85% PA** | COVID-19 DB; TB; CheXpert |
| C | 3 | 1,500 | **92% AP** / 8% PA | COVID CXR; RSNA |
| D | 2 | 500 | 55% AP / 45% PA | COVID CXR; RSNA |
| E | 4 | 12,000 | 48% AP / 52% PA | COVID-19 DB; TB; CheXpert; NIH |

**Data sources:**
- [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) (Kaggle)
- [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Kaggle)
- [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)

### MIMIC-CXR

Available via PhysioNet (credentialed access required):  
[https://physionet.org/content/mimic-cxr/2.0.0/](https://physionet.org/content/mimic-cxr/2.0.0/)

Version used: **mimic-cxr-2.0.0** (downloaded 2024-11-15)

---

## Quick Start

### 1. Prepare data manifests

```bash
# Build manifest from a class-labelled directory tree
python 01_data_loading.py --mode build \
    --data_dir /path/to/dataset_b \
    --source_name covid19_db \
    --output_dir data/manifests/dataset_b

# Extract MIMIC-CXR projection labels from DICOM metadata
python 01_data_loading.py --mode mimic \
    --mimic_root     /path/to/mimic-cxr-jpg \
    --metadata_csv   /path/to/mimic-cxr-2.0.0-metadata.csv \
    --labels_csv     /path/to/mimic-cxr-2.0.0-chexpert.csv \
    --output_dir     data/manifests/mimic

# Match AP/PA pairs for view-consistency training
python 01_data_loading.py --mode pairs \
    --manifest   data/manifests/mimic/train.csv \
    --output_dir data/manifests/mimic
```

### 2. Train with view-consistency regularization

```bash
# Optimal configuration (λ=0.1, Table 4 of paper):
python 02_model_training.py \
    --train_csv   data/manifests/dataset_a/train.csv \
    --val_csv     data/manifests/dataset_a/val.csv \
    --pairs_csv   data/manifests/dataset_a/ap_pa_pairs.csv \
    --arch        covidcxnet \
    --num_classes 3 \
    --lambda_view 0.1 \
    --out_dir     checkpoints/dataset_a \
    --seeds       0 1 2 3 4

# Run full λ grid search (replicates Table 4):
python 02_model_training.py \
    --train_csv   data/manifests/dataset_a/train.csv \
    --val_csv     data/manifests/dataset_a/val.csv \
    --arch        covidcxnet \
    --lambda_grid 0.0 0.01 0.05 0.10 0.15 0.20 0.30 \
    --out_dir     checkpoints/gridsearch
```

### 3. Evaluate (VBS computation)

```bash
# Within-dataset projection-stratified evaluation (Table 8):
python 03_evaluation.py \
    --checkpoint checkpoints/dataset_a/lambda_0.100/seed_0/best_seed0.pth \
    --test_csv   data/manifests/dataset_a/test.csv \
    --mode       within

# Cross-dataset transfer B→C (Table 3, worst case 21.4% drop):
python 03_evaluation.py \
    --checkpoint checkpoints/dataset_b/lambda_0.100/seed_0/best_seed0.pth \
    --test_csv   data/manifests/dataset_c/test.csv \
    --mode       cross \
    --train_name "B (85% PA)" \
    --test_name  "C (92% AP)"

# Per-pathology VBS on MIMIC-CXR (Table 10):
python 03_evaluation.py \
    --checkpoint checkpoints/mimic/best_seed0.pth \
    --test_csv   data/manifests/mimic/test.csv \
    --mode       mimic_pathology
```

### 4. Statistical tests

```bash
# McNemar's test (paired comparison of two models):
python 04_statistical_testing.py --mode pair \
    --pred_a       results/preds_baseline.csv \
    --pred_b       results/preds_view_consistency.csv \
    --n_comparisons 4 \
    --label         "Baseline vs View-Consistency"

# VBS sensitivity analysis (Supplementary Figure 1):
python 04_statistical_testing.py --mode sensitivity \
    --pred_a results/preds_cross_BC.csv \
    --out_dir results/stats
```

### 5. Generate Grad-CAM figures

```bash
# Generate Figures 3, 4, 5 (paper submission format, 300 DPI):
python 05_gradcam.py --mode figures \
    --chexnet_ckpt checkpoints/chexnet_pretrained.pth \
    --covid_img    data/COVID-19_Radiography/COVID/images/COVID-197.png \
    --pneumo_img   data/chest_xray/PNEUMONIA/Pneumonia-190.jpeg \
    --cardio_img   data/chest_xray/PNEUMONIA/Pneumonia-146.jpeg \
    --out_dir      figures/

# Compute IoU over 50 matched AP/PA pairs (paper Section 3.6):
python 05_gradcam.py --mode iou \
    --checkpoint checkpoints/ap_trained/best_seed0.pth \
    --pairs_csv  data/manifests/mimic/ap_pa_pairs.csv \
    --arch       covidcxnet \
    --class_idx  1 \
    --region     cardiac \
    --n_pairs    50
```

---

## Model Architecture

**COVID-CX-Net** (8.99M parameters, ~2M trainable):

```
Input (224×224×3)
    ├── VGG-16 Block 1  (64 filters, 2 conv)  → low-level edges/textures
    │       ↓ MaxPool + VGG Block 3 (256 filters, 3 conv)
    │       ↓ 2× MaxPool + 1×1 projection → (B, 128, 7, 7)
    │       ↓ GAP → (B, 128)
    │
    └── DenseNet-121 features  (ChestX-ray14 pretrained)
            ↓ ReLU + GAP → (B, 1024)

Fusion: concat → (B, 1152) → Dropout(0.3) → Linear(num_classes)
```

**Baseline architectures** supported: `vgg16`, `resnet50`, `densenet121`, `chexnet`

---

## View-Consistency Regularization

The view-consistency loss (Equation 2 in the paper) penalises differences between
normalised embedding vectors of matched AP/PA image pairs:

```
L_view = (1/|P|) * Σ ‖ h(x_AP)/‖h‖ − h(x_PA)/‖h‖ ‖²₂

L_total = L_CE + λ * L_view    (λ = 0.1 optimal, Table 4)
```

This is implemented in [`utils/view_consistency.py`](utils/view_consistency.py).

**Effect:** Reduces cross-dataset VBS from **11.5% → 6.7%** (p < 0.001, McNemar's test,
Bonferroni-corrected α_adj = 0.0125) with no inference overhead.

---

## VBS Threshold Guidance

| VBS | Interpretation | Action |
|-----|---------------|--------|
| < 5% | Acceptable | Deploy with standard monitoring |
| 5–10% | Elevated | Projection-stratified monitoring required |
| > 10% | Clinically unacceptable | Do not deploy without retraining |

---

## Reproducibility

| Setting | Value |
|---------|-------|
| Seeds (composite datasets) | 0–4 (5 runs) |
| Seeds (MIMIC-CXR) | 0–9 (10 runs) |
| Horizontal flipping | **Disabled** (preserves anatomical orientation) |
| Patient-wise split | Enforced via `subject_id` stratification |
| Hardware | NVIDIA RTX A6000, 48 GB VRAM |
| CUDA | 11.8 |
| PyTorch | 2.1.0+cu118 |
| Python | 3.9.18 |
| MIMIC-CXR version | 2.0.0, downloaded 2024-11-15 |

All reported confidence intervals are **95% bootstrap CIs, 1,000 stratified
resampling iterations** (stratified jointly by pathology class and projection type).

---

## Statistical Testing

All pairwise comparisons use **McNemar's test with continuity correction**,
Bonferroni-corrected at:
- α_adj = 0.0125 (4 composite-dataset comparisons)
- α_adj = 0.0083 (6 MIMIC-CXR comparisons)

Significance notation: \*\*\* p < 0.001, \*\* p < 0.01, \* p < 0.05

---

## Citation

If you use this code or the VBS metric in your work, please cite:

```bibtex
@article{mahdi2026projection,
  title     = {Projection-Induced Domain Shift in Chest X-Ray Classification:
               A Projection-Aware Evaluation and Regularization Framework},
  author    = {Mahdi, Akif and Prity, Jarin Alam and Kabir, M. Hasnat and
               Shad, Md. Ibne Shihab and Nahin, Sahriar Nur and
               Wardak, Ahmad Wasim and Eysha, Nafees Nusrat},
  journal   = {Biomedical Signal Processing and Control},
  year      = {2026},
  publisher = {Elsevier}
}
```

---

## Authors

| Name | Affiliation | Email |
|------|-------------|-------|
| **Akif Mahdi** (Corresponding) | Dept. of ICE, Pabna University of Science and Technology, Bangladesh | akif2100@gmail.com |
| Jarin Alam Prity | Dept. of CSE, Metropolitan University, Sylhet, Bangladesh | jarinprity438@gmail.com |
| M. Hasnat Kabir | Dept. of ICE, University of Rajshahi, Bangladesh | hasnatkabir11@gmail.com |
| Md. Ibne Shihab Shad | Dept. of ICE, Pabna University of Science and Technology, Bangladesh | ibn.shihab17@gmail.com |
| Ahmad Wasim Wardak | Dept. of CS, University of California, Irvine, CA, USA | ahwwardak@gmail.com |
| Nafees Nusrat Eysha | Dept. of Obstetrics & Gynecology, Rajshahi Medical College and Hospital, Bangladesh | nafees.nusrat.nn@gmail.com |

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

**Patient data:** All CXR images used in this study are from publicly available, 
de-identified datasets. No identifiable patient data is included in this repository.
MIMIC-CXR data requires PhysioNet credentialed access per the Data Use Agreement.

