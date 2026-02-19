
# COVID-CX-Net: Projection-Aware Chest X-Ray Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)

Official implementation of **"COVID-CX-Net: A Transfer Learning Approach to Detect COVID-19 Using Chest X-ray Images"** (ICCIT 2023) and **"Projection-Aware Chest X-Ray Classification: Systematic Analysis of AP/PA Projection Effects on Deep Learning Model Performance"**.

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Results](#results)
- [Projection-Aware Analysis](#projection-aware-analysis)
- [Citation](#citation)
- [Contact](#contact)

## 🎯 Overview

COVID-CX-Net is a hybrid deep learning architecture for chest X-ray (CXR) classification that addresses critical challenges in medical AI deployment:

- **High Accuracy**: Achieves ~99% accuracy on COVID-19 detection
- **Projection-Aware**: First systematic analysis of AP/PA projection bias in CXR classification
- **Clinical Robustness**: Designed for real-world deployment across diverse clinical settings

### The Problem We Solve

Standard CXR classification models achieve >95% accuracy in research but fail in clinical deployment. We demonstrate that **AP/PA projection differences create systematic biases** overlooked by standard evaluation:

- Models trained on adult PA images show **21.4% accuracy degradation** on pediatric AP images
- Grad-CAM analysis confirms models learn projection-specific features (IoU=0.73 with cardiac regions) rather than pathology patterns
- View-consistency regularization reduces performance drops from 11.5% to 6.7%

## ✨ Key Features

- **Hybrid Architecture**: DenseNet-121 backbone with VGG-16 inspired feature enhancement layers
- **Domain-Specific Pretraining**: Uses ChestX-ray14 weights instead of ImageNet for medical relevance
- **Projection-Aware Evaluation**: Comprehensive cross-dataset validation protocols
- **View-Consistency Regularization**: Mitigates projection-specific bias
- **Multi-Class Support**: COVID-19, Normal, Pneumonia, and Tuberculosis classification

## 🏗️ Architecture

```
Input (224×224 CXR Image)
    ↓
DenseNet-121 (Pretrained on ChestX-ray14)
    ↓
VGG-16 Block 1 (64 filters) ──┐
    ↓                          │
VGG-16 Block 3 (256 filters) ─┤ Extra Trainable Layers
    ↓                          │
Global Average Pooling (GAP)  ←┘
    ↓
Fully Connected + Softmax
    ↓
Output (3-4 Classes)
```

**Total Parameters**: ~8.99M (2M trainable, 7M frozen)

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/Akif-Mahdi/COVID-CX-Net.git
cd COVID-CX-Net

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- TensorFlow 2.x
- Keras
- NumPy, Pandas, Matplotlib
- OpenCV
- scikit-learn
- Grad-CAM visualization tools

## 💻 Usage

### Training

```python
from covid_cx_net import COVIDCXNet

# Initialize model with view-consistency regularization
model = COVIDCXNet(
    input_shape=(224, 224, 3),
    num_classes=3,
    use_view_consistency=True,
    lambda_view=0.1
)

# Train with projection-aware data augmentation
model.train(
    train_dir='data/train',
    val_dir='data/val',
    epochs=100,
    batch_size=32,
    optimizer='adam',
    learning_rate=1e-4
)
```

### Inference

```python
# Load pretrained weights
model.load_weights('weights/covid_cx_net_best.h5')

# Predict with confidence scores
prediction, confidence = model.predict(image_path)
```

### Projection-Aware Evaluation

```python
# Cross-dataset evaluation
results = model.evaluate_projection_aware(
    test_datasets=['Dataset_A', 'Dataset_B', 'Dataset_C'],
    projection_labels=['AP', 'PA', 'Mixed']
)

# Generate View Bias Score (VBS)
vbs = calculate_vbs(results)
print(f"View Bias Score: {vbs:.2f}%")
```

## 📊 Datasets

The model supports multiple publicly available datasets:

| Dataset | Classes | Images | Projection | Primary Source |
|---------|---------|--------|------------|----------------|
| **Dataset A** | 3 | 10,500 | 62% AP / 38% PA | COVID-19 Radiography DB, TB CXR DB, CheXpert |
| **Dataset B** | 3 | 1,800 | 15% AP / 85% PA | COVID-19 Radiography DB, TB CXR DB, CheXpert |
| **Dataset C** | 3 | 1,500 | 92% AP / 8% PA | Pediatric-predominant |
| **Dataset D** | 2 | 500 | 55% AP / 45% PA | Balanced |
| **Dataset E** | 4 | 12,000 | 48% AP / 52% PA | Mixed adult |

### Data Sources
- [COVID-19 Radiography Database (Kaggle)](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- [NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)
- [TB Chest X-ray Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)

**For dataset access issues or research collaborations, contact: akif2100@gmail.com**

## 📈 Results

### Within-Dataset Performance

| Model | Dataset A (10,500) | Dataset B (1,800) | Dataset C (12,000) |
|-------|-------------------|-------------------|-------------------|
| VGG-16 | 97.00% | 90.67% | - |
| ResNet50 | 95.00% | 94.00% | - |
| DenseNet-121 | 96.67% | 97.25% | - |
| **COVID-CX-Net (Ours)** | **99.67%** | **99.00%** | **98.06%** |

### Cross-Dataset Generalization (Projection-Aware)

| Training → Testing | Accuracy | Performance Drop | VBS |
|-------------------|----------|------------------|-----|
| B (PA) → C (AP) | 78.3% | **21.4%** ⚠️ | High |
| C (AP) → B (PA) | 84.7% | 14.9% | High |
| C (AP) → A (Mixed) | 87.1% | 11.6% | Moderate |
| B (PA) → A (Mixed) | 91.2% | 7.5% | Moderate |

### View-Consistency Regularization Impact

| λ Value | Within-Dataset | Cross-Dataset | Performance Drop |
|---------|---------------|---------------|------------------|
| 0.00 (Baseline) | 98.76% | 87.3% | 11.5% |
| **0.10 (Optimal)** | **98.41%** | **91.7%** | **6.7%** ✓ |
| 0.30 | 97.85% | 90.8% | 7.0% |

## 🔍 Projection-Aware Analysis

### Key Findings

1. **Systematic Projection Bias**: Models exhibit significant performance degradation (mean 11.5%) across projection views
2. **Learnable Bias**: Grad-CAM shows models learn projection-specific features (IoU=0.73 with cardiac regions for AP-trained models)
3. **Demographic Interaction**: 26.8% VBS for adult PA → pediatric AP transitions
4. **Partial Mitigation**: View-consistency regularization reduces bias but doesn't eliminate it

### Clinical Deployment Recommendations

✅ **DO**:
- Validate on local data with local projection distribution
- Monitor performance separately for AP and PA images
- Implement fallback protocols for low-confidence predictions
- Use projection-stratified test sets

❌ **DON'T**:
- Assume high benchmark accuracy guarantees clinical reliability
- Deploy without cross-projection evaluation
- Ignore pediatric vs. adult demographic differences

## 📚 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@inproceedings{mahdi2023covid,
  title={COVID-CX-Net: A Transfer Learning Approach to Detect COVID-19 Using Chest X-ray Images},
  author={Mahdi, Akif and Kabir, M. Hasnat},
  booktitle={2023 26th International Conference on Computer and Information Technology (ICCIT)},
  pages={1--6},
  year={2023},
  organization={IEEE},
  doi={979-8-3503-5901-5/23/$31.00}
}

```

## 📧 Contact

**For questions about the code, dataset access, or research collaboration:**

📩 **Akif Mahdi**: akif2100@gmail.com

**Affiliations:**
- Department of Information and Communication Engineering, University of Rajshahi, Rajshahi-6205, Bangladesh

## 🙏 Acknowledgments

- [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) team
- [NIH ChestX-ray14](https://www.nih.gov/) dataset creators
- [CheXpert](https://stanfordmlgroup.github.io/) team at Stanford
- Medical professionals from Hamad Medical Corporation, Qatar and Bangladesh

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**⚠️ Medical Disclaimer**: This software is intended for research purposes only and should not be used as a primary diagnostic tool without appropriate clinical validation and regulatory approval.
```

