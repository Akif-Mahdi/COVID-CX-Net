"""
models/covid_cx_net.py
======================
COVID-CX-Net: DenseNet-121 backbone augmented with the 1st and 3rd VGG-16
convolutional blocks for low- and mid-level feature enhancement.

Reference:
    Mahdi, A., Kabir, M.H.: COVID-CX-Net: A transfer learning approach to
    detect COVID-19 using chest X-ray images. Proc. ICCIT 2023, pp. 1-6.

Architecture (8.99M parameters, ~2M trainable with standard frozen backbone):
    - VGG-16 Block 1  (2x Conv 3x3, 64 filters)       → low-level edges
    - VGG-16 Block 3  (3x Conv 3x3, 256 filters)      → mid-level textures
    - Feature fusion  (concatenation + 1x1 projection)
    - DenseNet-121    (domain-pretrained on ChestX-ray14)
    - Global Average Pooling + Linear classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional


# ---------------------------------------------------------------------------
# VGG-16 feature blocks
# ---------------------------------------------------------------------------

class VGGBlock(nn.Module):
    """A single VGG-style convolutional block (N conv layers + max-pool)."""

    def __init__(self, in_channels: int, out_channels: int, n_convs: int = 2):
        super().__init__()
        layers = []
        for i in range(n_convs):
            c_in = in_channels if i == 0 else out_channels
            layers += [
                nn.Conv2d(c_in, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# COVID-CX-Net
# ---------------------------------------------------------------------------

class CovidCXNet(nn.Module):
    """
    COVID-CX-Net for chest X-ray classification.

    Args:
        num_classes:       Number of output classes (default 3: COVID/Normal/Pneumonia).
        pretrained_densenet: Load ImageNet-pretrained DenseNet-121 weights.
        freeze_densenet:   Freeze DenseNet-121 feature extractor weights.
        dropout_rate:      Dropout probability before final classifier.
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained_densenet: bool = True,
        freeze_densenet: bool = False,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes

        # ── VGG-16 Block 1: 3 → 64, 2 convs, pool to H/2 × W/2 ──────────
        self.vgg_block1 = VGGBlock(in_channels=3, out_channels=64, n_convs=2)

        # ── VGG-16 Block 3: 64 → 256, 3 convs, pool to H/8 × W/8 ────────
        # (blocks 1+2 already applied above; block 3 takes 64-ch input here
        #  after an extra pool to match spatial scale)
        self.vgg_pool2   = nn.MaxPool2d(kernel_size=2, stride=2)   # → H/4
        self.vgg_block3  = VGGBlock(in_channels=64, out_channels=256, n_convs=3)

        # ── DenseNet-121 backbone ─────────────────────────────────────────
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained_densenet else None
        densenet = models.densenet121(weights=weights)
        self.densenet_features = densenet.features  # output: (B, 1024, H/32, W/32)

        if freeze_densenet:
            for p in self.densenet_features.parameters():
                p.requires_grad = False

        # ── Fusion: project VGG3 features to match DenseNet spatial scale ─
        # VGG block1+pool2+block3 → H/8; we need H/32.
        # Apply two extra 2x2 max-pools (×4 spatial reduction) + 1×1 conv.
        self.fusion_pool   = nn.Sequential(
            nn.MaxPool2d(2, 2),   # H/16
            nn.MaxPool2d(2, 2),   # H/32
        )
        self.fusion_proj   = nn.Conv2d(256, 128, kernel_size=1)
        self.fusion_bn     = nn.BatchNorm2d(128)

        # ── Classifier head ───────────────────────────────────────────────
        # DenseNet output: 1024; fused VGG: 128; total: 1152
        self.gap       = nn.AdaptiveAvgPool2d(1)
        self.dropout   = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(1024 + 128, num_classes)

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────
    def _init_weights(self):
        for m in [self.vgg_block1, self.vgg_pool2, self.vgg_block3,
                  self.fusion_proj, self.fusion_bn, self.classifier]:
            for layer in (m.modules() if hasattr(m, 'modules') else [m]):
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # VGG branch
        v = self.vgg_block1(x)          # (B, 64,  H/2,  W/2)
        v = self.vgg_pool2(v)            # (B, 64,  H/4,  W/4)
        v = self.vgg_block3(v)           # (B, 256, H/8,  W/8)
        v = self.fusion_pool(v)          # (B, 256, H/32, W/32)
        v = F.relu(self.fusion_bn(self.fusion_proj(v)))  # (B, 128, H/32, W/32)
        v = self.gap(v).flatten(1)       # (B, 128)

        # DenseNet branch
        d = self.densenet_features(x)    # (B, 1024, H/32, W/32)
        d = F.relu(d, inplace=True)
        d = self.gap(d).flatten(1)       # (B, 1024)

        # Fusion + classify
        feat = torch.cat([d, v], dim=1)  # (B, 1152)
        feat = self.dropout(feat)
        return self.classifier(feat)

    # ── Embedding (for view-consistency regularization) ───────────────────
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pre-classifier embedding vector (B, 1152)."""
        with torch.no_grad():
            v = self.vgg_block1(x)
            v = self.vgg_pool2(v)
            v = self.vgg_block3(v)
            v = self.fusion_pool(v)
            v = F.relu(self.fusion_bn(self.fusion_proj(v)))
            v = self.gap(v).flatten(1)
            d = self.densenet_features(x)
            d = F.relu(d, inplace=True)
            d = self.gap(d).flatten(1)
        return torch.cat([d, v], dim=1)

    # ── Grad-CAM target layer ─────────────────────────────────────────────
    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Last convolutional layer for Grad-CAM targeting."""
        return self.densenet_features.norm5


# ---------------------------------------------------------------------------
# CheXNet wrapper (DenseNet-121, 14-class NIH ChestX-ray14 pretrained)
# ---------------------------------------------------------------------------

class CheXNet(nn.Module):
    """
    CheXNet: DenseNet-121 pretrained on NIH ChestX-ray14 (14-class multi-label).
    Matches the key prefix 'densenet121.*' used in the pretrained checkpoint.

    Reference:
        Rajpurkar, P., et al.: CheXNet: Radiologist-level pneumonia detection
        on chest X-rays with deep learning. arXiv:1711.05225 (2017).
    """

    NIH_CLASSES = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
        "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
        "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
    ]

    def __init__(self, num_classes: int = 14):
        super().__init__()
        self.densenet121 = models.densenet121(weights=None)
        self.densenet121.classifier = nn.Sequential(nn.Linear(1024, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.densenet121(x)

    @property
    def gradcam_target_layer(self) -> nn.Module:
        return self.densenet121.features.norm5


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_model(
    arch: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Build a model by architecture name.

    Args:
        arch:            One of: covidcxnet, vgg16, resnet50, densenet121, chexnet
        num_classes:     Output class count
        pretrained:      Use ImageNet pretrained weights for backbone
        freeze_backbone: Freeze feature extractor weights

    Returns:
        Configured nn.Module
    """
    arch = arch.lower()
    weights_flag = 'IMAGENET1K_V1' if pretrained else None

    if arch == "covidcxnet":
        return CovidCXNet(num_classes=num_classes,
                          pretrained_densenet=pretrained,
                          freeze_densenet=freeze_backbone)

    elif arch == "vgg16":
        w = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vgg16(weights=w)
        if freeze_backbone:
            for p in model.features.parameters():
                p.requires_grad = False
        model.classifier[6] = nn.Linear(4096, num_classes)
        return model

    elif arch == "resnet50":
        w = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=w)
        if freeze_backbone:
            for name, p in model.named_parameters():
                if "fc" not in name:
                    p.requires_grad = False
        model.fc = nn.Linear(2048, num_classes)
        return model

    elif arch == "densenet121":
        w = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=w)
        if freeze_backbone:
            for name, p in model.named_parameters():
                if "classifier" not in name:
                    p.requires_grad = False
        model.classifier = nn.Linear(1024, num_classes)
        return model

    elif arch == "chexnet":
        return CheXNet(num_classes=num_classes)

    else:
        raise ValueError(
            f"Unknown architecture: '{arch}'. "
            "Choose: covidcxnet | vgg16 | resnet50 | densenet121 | chexnet"
        )
