"""Basic smoke tests — run with: pytest tests/"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch


def test_model_builds():
    from models import build_model
    for arch in ["covidcxnet", "vgg16", "resnet50", "densenet121"]:
        m = build_model(arch, num_classes=3, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = m(x)
        assert out.shape == (2, 3), f"{arch} wrong output shape: {out.shape}"
    print("All architectures: OK")


def test_vbs():
    from utils.metrics import compute_vbs
    result = compute_vbs(0.90, 0.69)
    assert abs(result - 0.21) < 1e-6


def test_iou():
    from utils.metrics import compute_iou, threshold_cam
    h = np.ones((224, 224), dtype=np.float32)
    mask = threshold_cam(h, tau=0.5)
    assert compute_iou(mask, mask) == 1.0


def test_view_consistency_loss():
    from utils.view_consistency import ViewConsistencyLoss
    loss_fn = ViewConsistencyLoss()
    a = torch.randn(4, 128)
    assert loss_fn(a, a).item() < 1e-6   # identical embeds → zero loss
    assert loss_fn(a, torch.randn(4, 128)).item() >= 0


def test_projection_aware_loss():
    from utils.view_consistency import ProjectionAwareLoss
    fn = ProjectionAwareLoss(lambda_view=0.1, num_classes=3)
    logits = torch.randn(4, 3)
    labels = torch.randint(0, 3, (4,))
    loss, comps = fn(logits, labels)
    assert loss.item() > 0
    assert "ce" in comps and "total" in comps


if __name__ == "__main__":
    test_model_builds()
    test_vbs()
    test_iou()
    test_view_consistency_loss()
    test_projection_aware_loss()
    print("\nAll smoke tests passed.")
