# Grad-CAM Pipeline

Run `../05_gradcam.py` to generate Figures 3, 4, 5. See main README for full usage.

```bash
python ../05_gradcam.py --mode figures \
    --chexnet_ckpt  /path/to/chexnet.pth \
    --covid_img     /path/to/COVID-197.png \
    --pneumo_img    /path/to/Pneumonia-190.jpeg \
    --cardio_img    /path/to/Pneumonia-146.jpeg \
    --out_dir       ./
```
