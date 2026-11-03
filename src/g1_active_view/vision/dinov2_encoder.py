"""Offline DINOv2 ViT-B/14 encoder used by the retained policy.

This module never uses torch.hub. Both the source tree and checkpoint are explicit
paths and missing files fail before any model is initialized.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn

FEATURE_DIM = 768 * 17

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]

def image_tensor(image):
    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            im = im.convert("RGB").resize((280, 210), Image.Resampling.BILINEAR)
            arr = np.asarray(im).copy()
    else:
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError(f"expected HxWx3 image, got {arr.shape}")
        arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.max(initial=0) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        im = Image.fromarray(arr, mode="RGB").resize((280, 210), Image.Resampling.BILINEAR)
        arr = np.asarray(im).copy()
    x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    return (x - mean) / std

class DINOv2ViTB14(nn.Module):
    def __init__(self, source_dir: Path, weights_path: Path):
        super().__init__()
        source_dir = Path(source_dir).resolve(); weights_path = Path(weights_path).resolve()
        if not source_dir.exists(): raise FileNotFoundError(f"DINOv2 source missing: {source_dir}")
        if not weights_path.exists(): raise FileNotFoundError(f"DINOv2 weights missing: {weights_path}")
        sys.path.insert(0, str(source_dir.parent))
        from dinov2.hub.backbones import dinov2_vitb14
        self.net = dinov2_vitb14(pretrained=False)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.net.load_state_dict(state)
        self.requires_grad_(False); self.eval()

    @torch.inference_mode()
    def forward(self, x):
        d = self.net.forward_features(x)
        p = d['x_norm_patchtokens'].transpose(1, 2).reshape(len(x), 768, x.shape[-2] // 14, x.shape[-1] // 14)
        return torch.cat([d['x_norm_clstoken'], torch.nn.functional.adaptive_avg_pool2d(p, (4, 4)).flatten(1)], 1)
