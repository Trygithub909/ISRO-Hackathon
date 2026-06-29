"""
Phase 7 — Image Evaluation
Metrics: PSNR, SSIM, LPIPS
Produces a CSV report + bar chart summary.
"""

import os
import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
import cv2
from tqdm import tqdm

from phase1_dataset_preparation import IRRGBDataset
from phase3_generator import ResidualUNetGenerator


# ─────────────────────────────────────────────
# PSNR
# ─────────────────────────────────────────────

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Peak Signal-to-Noise Ratio.
    pred and target: (B, C, H, W) tensors in [-1, 1].
    Returns average PSNR in dB across the batch.
    Higher is better (typical: 25–40 dB for good colourisation).
    """
    # Convert [-1,1] → [0,1]
    pred   = (pred.clamp(-1, 1) + 1) / 2
    target = (target.clamp(-1, 1) + 1) / 2

    mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
    psnr = -10 * torch.log10(mse + 1e-10)
    return psnr.mean().item()


# ─────────────────────────────────────────────
# SSIM
# ─────────────────────────────────────────────

def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.outer(g).unsqueeze(0).unsqueeze(0)


_SSIM_KERNEL = None


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Structural Similarity Index.
    Returns value in [0, 1]; closer to 1 = more similar.
    """
    global _SSIM_KERNEL
    device = pred.device

    pred   = (pred.clamp(-1, 1) + 1) / 2
    target = (target.clamp(-1, 1) + 1) / 2

    if _SSIM_KERNEL is None or _SSIM_KERNEL.device != device:
        _SSIM_KERNEL = _gaussian_kernel(11, 1.5).to(device)

    C1, C2 = 0.01**2, 0.03**2
    ssim_vals = []

    for c in range(pred.shape[1]):
        p = pred[:, c:c+1]
        t = target[:, c:c+1]

        mu_p = F.conv2d(p, _SSIM_KERNEL, padding=5)
        mu_t = F.conv2d(t, _SSIM_KERNEL, padding=5)

        mu_p2 = mu_p * mu_p
        mu_t2 = mu_t * mu_t
        mu_pt = mu_p * mu_t

        sig_p  = F.conv2d(p*p, _SSIM_KERNEL, padding=5) - mu_p2
        sig_t  = F.conv2d(t*t, _SSIM_KERNEL, padding=5) - mu_t2
        sig_pt = F.conv2d(p*t, _SSIM_KERNEL, padding=5) - mu_pt

        num = (2*mu_pt + C1) * (2*sig_pt + C2)
        den = (mu_p2 + mu_t2 + C1) * (sig_p + sig_t + C2)
        ssim_vals.append((num / den).mean().item())

    return sum(ssim_vals) / len(ssim_vals)


# ─────────────────────────────────────────────
# LPIPS (VGG-based perceptual distance)
# ─────────────────────────────────────────────

class LPIPSMetric(torch.nn.Module):
    """
    Lightweight LPIPS using VGG16 features (no external lpips package required).
    Lower = more perceptually similar.
    """

    LAYERS = [4, 9, 16, 23]   # relu1_2, relu2_2, relu3_3, relu4_3

    def __init__(self, device: str = "cpu"):
        super().__init__()
        vgg    = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        layers = list(vgg.children())

        self.slices = torch.nn.ModuleList()
        prev = 0
        for idx in self.LAYERS:
            self.slices.append(torch.nn.Sequential(*layers[prev:idx]))
            prev = idx

        for p in self.parameters():
            p.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x.clamp(-1, 1) + 1) / 2
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred   = self._preprocess(pred)
        target = self._preprocess(target)

        loss = 0.0
        for sl in self.slices:
            pred   = sl(pred)
            target = sl(target)
            # Normalise each feature map
            p_norm = pred   / (pred.norm(dim=1, keepdim=True)   + 1e-10)
            t_norm = target / (target.norm(dim=1, keepdim=True) + 1e-10)
            loss  += (p_norm - t_norm).pow(2).mean().item()

        return loss / len(self.LAYERS)


# ─────────────────────────────────────────────
# Full evaluation loop
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    manifest_path: str,
    checkpoint_path: str,
    split: str = "test",
    batch_size: int = 8,
    image_size: int = 256,
    output_csv: Optional[str] = "evaluation_results.csv",
) -> dict:
    """
    Run the generator on the test split and compute PSNR, SSIM, LPIPS.
    Returns: dict with mean ± std for each metric.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device} ...")

    # Load model
    gen = ResidualUNetGenerator(in_channels=1, out_channels=3).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(state.get("gen", state))   # handles raw state_dict too
    gen.eval()

    # Data
    ds = IRRGBDataset(manifest_path, split=split, image_size=image_size,
                      augment=False, enhance=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    lpips_fn = LPIPSMetric(device).to(device)

    psnrs, ssims, lpips_scores = [], [], []
    rows = []

    for batch in tqdm(dl, desc="Evaluating"):
        ir  = batch["ir"].to(device)
        rgb = batch["rgb"].to(device)
        fake = gen(ir)

        for i in range(ir.shape[0]):
            p = fake[i:i+1];  t = rgb[i:i+1]
            psnr_val  = compute_psnr(p, t)
            ssim_val  = compute_ssim(p, t)
            lpips_val = lpips_fn(p, t)

            psnrs.append(psnr_val)
            ssims.append(ssim_val)
            lpips_scores.append(lpips_val)
            rows.append({
                "path":  batch["path"][i],
                "psnr":  f"{psnr_val:.4f}",
                "ssim":  f"{ssim_val:.4f}",
                "lpips": f"{lpips_val:.4f}",
            })

    # Aggregates
    summary = {
        "PSNR":  {"mean": float(np.mean(psnrs)),  "std": float(np.std(psnrs))},
        "SSIM":  {"mean": float(np.mean(ssims)),  "std": float(np.std(ssims))},
        "LPIPS": {"mean": float(np.mean(lpips_scores)), "std": float(np.std(lpips_scores))},
        "n_samples": len(psnrs),
    }

    print("\n=== Evaluation Summary ===")
    for metric, vals in summary.items():
        if isinstance(vals, dict):
            print(f"  {metric:5s}  {vals['mean']:.4f} ± {vals['std']:.4f}")

    # Save CSV
    if output_csv:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "psnr", "ssim", "lpips"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPer-image results saved → {output_csv}")

    return summary


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",    required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--split",       default="test")
    parser.add_argument("--batch",       type=int,  default=8)
    parser.add_argument("--output_csv",  default="evaluation_results.csv")
    args = parser.parse_args()

    summary = evaluate(
    manifest_path=args.manifest,
    checkpoint_path=args.checkpoint,
    split=args.split,
    batch_size=args.batch,
    output_csv=args.output_csv,
    
)
    print("\n✅ Phase 7 complete.")
