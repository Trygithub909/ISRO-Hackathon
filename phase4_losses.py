"""
Phase 4 — Multi-Loss Training Functions
Losses: GAN + L1 + Perceptual (VGG19) + SSIM + Edge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────
# Loss weights config
# ─────────────────────────────────────────────

@dataclass
class LossWeights:
    gan:        float = 1.0
    l1:         float = 100.0
    perceptual: float = 10.0
    ssim:       float = 10.0
    edge:       float = 5.0


# ─────────────────────────────────────────────
# 1. GAN Losses (Least-Squares GAN = more stable)
# ─────────────────────────────────────────────

class GANLoss(nn.Module):
    """
    LSGAN loss (MSE-based).
    real_label = 1,  fake_label = 0
    Discriminator: max  E[(D(real)-1)^2] + E[D(fake)^2]
    Generator:     min  E[(D(fake)-1)^2]
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def _tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
        return self.loss(prediction, target)

    def discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
    ) -> torch.Tensor:
        return 0.5 * (self._tensor(real_pred, True) + self._tensor(fake_pred, False))

    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        return self._tensor(fake_pred, True)


# ─────────────────────────────────────────────
# 2. L1 Loss
# ─────────────────────────────────────────────

class L1Loss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


# ─────────────────────────────────────────────
# 3. Perceptual Loss (VGG19 feature matching)
# ─────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """
    Extract features from relu1_2, relu2_2, relu3_3, relu4_3 of pretrained VGG19.
    Compute L1 distance in feature space.
    """

    LAYER_NAMES = {
        "relu1_2": 4,
        "relu2_2": 9,
        "relu3_3": 18,
        "relu4_3": 27,
    }
    LAYER_WEIGHTS = [1.0, 0.75, 0.5, 0.25]

    def __init__(self, device: str = "cpu"):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        features = list(vgg.features.children())

        # One sub-network per feature level
        self.slices = nn.ModuleList()
        prev = 0
        for idx in sorted(self.LAYER_NAMES.values()):
            self.slices.append(nn.Sequential(*features[prev:idx]))
            prev = idx

        # Freeze VGG
        for p in self.parameters():
            p.requires_grad = False

        # ImageNet normalisation
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from [-1,1] to ImageNet-normalised."""
        x = (x + 1.0) / 2.0          # [-1,1] → [0,1]
        return (x - self.mean) / self.std

    def _expand_if_grey(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = self._normalize(self._expand_if_grey(pred))
        target = self._normalize(self._expand_if_grey(target))

        loss = torch.tensor(0.0, device=pred.device)
        for w, sl in zip(self.LAYER_WEIGHTS, self.slices):
            pred   = sl(pred)
            target = sl(target)
            loss   = loss + w * F.l1_loss(pred, target)

        return loss


# ─────────────────────────────────────────────
# 4. SSIM Loss
# ─────────────────────────────────────────────

def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """1-D Gaussian; outer product gives 2-D kernel."""
    coords = torch.arange(size, dtype=torch.float) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.outer(g).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)


class SSIMLoss(nn.Module):
    """
    Differentiable SSIM.  Loss = 1 - mean_SSIM (lower = better).
    Handles multi-channel by computing per-channel then averaging.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        kernel = _gaussian_kernel(window_size, sigma)
        self.register_buffer("kernel", kernel)
        self.window_size = window_size
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def _ssim_channel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x, y: (B, 1, H, W) in [-1, 1]."""
        kernel = self.kernel.to(x.device)
        pad = self.window_size // 2
        mu_x  = F.conv2d(x, kernel, padding=pad)
        mu_y  = F.conv2d(y, kernel, padding=pad)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sig_x  = F.conv2d(x*x, kernel, padding=pad) - mu_x2
        sig_y  = F.conv2d(y*y, kernel, padding=pad) - mu_y2
        sig_xy = F.conv2d(x*y, kernel, padding=pad) - mu_xy

        num = (2*mu_xy + self.C1) * (2*sig_xy + self.C2)
        den = (mu_x2 + mu_y2 + self.C1) * (sig_x + sig_y + self.C2)
        return (num / den).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ssim_val = torch.tensor(0.0, device=pred.device)
        for c in range(pred.shape[1]):
            ssim_val += self._ssim_channel(
                pred[:, c:c+1],
                target[:, c:c+1],
            )
        return 1.0 - ssim_val / pred.shape[1]


# ─────────────────────────────────────────────
# 5. Edge Loss (Sobel)
# ─────────────────────────────────────────────

class EdgeLoss(nn.Module):
    """
    Computes Sobel edges on predicted and target images,
    then penalises L1 difference — sharpens roads, buildings, boundaries.
    """

    def __init__(self):
        super().__init__()
        Kx = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]]).view(1, 1, 3, 3)
        Ky = torch.tensor([[-1., -2., -1.],
                            [ 0.,  0.,  0.],
                            [ 1.,  2.,  1.]]).view(1, 1, 3, 3)
        self.register_buffer("Kx", Kx)
        self.register_buffer("Ky", Ky)

    def _edges(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 3:
            grey = (
            0.2126 * x[:, 0:1]
            + 0.7152 * x[:, 1:2]
            + 0.0722 * x[:, 2:3]
        )
        else:
            grey = x

        Kx = self.Kx.to(grey.device)
        Ky = self.Ky.to(grey.device)

        ex = F.conv2d(grey, Kx, padding=1)
        ey = F.conv2d(grey, Ky, padding=1)

        return torch.sqrt(ex**2 + ey**2 + 1e-8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._edges(pred), self._edges(target))


# ─────────────────────────────────────────────
# 6. Combined Loss orchestrator
# ─────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Wraps all individual losses and returns a scalar generator loss
    plus a dict of individual components for logging.
    """

    def __init__(self, weights: Optional[LossWeights] = None, device: str = "cpu"):
        super().__init__()
        if weights is None:
            weights = LossWeights()
        self.w  = weights

        self.gan = GANLoss().to(device)
        self.l1 = L1Loss().to(device)
        self.perceptual = PerceptualLoss(device).to(device)
        self.ssim = SSIMLoss().to(device)
        self.edge = EdgeLoss().to(device)

    def generator_loss(
        self,
        fake_pred:  torch.Tensor,   # discriminator output on fake
        pred_rgb:   torch.Tensor,   # generator output
        target_rgb: torch.Tensor,   # ground truth RGB
    ) -> tuple[torch.Tensor, dict]:

        loss_gan  = self.gan.generator_loss(fake_pred)
        loss_l1   = self.l1(pred_rgb, target_rgb)
        loss_perc = self.perceptual(pred_rgb, target_rgb)
        loss_ssim = self.ssim(pred_rgb, target_rgb)
        loss_edge = self.edge(pred_rgb, target_rgb)

        total = (
            self.w.gan        * loss_gan  +
            self.w.l1         * loss_l1   +
            self.w.perceptual * loss_perc +
            self.w.ssim       * loss_ssim +
            self.w.edge       * loss_edge
        )

        log = {
            "loss/gen_total":      total.item(),
            "loss/gan_gen":        loss_gan.item(),
            "loss/l1":             loss_l1.item(),
            "loss/perceptual":     loss_perc.item(),
            "loss/ssim":           loss_ssim.item(),
            "loss/edge":           loss_edge.item(),
        }
        return total, log

    def discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        loss = self.gan.discriminator_loss(real_pred, fake_pred)
        return loss, {"loss/disc": loss.item()}


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    crit = CombinedLoss(device=device)

    B = 2
    fake_pred  = torch.randn(B, 1, 30, 30).to(device)
    real_pred  = torch.randn(B, 1, 30, 30).to(device)
    pred_rgb   = torch.randn(B, 3, 256, 256).clamp(-1, 1).to(device)
    target_rgb = torch.randn(B, 3, 256, 256).clamp(-1, 1).to(device)

    gen_loss,  gen_log  = crit.generator_loss(fake_pred, pred_rgb, target_rgb)
    disc_loss, disc_log = crit.discriminator_loss(real_pred, fake_pred)

    print("Generator losses:")
    for k, v in gen_log.items():
        print(f"  {k:30s} {v:.4f}")

    print(f"\nDiscriminator loss:  {disc_log['loss/disc']:.4f}")
    print("✅ Phase 4 complete.")
