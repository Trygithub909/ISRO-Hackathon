"""
Phase 5 — Improved Discriminator
Features: Spectral Normalization + Multi-scale PatchGAN (3 scales)
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from typing import List


# ─────────────────────────────────────────────
# Single-scale PatchGAN with Spectral Norm
# ─────────────────────────────────────────────

def _sn_conv(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """Spectral-normalised Conv2d."""
    return spectral_norm(
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=False)
    )


class PatchDiscriminator(nn.Module):
    """
    70×70 PatchGAN with Spectral Normalization.
    Classifies overlapping 70×70 patches as real/fake.
    """

    def __init__(self, in_channels: int = 4, base_ch: int = 64, n_layers: int = 3):
        """
        Args:
            in_channels: IR (1) + RGB (3) = 4  (condition on IR)
            base_ch: feature map multiplier base
            n_layers: depth of discriminator
        """
        super().__init__()

        layers = [
            # First layer: no BatchNorm (standard practice)
            _sn_conv(in_channels, base_ch, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        ch = base_ch
        for i in range(1, n_layers):
            prev_ch = ch
            ch = min(ch * 2, 512)
            layers += [
                _sn_conv(prev_ch, ch, stride=2),
                nn.BatchNorm2d(ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # stride-1 conv before final output
        layers += [
            _sn_conv(ch, ch * 2, stride=1),
            nn.BatchNorm2d(ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            _sn_conv(ch * 2, 1, stride=1),   # output: 1 channel patch map
        ]

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, ir: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """Concatenate IR (condition) with RGB (generated or real) then classify."""
        return self.model(torch.cat([ir, rgb], dim=1))


# ─────────────────────────────────────────────
# Multi-scale PatchGAN
# ─────────────────────────────────────────────

class MultiScaleDiscriminator(nn.Module):
    """
    Three PatchGAN discriminators operating at different scales:
        Scale 0 → original resolution (fine details)
        Scale 1 → 2× downsampled       (mid-level structure)
        Scale 2 → 4× downsampled       (global consistency)

    Losses are averaged across scales.
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_ch: int = 64,
        n_scales: int = 3,
        n_layers: int = 3,
    ):
        super().__init__()
        self.n_scales = n_scales
        self.discs = nn.ModuleList([
            PatchDiscriminator(in_channels, base_ch, n_layers)
            for _ in range(n_scales)
        ])
        self.down = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, ir: torch.Tensor, rgb: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns a list of patch-map tensors, one per scale.
        The training loop averages the losses across scales.
        """
        outputs = []
        for i, disc in enumerate(self.discs):
            if i > 0:
                ir  = self.down(ir)
                rgb = self.down(rgb)
            outputs.append(disc(ir, rgb))
        return outputs


# ─────────────────────────────────────────────
# Discriminator loss helpers
# ─────────────────────────────────────────────

def disc_loss_multiscale(
    disc: MultiScaleDiscriminator,
    ir:   torch.Tensor,
    real: torch.Tensor,
    fake: torch.Tensor,
) -> torch.Tensor:
    """
    Compute LSGAN discriminator loss over all scales.
    fake should be detached before calling this.
    """
    real_preds = disc(ir, real)
    fake_preds = disc(ir, fake.detach())

    loss = torch.tensor(0.0, device=ir.device)
    for rp, fp in zip(real_preds, fake_preds):
        loss += 0.5 * (
            torch.mean((rp - 1.0) ** 2) +
            torch.mean(fp ** 2)
        )
    return loss / len(real_preds)


def gen_loss_multiscale(
    disc: MultiScaleDiscriminator,
    ir:   torch.Tensor,
    fake: torch.Tensor,
) -> torch.Tensor:
    """
    Generator adversarial loss over all scales (fool the discriminator).
    """
    fake_preds = disc(ir, fake)
    loss = torch.tensor(0.0, device=ir.device)
    for fp in fake_preds:
        loss += torch.mean((fp - 1.0) ** 2)
    return loss / len(fake_preds)


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    disc = MultiScaleDiscriminator(in_channels=4, base_ch=64, n_scales=3).to(device)
    params = sum(p.numel() for p in disc.parameters() if p.requires_grad)
    print(f"Discriminator parameters: {params:,}")

    B = 2
    ir   = torch.randn(B, 1, 256, 256).to(device)
    real = torch.randn(B, 3, 256, 256).to(device)
    fake = torch.randn(B, 3, 256, 256).to(device)

    preds = disc(ir, real)
    print("Patch map shapes per scale:")
    for i, p in enumerate(preds):
        print(f"  Scale {i}: {p.shape}")

    d_loss = disc_loss_multiscale(disc, ir, real, fake)
    g_loss = gen_loss_multiscale(disc, ir, fake)
    print(f"\nDisc loss: {d_loss.item():.4f}  |  Gen adv loss: {g_loss.item():.4f}")
    print("✅ Phase 5 complete.")
