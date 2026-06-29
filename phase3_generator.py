"""
Phase 3 — Advanced Generator
Architecture: Residual U-Net + CBAM (Channel & Spatial Attention) + Skip Attention
Input:  1-channel enhanced IR   →   Output: 3-channel RGB
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention (part of CBAM)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.fc(self.avg_pool(x))
        mx  = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    """Spatial attention (part of CBAM)."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel then spatial)."""

    def __init__(self, channels: int, reduction: int = 16, spatial_k: int = 7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(spatial_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)
        x = self.spatial(x)
        return x


class ResBlock(nn.Module):
    """
    Residual block with two Conv-BN-ReLU layers and an optional projection.
    Optionally fuses CBAM attention after the residual sum.
    """

    def __init__(self, channels: int, use_attention: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)
        self.cbam  = CBAM(channels) if use_attention else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        if self.cbam is not None:
            out = self.cbam(out)
        return self.relu(out)


class SkipAttention(nn.Module):
    """
    Gate the skip connection with a learnable spatial attention map.
    Prevents the decoder from blindly copying noisy encoder features.
    """

    def __init__(self, enc_ch: int, dec_ch: int, mid_ch: int = None):
        super().__init__()
        if mid_ch is None:
            mid_ch = enc_ch // 2

        self.W_enc = nn.Sequential(
            nn.Conv2d(enc_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
        )
        self.W_dec = nn.Sequential(
            nn.Conv2d(dec_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, enc: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        # Upsample dec to match enc spatial size
        dec_up = F.interpolate(dec, size=enc.shape[2:], mode="bilinear", align_corners=False)
        gate   = self.psi(self.W_enc(enc) + self.W_dec(dec_up))
        return enc * gate


class EncoderBlock(nn.Module):
    """Encoder stage: ResBlock(s) + optional CBAM + MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, n_res: int = 2, use_attention: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                  nn.BatchNorm2d(out_ch),
                  nn.ReLU(inplace=True)]
        for i in range(n_res):
            layers.append(ResBlock(out_ch, use_attention=(use_attention and i == n_res - 1)))
        self.conv  = nn.Sequential(*layers)
        self.pool  = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        return skip, self.pool(skip)


class DecoderBlock(nn.Module):
    """Decoder stage: ConvTranspose + SkipAttention + concat + ResBlock(s)."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, n_res: int = 2, use_attention: bool = True):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.skip_attn = SkipAttention(skip_ch, out_ch)

        layers = [nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
                  nn.BatchNorm2d(out_ch),
                  nn.ReLU(inplace=True)]
        for i in range(n_res):
            layers.append(ResBlock(out_ch, use_attention=(use_attention and i == n_res - 1)))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x    = self.up(x)
        skip = self.skip_attn(skip, x)
        # Pad if sizes differ by 1 pixel (odd input dimensions)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────
# Full Generator
# ─────────────────────────────────────────────

class ResidualUNetGenerator(nn.Module):
    """
    Residual U-Net Generator
    ─────────────────────────────────────────────
    Encoder depths:    [64, 128, 256, 512]
    Bottleneck:        1024 channels + 2 ResBlocks with CBAM
    Decoder mirrors encoder via skip attention gates.
    Output activation: Tanh (maps to [-1, 1] for GAN training)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 3, base_ch: int = 64):
        super().__init__()

        c = base_ch       # 64
        # Encoder
        self.enc1 = EncoderBlock(in_channels, c,      n_res=2, use_attention=False)
        self.enc2 = EncoderBlock(c,           c*2,    n_res=2, use_attention=True)
        self.enc3 = EncoderBlock(c*2,         c*4,    n_res=2, use_attention=True)
        self.enc4 = EncoderBlock(c*4,         c*8,    n_res=2, use_attention=True)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c*8, c*16, 3, padding=1, bias=False),
            nn.BatchNorm2d(c*16),
            nn.ReLU(inplace=True),
            ResBlock(c*16, use_attention=True),
            ResBlock(c*16, use_attention=True),
        )

        # Decoder
        self.dec4 = DecoderBlock(c*16, c*8,  c*8,  n_res=2, use_attention=True)
        self.dec3 = DecoderBlock(c*8,  c*4,  c*4,  n_res=2, use_attention=True)
        self.dec2 = DecoderBlock(c*4,  c*2,  c*2,  n_res=2, use_attention=True)
        self.dec1 = DecoderBlock(c*2,  c,    c,    n_res=2, use_attention=False)

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(c, out_channels, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.head(x)


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    gen = ResidualUNetGenerator(in_channels=1, out_channels=3, base_ch=64).to(device)

    # Parameter count
    params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    print(f"Generator parameters: {params:,}")

    # Forward pass
    dummy = torch.randn(2, 1, 256, 256).to(device)
    out   = gen(dummy)
    print(f"Input:  {dummy.shape}  →  Output: {out.shape}")
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]  (should be within [-1, 1])")
    print("✅ Phase 3 complete.")
