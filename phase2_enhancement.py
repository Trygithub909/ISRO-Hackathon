"""
Phase 2 — Infrared Image Enhancement
Pipeline: Raw IR → CLAHE → Gamma Correction → Bilateral Filter → Unsharp Mask → Enhanced IR
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

@dataclass
class EnhancementConfig:
    # CLAHE
    clahe_clip_limit:   float = 3.0
    clahe_tile_grid:    int   = 8       # tile grid size (NxN)

    # Gamma correction
    gamma:              float = 1.2     # >1 brightens midtones

    # Bilateral filter
    bilateral_d:        int   = 9       # diameter of pixel neighbourhood
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0

    # Unsharp mask
    unsharp_kernel:     int   = 5       # must be odd
    unsharp_strength:   float = 1.5     # how much to add back
    unsharp_threshold:  int   = 10      # only sharpen if diff > threshold


# ─────────────────────────────────────────────
# Individual steps
# ─────────────────────────────────────────────

def apply_clahe(image: np.ndarray, cfg: EnhancementConfig) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization.
    Enhances local contrast without over-amplifying noise.
    """
    assert image.ndim == 2, "CLAHE expects a grayscale image."
    clahe = cv2.createCLAHE(
        clipLimit   = cfg.clahe_clip_limit,
        tileGridSize= (cfg.clahe_tile_grid, cfg.clahe_tile_grid),
    )
    return clahe.apply(image)


def apply_gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    """
    Gamma correction with a precomputed look-up table (fast).
    gamma < 1  → darker   |  gamma > 1 → brighter midtones
    """
    inv_gamma = 1.0 / gamma
    lut = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, lut)


def apply_bilateral_filter(image: np.ndarray, cfg: EnhancementConfig) -> np.ndarray:
    """
    Edge-preserving smoothing — reduces sensor noise while keeping boundaries sharp.
    """
    return cv2.bilateralFilter(
        image,
        d           = cfg.bilateral_d,
        sigmaColor  = cfg.bilateral_sigma_color,
        sigmaSpace  = cfg.bilateral_sigma_space,
    )


def apply_unsharp_mask(image: np.ndarray, cfg: EnhancementConfig) -> np.ndarray:
    """
    Unsharp masking:  output = original + strength * (original - blurred)
    Only applied where the difference exceeds the threshold to avoid noise amplification.
    """
    blurred = cv2.GaussianBlur(image, (cfg.unsharp_kernel, cfg.unsharp_kernel), 0)
    diff    = image.astype(np.int16) - blurred.astype(np.int16)

    mask    = np.abs(diff) > cfg.unsharp_threshold
    sharpened = image.astype(np.float32) + cfg.unsharp_strength * diff * mask

    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# Full pipeline (importable by Dataset)
# ─────────────────────────────────────────────

_default_cfg = EnhancementConfig()


def enhance_ir(
    image: np.ndarray,
    cfg: Optional[EnhancementConfig] = None,
) -> np.ndarray:
    """
    Main entry point used by phase1_dataset_preparation.IRRGBDataset.

    Args:
        image: uint8 grayscale (H, W) array.
        cfg:   EnhancementConfig; uses defaults if None.

    Returns:
        Enhanced uint8 grayscale array of the same shape.
    """
    if cfg is None:
        cfg = _default_cfg

    # Ensure uint8 grayscale
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    step1 = apply_clahe(image, cfg)
    step2 = apply_gamma_correction(step1, cfg.gamma)
    step3 = apply_bilateral_filter(step2, cfg)
    step4 = apply_unsharp_mask(step3, cfg)

    return step4


# ─────────────────────────────────────────────
# Visual comparison utility
# ─────────────────────────────────────────────

def visualize_pipeline(image_path: str, save_path: Optional[str] = None):
    """Save a side-by-side strip: Raw → CLAHE → Gamma → Bilateral → Unsharp."""
    raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(image_path)

    cfg   = EnhancementConfig()
    s1    = apply_clahe(raw, cfg)
    s2    = apply_gamma_correction(s1, cfg.gamma)
    s3    = apply_bilateral_filter(s2, cfg)
    s4    = apply_unsharp_mask(s3, cfg)

    labels = ["Raw IR", "CLAHE", "Gamma", "Bilateral", "Unsharp (Final)"]
    steps  = [raw, s1, s2, s3, s4]

    h, w   = raw.shape
    canvas = np.zeros((h + 30, w * len(steps), 3), dtype=np.uint8)

    for i, (img, label) in enumerate(zip(steps, labels)):
        col = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        canvas[30:, i*w:(i+1)*w] = col
        cv2.putText(
            canvas, label, (i*w + 5, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )

    if save_path:
        cv2.imwrite(save_path, canvas)
        print(f"Saved pipeline visualization → {save_path}")
    else:
        cv2.imshow("Enhancement Pipeline", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return canvas


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Phase 2: Batch IR Enhancement")

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Folder containing IR images"
    )

    parser.add_argument(
        "--output_dir",
        default="enhanced_ir",
        help="Folder to save enhanced images"
    )

    parser.add_argument("--gamma", type=float, default=1.2)
    parser.add_argument("--clip", type=float, default=3.0)

    args = parser.parse_args()

    cfg = EnhancementConfig(
        clahe_clip_limit=args.clip,
        gamma=args.gamma
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = []

    for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
        image_files.extend(input_dir.glob(ext))

    if len(image_files) == 0:
        raise RuntimeError(f"No images found in {input_dir}")

    print(f"Found {len(image_files)} images")

    count = 0

    for img_path in image_files:

        raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if raw is None:
            print(f"Skipping {img_path.name}")
            continue

        enhanced = enhance_ir(raw, cfg)

        save_path = output_dir / img_path.name

        cv2.imwrite(str(save_path), enhanced)

        count += 1

        if count % 100 == 0:
            print(f"Processed {count} images")

    print(f"\n✅ Enhanced {count} images")
    print(f"Saved to: {output_dir}")