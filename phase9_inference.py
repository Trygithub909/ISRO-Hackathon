"""
Phase 9 — Inference
Accepts a single IR image or a folder of images and outputs colorised RGB images.
"""

import os
import time
from pathlib import Path
from typing import Union, List, Optional

import cv2
import numpy as np
import torch

from phase2_enhancement import enhance_ir
from phase3_generator   import ResidualUNetGenerator


# ─────────────────────────────────────────────
# Core inference function
# ─────────────────────────────────────────────

def load_generator(checkpoint_path: str, device: str = "cpu") -> ResidualUNetGenerator:
    """Load generator weights from a checkpoint file."""
    gen = ResidualUNetGenerator(in_channels=1, out_channels=3).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(state.get("gen", state))  # handles full checkpoint or bare state_dict
    gen.eval()
    print(f"Generator loaded from {checkpoint_path}")
    return gen


def preprocess_ir(image_path: str, image_size: int = 256) -> tuple[np.ndarray, torch.Tensor]:
    """
    Read, resize, enhance, and tensorise a single IR image.
    Returns (enhanced_np_uint8, tensor_[-1,1]).
    """
    raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Could not read: {image_path}")

    # 16-bit → 8-bit
    if raw.dtype == np.uint16:
        raw = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

    raw     = cv2.resize(raw, (image_size, image_size))
    enhanced = enhance_ir(raw)

    tensor = torch.from_numpy(enhanced).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    tensor = tensor / 127.5 - 1.0
    return enhanced, tensor


@torch.no_grad()
def infer_single(
    gen: ResidualUNetGenerator,
    ir_path: str,
    output_path: str,
    image_size: int = 256,
    device: str = "cpu",
    save_enhanced: bool = False,
) -> np.ndarray:
    """
    Colorise a single IR image and save the result.
    Returns the output BGR image as numpy array.
    """
    enhanced, tensor = preprocess_ir(ir_path, image_size)
    tensor = tensor.to(device)

    rgb_tensor = gen(tensor)                              # (1,3,H,W) in [-1,1]
    rgb = rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    rgb = ((rgb + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, rgb_bgr)

    if save_enhanced:
        enh_path = output_path.replace(".png", "_enhanced_ir.png")
        cv2.imwrite(enh_path, enhanced)

    return rgb_bgr


def infer_folder(
    gen: ResidualUNetGenerator,
    input_dir: str,
    output_dir: str,
    image_size: int = 256,
    device: str = "cpu",
    extensions: tuple = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    save_enhanced: bool = False,
) -> List[str]:
    """
    Colorise all IR images in a folder. Returns list of saved output paths.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    paths = [
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in extensions
    ]

    if not paths:
        raise FileNotFoundError(f"No valid images found in {input_dir}")

    saved = []
    t_start = time.time()

    for i, p in enumerate(paths, 1):
        out_path = str(Path(output_dir) / f"{p.stem}_colorized.png")
        infer_single(gen, str(p), out_path, image_size, device, save_enhanced)
        saved.append(out_path)
        elapsed = time.time() - t_start
        fps = i / elapsed
        print(f"  [{i:04d}/{len(paths)}]  {p.name}  →  {out_path}  ({fps:.2f} img/s)")

    total = time.time() - t_start
    print(f"\n✅ {len(saved)} images processed in {total:.1f}s  "
          f"({len(saved)/total:.2f} img/s)")
    return saved


# ─────────────────────────────────────────────
# Tile-based inference for large images
# ─────────────────────────────────────────────

@torch.no_grad()
def infer_tiled(
    gen: ResidualUNetGenerator,
    ir_path: str,
    output_path: str,
    tile_size: int = 256,
    overlap: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """
    Run inference on large images by splitting into overlapping tiles
    and blending the outputs with a cosine window to hide seams.
    """
    raw = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(ir_path)
    if raw.dtype == np.uint16:
        raw = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

    raw = enhance_ir(raw)
    H, W = raw.shape

    output = np.zeros((H, W, 3), dtype=np.float32)
    weight = np.zeros((H, W, 1),  dtype=np.float32)

    # Cosine blending window
    def _cos_window(size):
        w = np.hanning(size).astype(np.float32)
        return w[:, None] * w[None, :]   # outer product → 2D

    stride = tile_size - overlap

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y2 = min(y + tile_size, H)
            x2 = min(x + tile_size, W)
            tile = raw[y:y2, x:x2]

            # Pad to tile_size if at edge
            ph = tile_size - tile.shape[0]
            pw = tile_size - tile.shape[1]
            if ph > 0 or pw > 0:
                tile = np.pad(tile, ((0, ph), (0, pw)), mode="reflect")

            t = torch.from_numpy(tile).float().unsqueeze(0).unsqueeze(0).to(device)
            t = t / 127.5 - 1.0
            out_t = gen(t).squeeze(0).permute(1, 2, 0).cpu().numpy()
            out_t = ((out_t + 1) / 2 * 255).clip(0, 255)

            # Crop back to actual tile size
            th = y2 - y;  tw = x2 - x
            out_t = out_t[:th, :tw]

            win = _cos_window(tile_size)[:th, :tw, None]
            output[y:y2, x:x2] += out_t * win
            weight[y:y2, x:x2] += win

    output = (output / (weight + 1e-8)).clip(0, 255).astype(np.uint8)
    rgb_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, rgb_bgr)
    return rgb_bgr


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 9: IR Colorisation Inference")
    parser.add_argument("--checkpoint",  required=True, help="Path to best_model.pt")
    parser.add_argument("--input",       required=True, help="Single image path OR folder path")
    parser.add_argument("--output",      default="colorized_output")
    parser.add_argument("--size",        type=int,  default=256)
    parser.add_argument("--tiled",       action="store_true", help="Use tiled inference (large images)")
    parser.add_argument("--save_enhanced", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen    = load_generator(args.checkpoint, device)

    inp = Path(args.input)
    if inp.is_dir():
        infer_folder(gen, str(inp), args.output, args.size, device,
                     save_enhanced=args.save_enhanced)
    elif inp.is_file():
        out_path = str(Path(args.output) / f"{inp.stem}_colorized.png") \
                   if Path(args.output).suffix == "" \
                   else args.output
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if args.tiled:
            infer_tiled(gen, str(inp), out_path, args.size, device=device)
        else:
            infer_single(gen, str(inp), out_path, args.size, device,
                         args.save_enhanced)
        print(f"✅ Saved → {out_path}")
    else:
        print(f"Input not found: {inp}")
