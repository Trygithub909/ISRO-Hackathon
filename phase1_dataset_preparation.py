"""
Phase 1 — Dataset Preparation
Fixes the empty manifest issue and builds a reliable training pipeline.
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image



# ─────────────────────────────────────────────
# 1. Patch Generator  (fixes the empty manifest)
# ─────────────────────────────────────────────

def generate_patches(
    ir_dir: str,
    rgb_dir: str,
    output_dir: str,
    patch_size: int = 256,
    stride: int = 128,          # 50 % overlap → more patches per image
    min_valid_ratio: float = 0.7,  # skip mostly-black / nodata patches
    splits: dict = None,
) -> dict:
    """
    Slice every (IR, RGB) pair into patch_size×patch_size tiles,
    write them to disk, and return a manifest dict.

    Directory layout created:
        output_dir/
            train/ir/  train/rgb/
            val/ir/    val/rgb/
            test/ir/   test/rgb/
            manifest.json
    """
    if splits is None:
        splits = {"train": 0.8, "val": 0.1, "test": 0.1}

    ir_paths  = sorted(Path(ir_dir).glob("*.tif"))  + \
                sorted(Path(ir_dir).glob("*.png"))  + \
                sorted(Path(ir_dir).glob("*.jpg"))
    rgb_paths = {p.stem: p for p in (
                    list(Path(rgb_dir).glob("*.tif")) +
                    list(Path(rgb_dir).glob("*.png")) +
                    list(Path(rgb_dir).glob("*.jpg")))}

    if not ir_paths:
        raise FileNotFoundError(f"No IR images found in {ir_dir}")

    # Create output dirs
    out = Path(output_dir)
    manifest = {"train": [], "val": [], "test": []}

    for split in splits:
        (out / split / "ir").mkdir(parents=True, exist_ok=True)
        (out / split / "rgb").mkdir(parents=True, exist_ok=True)

    all_pairs: List[Tuple[Path, Path]] = []

    for ir_p in ir_paths:

    # Example:
    # scene_ir.png -> scene_rgb.png
        rgb_filename = ir_p.name.replace("_ir.", "_rgb.")
        rgb_p = Path(rgb_dir) / rgb_filename

        if rgb_p.exists():
            all_pairs.append((ir_p, rgb_p))
        else:
            print(f"  [SKIP] No RGB match for {ir_p.name}")
    if not all_pairs:
        raise RuntimeError("No matched (IR, RGB) pairs found. Check directory names.")

    random.shuffle(all_pairs)
    n = len(all_pairs)
    train_end = int(n * splits["train"])
    val_end   = train_end + int(n * splits["val"])

    split_map = (
        ["train"] * train_end +
        ["val"]   * (val_end - train_end) +
        ["test"]  * (n - val_end)
    )

    total_patches = 0
    for (ir_p, rgb_p), split in zip(all_pairs, split_map):
        ir_img  = _load_image(ir_p,  grayscale=True)
        rgb_img = _load_image(rgb_p, grayscale=False)

        if ir_img is None or rgb_img is None:
            continue

        # Resize rgb to match ir if needed
        if ir_img.shape[:2] != rgb_img.shape[:2]:
            rgb_img = cv2.resize(rgb_img, (ir_img.shape[1], ir_img.shape[0]))

        h, w = ir_img.shape[:2]
        patch_count = 0

        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                ir_patch  = ir_img [y:y+patch_size, x:x+patch_size]
                rgb_patch = rgb_img[y:y+patch_size, x:x+patch_size]

                # Skip near-black / nodata patches
                if ir_patch.mean() / 255.0 < (1 - min_valid_ratio):
                    continue

                name = f"{ir_p.stem}_{y:04d}_{x:04d}.png"
                ir_out  = out / split / "ir"  / name
                rgb_out = out / split / "rgb" / name

                cv2.imwrite(str(ir_out),  ir_patch)
                cv2.imwrite(str(rgb_out), rgb_patch)

                manifest[split].append({
                    "ir":  str(ir_out),
                    "rgb": str(rgb_out),
                })
                patch_count += 1

        total_patches += patch_count
        print(f"  {ir_p.name} → {patch_count} patches [{split}]")

    # Save manifest
    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Total patches: {total_patches}")
    for s, items in manifest.items():
        print(f"   {s}: {len(items)}")
    print(f"   Manifest saved → {manifest_path}")

    return manifest


def _load_image(path: Path, grayscale: bool) -> Optional[np.ndarray]:
    """Load image robustly (handles 16-bit GeoTIFF)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"  [WARN] Could not read {path}")
        return None

    # 16-bit → 8-bit
    if img.dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if grayscale:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = img[:, :, :3]  # drop alpha

    return img


# ─────────────────────────────────────────────
# 2. PyTorch Dataset
# ─────────────────────────────────────────────

class IRRGBDataset(Dataset):
    """
    Loads (IR, RGB) patch pairs from the manifest.
    Optionally applies the enhancement pipeline from Phase 2.
    """

    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        image_size: int = 256,
        augment: bool = True,
        enhance: bool = True,          # Phase 2 hook
    ):
        with open(manifest_path) as f:
            manifest = json.load(f)

        self.pairs    = manifest[split]
        self.size     = image_size
        self.augment  = augment and (split == "train")
        self.enhance  = enhance

        # Lazy import so Phase 2 is optional
        if enhance:
            try:
                from phase2_enhancement import enhance_ir
                self._enhance = enhance_ir
            except ImportError:
                print("[WARN] phase2_enhancement not found; running without enhancement.")
                self._enhance = lambda x: x

        self.to_tensor = transforms.ToTensor()   # [0,255] HWC → [0,1] CHW

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]

        ir  = cv2.imread(pair["ir"],  cv2.IMREAD_GRAYSCALE)
        rgb = cv2.imread(pair["rgb"], cv2.IMREAD_COLOR)

        if ir is None or rgb is None:
            # Return a blank sample (handled by collate) instead of crashing
            ir  = np.zeros((self.size, self.size),    dtype=np.uint8)
            rgb = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        # Resize to target size
        ir  = cv2.resize(ir,  (self.size, self.size))
        rgb = cv2.resize(rgb, (self.size, self.size))

        # Phase 2 enhancement
        if self.enhance:
            ir = self._enhance(ir)

        # Augmentation (train only)
        if self.augment:
            ir, rgb = _augment_pair(ir, rgb)

        # Convert to float tensors in [-1, 1]  (standard for GANs)
        ir_t  = torch.from_numpy(ir ).float().unsqueeze(0) / 127.5 - 1.0
        rgb_t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 127.5 - 1.0

        return {"ir": ir_t, "rgb": rgb_t, "path": pair["ir"]}


def _augment_pair(ir: np.ndarray, rgb: np.ndarray):
    """Apply the same random flip / rotation to both images."""
    if random.random() > 0.5:
        ir  = cv2.flip(ir,  1)
        rgb = cv2.flip(rgb, 1)
    if random.random() > 0.5:
        ir  = cv2.flip(ir,  0)
        rgb = cv2.flip(rgb, 0)
    k = random.choice([0, 1, 2, 3])
    if k:
        ir  = np.rot90(ir,  k)
        rgb = np.rot90(rgb, k)
    return np.ascontiguousarray(ir), np.ascontiguousarray(rgb)


# ─────────────────────────────────────────────
# 3. DataLoader factory
# ─────────────────────────────────────────────

def get_dataloaders(
    manifest_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = 256,
    enhance: bool = True,
) -> dict:
    loaders = {}
    for split in ("train", "val", "test"):
        ds = IRRGBDataset(
            manifest_path,
            split=split,
            image_size=image_size,
            augment=(split == "train"),
            enhance=enhance,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
        print(f"  {split}: {len(ds)} samples  |  {len(loaders[split])} batches")
    return loaders


# ─────────────────────────────────────────────
# 4. Quick sanity check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ir_dir",     required=True)
    parser.add_argument("--rgb_dir",    required=True)
    parser.add_argument("--output_dir", default="dataset_patches")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--stride",     type=int, default=128)
    args = parser.parse_args()

    print("=== Phase 1: Generating patches ===")
    manifest = generate_patches(
        ir_dir     = args.ir_dir,
        rgb_dir    = args.rgb_dir,
        output_dir = args.output_dir,
        patch_size = args.patch_size,
        stride     = args.stride,
    )

    print("\n=== Phase 1: Building DataLoaders ===")
    manifest_path = str(Path(args.output_dir) / "manifest.json")
    loaders = get_dataloaders(manifest_path, batch_size=4, num_workers=0, enhance=False)

    batch = next(iter(loaders["train"]))
    print(f"\nSample batch  →  IR: {batch['ir'].shape}   RGB: {batch['rgb'].shape}")
    print("✅ Phase 1 complete.")
