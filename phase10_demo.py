"""
Phase 10 — Demo
Visualises the complete pipeline:
  Raw IR → Enhanced IR → Generated RGB → Ground Truth RGB → Detected Objects
Ideal for hackathon presentations.
"""

import os
import json
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
import torch

from phase2_enhancement      import enhance_ir, EnhancementConfig
from phase3_generator        import ResidualUNetGenerator
from phase9_inference        import load_generator, preprocess_ir

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# ─────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────

BG_COLOR     = (15,  15,  15)    # near-black background
HEADER_COLOR = (30,  30,  30)
TEXT_COLOR   = (240, 240, 240)
ACCENT       = (0,   210, 140)   # green
BOX_COLOR    = (60,  200, 255)   # cyan bounding boxes


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _add_label(img: np.ndarray, text: str, color=TEXT_COLOR) -> np.ndarray:
    """Add a centred label bar above an image."""
    h, w = img.shape[:2]
    bar = np.full((32, w, 3), HEADER_COLOR, dtype=np.uint8)
    tw, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0], None
    x = max((w - tw[0]) // 2, 5) if isinstance(tw, tuple) else 5
    cv2.putText(bar, text, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def _ensure_3ch(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _draw_detections(img: np.ndarray, detections: list) -> np.ndarray:
    out = img.copy()
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        label = f"{d['class_name']} {d['confidence']:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, 2)
        cv2.putText(out, label, (x1, max(y1-5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, BOX_COLOR, 1, cv2.LINE_AA)
    return out


def _run_yolo(yolo_model, image_bgr: np.ndarray, conf: float = 0.3) -> list:
    if not YOLO_AVAILABLE or yolo_model is None:
        return []
    results = yolo_model(image_bgr, conf=conf, verbose=False)[0]
    dets = []
    for box in results.boxes:
        dets.append({
            "class_name": yolo_model.names[int(box.cls.item())],
            "confidence": float(box.conf.item()),
            "bbox":       box.xyxy.squeeze().tolist(),
        })
    return dets


# ─────────────────────────────────────────────
# Single-image demo panel
# ─────────────────────────────────────────────

@torch.no_grad()
def create_demo_panel(
    ir_path: str,
    gen: ResidualUNetGenerator,
    device: str,
    gt_rgb_path: Optional[str] = None,
    image_size:  int = 256,
    yolo_model=None,
    conf_thresh: float = 0.3,
    enhancement_cfg: Optional[EnhancementConfig] = None,
) -> np.ndarray:
    """
    Build a 5-panel image:
    [Raw IR | Enhanced IR | Generated RGB | Ground Truth RGB | Detected Objects]
    Ground truth and object panels are grey if not available.
    """

    # ── Raw IR ────────────────────────────────
    raw = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(ir_path)
    if raw.dtype == np.uint16:
        raw = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    raw = cv2.resize(raw, (image_size, image_size))

    # ── Enhanced IR ────────────────────────────
    enhanced = enhance_ir(raw, enhancement_cfg)

    # ── Generated RGB ──────────────────────────
    t = torch.from_numpy(enhanced).float().unsqueeze(0).unsqueeze(0).to(device)
    t = t / 127.5 - 1.0
    rgb_t = gen(t)
    rgb_np = rgb_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    rgb_np = ((rgb_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

    # ── Ground Truth RGB ───────────────────────
    if gt_rgb_path and Path(gt_rgb_path).exists():
        gt_bgr = cv2.imread(gt_rgb_path)
        gt_bgr = cv2.resize(gt_bgr, (image_size, image_size))
    else:
        gt_bgr = np.full((image_size, image_size, 3), 50, dtype=np.uint8)
        cv2.putText(gt_bgr, "N/A", (image_size//2 - 20, image_size//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (150,150,150), 2)

    # ── Detected Objects ───────────────────────
    dets = _run_yolo(yolo_model, rgb_bgr, conf_thresh)
    det_panel = _draw_detections(rgb_bgr.copy(), dets)
    if dets:
        cv2.putText(det_panel, f"{len(dets)} objects", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ACCENT, 1)

    # ── Assemble panels ────────────────────────
    raw_3ch  = _ensure_3ch(raw)
    enh_3ch  = _ensure_3ch(enhanced)

    panels = [
        _add_label(raw_3ch,   "Raw IR",          TEXT_COLOR),
        _add_label(enh_3ch,   "Enhanced IR",     ACCENT),
        _add_label(rgb_bgr,   "Generated RGB",   ACCENT),
        _add_label(gt_bgr,    "Ground Truth RGB", TEXT_COLOR),
        _add_label(det_panel, "Detected Objects", BOX_COLOR),
    ]

    row = np.hstack(panels)

    # Add title bar
    title_bar = np.full((44, row.shape[1], 3), BG_COLOR, dtype=np.uint8)
    title = "IR Colorisation Demo  |  Raw → Enhanced → Generated → GT → Objects"
    cv2.putText(title_bar, title, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, ACCENT, 1, cv2.LINE_AA)

    return np.vstack([title_bar, row])


# ─────────────────────────────────────────────
# Batch demo (saves one panel per image + grid)
# ─────────────────────────────────────────────

def run_demo(
    ir_paths: List[str],
    gen_checkpoint: str,
    output_dir: str = "demo_output",
    gt_rgb_paths: Optional[List[str]] = None,
    image_size: int = 256,
    yolo_model_name: str = "yolov8n.pt",
    conf_thresh: float = 0.3,
    n_grid_rows: int = 4,   # how many images to include in the summary grid
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Demo running on {device}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    gen = load_generator(gen_checkpoint, device)

    yolo = YOLO(yolo_model_name) if YOLO_AVAILABLE else None
    if not YOLO_AVAILABLE:
        print("[WARN] YOLO not available; object panel will be undetected (plain RGB).")

    panels = []
    for i, ir_p in enumerate(ir_paths):
        gt_p = (gt_rgb_paths[i] if gt_rgb_paths and i < len(gt_rgb_paths) else None)
        try:
            panel = create_demo_panel(
                ir_path       = ir_p,
                gen           = gen,
                device        = device,
                gt_rgb_path   = gt_p,
                image_size    = image_size,
                yolo_model    = yolo,
                conf_thresh   = conf_thresh,
            )
            name    = Path(ir_p).stem
            out_p   = str(Path(output_dir) / f"{name}_demo.png")
            cv2.imwrite(out_p, panel)
            print(f"  Saved → {out_p}")
            panels.append(panel)
        except Exception as e:
            print(f"  [ERROR] {ir_p}: {e}")

    # Summary grid (first n_grid_rows)
    if panels:
        grid = np.vstack(panels[:n_grid_rows])
        grid_path = str(Path(output_dir) / "demo_grid.png")
        cv2.imwrite(grid_path, grid)
        print(f"\n🖼  Summary grid saved → {grid_path}")
        print(f"   ({len(panels[:n_grid_rows])} images, {grid.shape[1]}×{grid.shape[0]} px)")

    print("✅ Phase 10 Demo complete.")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, glob

    parser = argparse.ArgumentParser(description="Phase 10: Demo")
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--ir_dir",      required=True,    help="Folder of test IR images")
    parser.add_argument("--gt_rgb_dir",  default=None,     help="Folder of ground truth RGB (optional)")
    parser.add_argument("--output_dir",  default="demo_output")
    parser.add_argument("--size",        type=int, default=256)
    parser.add_argument("--yolo",        default="yolov8n.pt")
    parser.add_argument("--conf",        type=float, default=0.3)
    parser.add_argument("--n_grid",      type=int,   default=4)
    args = parser.parse_args()

    exts  = ("*.png", "*.jpg", "*.tif")
    ir_paths = []
    for ext in exts:
        ir_paths += glob.glob(os.path.join(args.ir_dir, ext))
    ir_paths.sort()

    gt_paths = None
    if args.gt_rgb_dir:
        gt_paths = []
        for ext in exts:
            gt_paths += glob.glob(os.path.join(args.gt_rgb_dir, ext))
        gt_paths.sort()

    run_demo(
        ir_paths         = ir_paths,
        gen_checkpoint   = args.checkpoint,
        output_dir       = args.output_dir,
        gt_rgb_paths     = gt_paths,
        image_size       = args.size,
        yolo_model_name  = args.yolo,
        conf_thresh      = args.conf,
        n_grid_rows      = args.n_grid,
    )
