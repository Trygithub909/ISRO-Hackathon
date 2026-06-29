"""
Phase 8 — Object Interpretation
Runs YOLOv8 on original IR images vs generated RGB images
and compares detection performance (confidence, class count, mAP proxy).
"""

import os
import json
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

# ─────────────────────────────────────────────
# YOLOv8 via ultralytics (pip install ultralytics)
# ─────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    warnings.warn("ultralytics not installed. Run: pip install ultralytics")

from phase3_generator import ResidualUNetGenerator


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def tensor_to_cv2(tensor: torch.Tensor) -> np.ndarray:
    """Convert (1, C, H, W) tensor in [-1,1] → uint8 BGR numpy."""
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = ((img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def detect_on_image(model, image_bgr: np.ndarray, conf: float = 0.3) -> dict:
    """Run YOLO on a single BGR image and return structured results."""
    results = model(image_bgr, conf=conf, verbose=False)[0]

    detections = []
    for box in results.boxes:
        detections.append({
            "class_id":   int(box.cls.item()),
            "class_name": model.names[int(box.cls.item())],
            "confidence": float(box.conf.item()),
            "bbox":       box.xyxy.squeeze().tolist(),
        })

    return {
        "n_detections":    len(detections),
        "mean_confidence": float(np.mean([d["confidence"] for d in detections])) if detections else 0.0,
        "classes_found":   list({d["class_name"] for d in detections}),
        "detections":      detections,
    }


def draw_detections(image_bgr: np.ndarray, result_dict: dict) -> np.ndarray:
    """Annotate an image with bounding boxes and labels."""
    img = image_bgr.copy()
    for d in result_dict["detections"]:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        label = f"{d['class_name']} {d['confidence']:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(img, label, (x1, max(y1-6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    return img


# ─────────────────────────────────────────────
# Main comparison function
# ─────────────────────────────────────────────

def compare_detection(
    ir_image_paths: list,
    gen_checkpoint: str,
    output_dir: str = "detection_comparison",
    yolo_model: str = "yolov8n.pt",   # nano for speed; use yolov8m for accuracy
    conf_thresh: float = 0.3,
    image_size: int = 256,
    device: str = "cpu",
) -> dict:
    """
    For each IR image:
      1. Run YOLO on the raw IR image (converted to 3-ch greyscale BGR).
      2. Generate RGB with the trained generator.
      3. Run YOLO on the generated RGB.
      4. Save annotated side-by-side image.
    Returns summary comparison dict.
    """
    if not YOLO_AVAILABLE:
        raise RuntimeError("Install ultralytics: pip install ultralytics")

    device = device if torch.cuda.is_available() else "cpu"
    out    = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load generator
    gen = ResidualUNetGenerator(in_channels=1, out_channels=3).to(device)
    ckpt = torch.load(gen_checkpoint, map_location=device)
    gen.load_state_dict(ckpt.get("gen", ckpt))
    gen.eval()

    # Load YOLO
    yolo = YOLO(yolo_model)

    all_results = []

    for ir_path in ir_image_paths:
        ir_bgr = cv2.imread(str(ir_path))
        if ir_bgr is None:
            print(f"  [SKIP] Could not read {ir_path}")
            continue

        ir_bgr = cv2.resize(ir_bgr, (image_size, image_size))
        ir_grey = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2GRAY)

        # ── Run YOLO on IR (grey expanded to BGR) ──
        ir_3ch = cv2.cvtColor(ir_grey, cv2.COLOR_GRAY2BGR)
        ir_res  = detect_on_image(yolo, ir_3ch, conf_thresh)

        # ── Generate RGB ──────────────────────────
        ir_t = torch.from_numpy(ir_grey).float().unsqueeze(0).unsqueeze(0).to(device)
        ir_t = ir_t / 127.5 - 1.0
        with torch.no_grad():
            rgb_t = gen(ir_t)
        rgb_bgr = tensor_to_cv2(rgb_t)

        # ── Run YOLO on generated RGB ──────────────
        rgb_res = detect_on_image(yolo, rgb_bgr, conf_thresh)

        # ── Annotate and save side-by-side ─────────
        ann_ir  = draw_detections(ir_3ch,  ir_res)
        ann_rgb = draw_detections(rgb_bgr, rgb_res)

        label_ir  = f"IR: {ir_res['n_detections']} obj, conf={ir_res['mean_confidence']:.2f}"
        label_rgb = f"RGB: {rgb_res['n_detections']} obj, conf={rgb_res['mean_confidence']:.2f}"

        canvas = np.zeros((image_size + 30, image_size * 2, 3), dtype=np.uint8)
        canvas[30:, :image_size]            = ann_ir
        canvas[30:, image_size:image_size*2] = ann_rgb
        cv2.putText(canvas, label_ir,  (5, 22),              cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
        cv2.putText(canvas, label_rgb, (image_size + 5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

        name = Path(ir_path).stem
        cv2.imwrite(str(out / f"{name}_comparison.png"), canvas)

        entry = {
            "image":          name,
            "ir_detections":  ir_res["n_detections"],
            "rgb_detections": rgb_res["n_detections"],
            "ir_conf":        ir_res["mean_confidence"],
            "rgb_conf":       rgb_res["mean_confidence"],
            "ir_classes":     ir_res["classes_found"],
            "rgb_classes":    rgb_res["classes_found"],
        }
        all_results.append(entry)
        print(f"  {name}  |  IR: {entry['ir_detections']} obj  →  RGB: {entry['rgb_detections']} obj")

    # Aggregate
    summary = {
        "n_images":          len(all_results),
        "avg_ir_detections":  float(np.mean([r["ir_detections"]  for r in all_results])) if all_results else 0,
        "avg_rgb_detections": float(np.mean([r["rgb_detections"] for r in all_results])) if all_results else 0,
        "avg_ir_conf":        float(np.mean([r["ir_conf"]        for r in all_results])) if all_results else 0,
        "avg_rgb_conf":       float(np.mean([r["rgb_conf"]       for r in all_results])) if all_results else 0,
        "per_image":          all_results,
    }

    with open(out / "detection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Detection Comparison Summary ===")
    print(f"  Avg detections  IR  : {summary['avg_ir_detections']:.2f}")
    print(f"  Avg detections  RGB : {summary['avg_rgb_detections']:.2f}")
    print(f"  Avg confidence  IR  : {summary['avg_ir_conf']:.3f}")
    print(f"  Avg confidence  RGB : {summary['avg_rgb_conf']:.3f}")
    print(f"  Results saved → {out}")

    return summary


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, glob

    parser = argparse.ArgumentParser()
    parser.add_argument("--ir_dir",     required=True,  help="Folder of IR images")
    parser.add_argument("--checkpoint", required=True,  help="Generator checkpoint .pt")
    parser.add_argument("--output_dir", default="detection_comparison")
    parser.add_argument("--yolo",       default="yolov8n.pt")
    parser.add_argument("--conf",       type=float, default=0.3)
    args = parser.parse_args()

    ir_paths = (
        glob.glob(os.path.join(args.ir_dir, "*.png")) +
        glob.glob(os.path.join(args.ir_dir, "*.jpg")) +
        glob.glob(os.path.join(args.ir_dir, "*.tif"))
    )
    if not ir_paths:
        print("No images found in", args.ir_dir)
        exit(1)

    compare_detection(
        ir_image_paths = ir_paths,
        gen_checkpoint = args.checkpoint,
        output_dir     = args.output_dir,
        yolo_model     = args.yolo,
        conf_thresh    = args.conf,
    )
    print("✅ Phase 8 complete.")
