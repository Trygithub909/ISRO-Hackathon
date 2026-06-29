# IR Colorisation Project — Requirements & Usage

## Install

```bash
pip install torch torchvision opencv-python numpy tqdm ultralytics tensorboard
```

---

## Phase-by-phase usage

### Phase 1 — Dataset Preparation (fixes empty manifest)
```bash
python phase1_dataset_preparation.py \
  --ir_dir  data/sentinel_ir \
  --rgb_dir data/sentinel_rgb \
  --output_dir dataset_patches \
  --patch_size 256 --stride 128
```

### Phase 2 — IR Enhancement (standalone test)
```bash
python phase2_enhancement.py path/to/single_ir.tif \
  --output enhanced.png --viz pipeline_viz.png
```

### Phase 3 — Generator (architecture test)
```bash
python phase3_generator.py
# Prints parameter count + forward pass shapes
```

### Phase 4 — Loss Functions (unit test)
```bash
python phase4_losses.py
```

### Phase 5 — Discriminator (unit test)
```bash
python phase5_discriminator.py
```

### Phase 6 — Training
```bash
python phase6_train.py \
  --manifest dataset_patches/manifest.json \
  --checkpoint_dir checkpoints \
  --log_dir runs/exp1 \
  --epochs 100 --batch 8

# Resume from checkpoint
python phase6_train.py --resume checkpoints/checkpoint_epoch_0050.pt ...

# TensorBoard
tensorboard --logdir runs/
```

### Phase 7 — Evaluation
```bash
python phase7_evaluation.py \
  --manifest  dataset_patches/manifest.json \
  --checkpoint checkpoints/best_model.pt \
  --split test --output_csv results.csv
```

### Phase 8 — Object Detection Comparison
```bash
python phase8_object_detection.py \
  --ir_dir dataset_patches/test/ir \
  --checkpoint checkpoints/best_model.pt \
  --output_dir detection_results
```

### Phase 9 — Inference (single image or folder)
```bash
# Single image
python phase9_inference.py \
  --checkpoint checkpoints/best_model.pt \
  --input path/to/ir.png --output colorized.png

# Folder
python phase9_inference.py \
  --checkpoint checkpoints/best_model.pt \
  --input path/to/ir_folder/ --output colorized_output/

# Large images (tiled)
python phase9_inference.py --tiled \
  --checkpoint checkpoints/best_model.pt \
  --input large_ir.tif --output large_colorized.png
```

### Phase 10 — Demo (hackathon presentation)
```bash
python phase10_demo.py \
  --checkpoint checkpoints/best_model.pt \
  --ir_dir dataset_patches/test/ir \
  --gt_rgb_dir dataset_patches/test/rgb \
  --output_dir demo_output --n_grid 4
# Opens demo_output/demo_grid.png — 5-panel strip per image
```

---

## File dependency map

```
phase1_dataset_preparation.py   ← standalone (fixes manifest)
phase2_enhancement.py           ← standalone; imported by phase1
phase3_generator.py             ← standalone
phase4_losses.py                ← imports phase3
phase5_discriminator.py         ← standalone
phase6_train.py                 ← imports phase1,3,4,5
phase7_evaluation.py            ← imports phase1,3
phase8_object_detection.py      ← imports phase3,9
phase9_inference.py             ← imports phase2,3
phase10_demo.py                 ← imports phase2,3,9
```
