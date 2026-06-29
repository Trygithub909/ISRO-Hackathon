"""
Phase 6 — Better Training Pipeline
Features: Mixed Precision, Gradient Clipping, Early Stopping,
          Auto-checkpoints, Best Model Saving, TensorBoard Logging
"""

import os
import time
import json
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from phase3_generator     import ResidualUNetGenerator
from phase5_discriminator import MultiScaleDiscriminator, disc_loss_multiscale, gen_loss_multiscale
from phase4_losses        import CombinedLoss, LossWeights
from phase1_dataset_preparation import get_dataloaders


# ─────────────────────────────────────────────
# Training config
# ─────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Paths
    manifest_path:  str  = "dataset_patches/manifest.json"
    checkpoint_dir: str  = "checkpoints"
    log_dir:        str  = "runs/ir_colorization"

    # Architecture
    base_ch:        int  = 64
    n_scales:       int  = 3      # discriminator scales

    # Training
    n_epochs:       int  = 100
    batch_size:     int  = 8
    image_size:     int  = 256
    num_workers:    int  = 4

    # Optimizers
    lr_gen:         float = 2e-4
    lr_disc:        float = 2e-4
    beta1:          float = 0.5
    beta2:          float = 0.999

    # Stability
    grad_clip:      float = 1.0    # max gradient norm
    mixed_precision: bool = True

    # Scheduling
    lr_decay_epoch: int  = 50      # start linear LR decay
    lr_decay_ratio: float = 0.1    # decay multiplier over remaining epochs

    # Early stopping
    patience:       int  = 15      # epochs without val improvement
    min_delta:      float = 1e-4   # minimum improvement

    # Logging / saving
    save_every:     int  = 5       # save checkpoint every N epochs
    val_vis_n:      int  = 4       # number of validation images to visualise

    # Loss weights
    w_gan:          float = 1.0
    w_l1:           float = 100.0
    w_perceptual:   float = 10.0
    w_ssim:         float = 10.0
    w_edge:         float = 5.0


# ─────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ─────────────────────────────────────────────
# Checkpoint utilities
# ─────────────────────────────────────────────

def save_checkpoint(
    epoch: int,
    gen: nn.Module,
    disc: nn.Module,
    opt_gen: torch.optim.Optimizer,
    opt_disc: torch.optim.Optimizer,
    scaler_gen: GradScaler,
    scaler_disc: GradScaler,
    val_loss: float,
    is_best: bool,
    save_dir: str,
):
    state = {
        "epoch":      epoch,
        "val_loss":   val_loss,
        "gen":        gen.state_dict(),
        "disc":       disc.state_dict(),
        "opt_gen":    opt_gen.state_dict(),
        "opt_disc":   opt_disc.state_dict(),
        "scaler_gen":  scaler_gen.state_dict(),
        "scaler_disc": scaler_disc.state_dict(),
    }
    ckpt = Path(save_dir) / f"checkpoint_epoch_{epoch:04d}.pt"
    torch.save(state, ckpt)
    if is_best:
        shutil.copyfile(ckpt, Path(save_dir) / "best_model.pt")
        print(f"  💾 New best model saved  (val_loss={val_loss:.5f})")
    return ckpt


def load_checkpoint(path: str, gen, disc, opt_gen, opt_disc, scaler_gen, scaler_disc, device):
    state = torch.load(path, map_location=device)
    gen.load_state_dict(state["gen"])
    disc.load_state_dict(state["disc"])
    opt_gen.load_state_dict(state["opt_gen"])
    opt_disc.load_state_dict(state["opt_disc"])
    scaler_gen.load_state_dict(state["scaler_gen"])
    scaler_disc.load_state_dict(state["scaler_disc"])
    return state["epoch"], state["val_loss"]


# ─────────────────────────────────────────────
# Learning-rate scheduler (linear decay)
# ─────────────────────────────────────────────

def build_lr_lambda(cfg: TrainConfig):
    """Constant LR until lr_decay_epoch, then linearly decay to lr*lr_decay_ratio."""
    def fn(epoch):
        if epoch < cfg.lr_decay_epoch:
            return 1.0
        progress = (epoch - cfg.lr_decay_epoch) / max(cfg.n_epochs - cfg.lr_decay_epoch, 1)
        return 1.0 - (1.0 - cfg.lr_decay_ratio) * progress
    return fn


# ─────────────────────────────────────────────
# Single epoch helpers
# ─────────────────────────────────────────────

def _train_step(
    batch,
    gen, disc, criterion, opt_gen, opt_disc,
    scaler_gen, scaler_disc,
    cfg: TrainConfig,
    device: str,
):
    ir  = batch["ir"].to(device)
    rgb = batch["rgb"].to(device)

    # ── Discriminator ──────────────────────
    opt_disc.zero_grad()
    with autocast(enabled=cfg.mixed_precision):
        fake = gen(ir).detach()
        d_loss = disc_loss_multiscale(disc, ir, rgb, fake)

    scaler_disc.scale(d_loss).backward()
    scaler_disc.unscale_(opt_disc)
    nn.utils.clip_grad_norm_(disc.parameters(), cfg.grad_clip)
    scaler_disc.step(opt_disc)
    scaler_disc.update()

    # ── Generator ──────────────────────────
    opt_gen.zero_grad()
    with autocast(enabled=cfg.mixed_precision):
        fake     = gen(ir)
        fake_pred = [p for p in disc(ir, fake)]   # list of scale outputs
        g_adv    = gen_loss_multiscale(disc, ir, fake)

        # Use the first scale prediction for combined loss GAN term
        g_total, g_log = criterion.generator_loss(
            fake_pred[0], fake, rgb
        )
        # Replace GAN component with multi-scale version
        g_total = g_total - cfg.w_gan * g_log["loss/gan_gen"] + cfg.w_gan * g_adv

    scaler_gen.scale(g_total).backward()
    scaler_gen.unscale_(opt_gen)
    nn.utils.clip_grad_norm_(gen.parameters(), cfg.grad_clip)
    scaler_gen.step(opt_gen)
    scaler_gen.update()

    return {
        "loss/disc": d_loss.item(),
        **{k: v for k, v in g_log.items()},
    }


@torch.no_grad()
def _val_step(batch, gen, criterion, device, cfg):
    ir  = batch["ir"].to(device)
    rgb = batch["rgb"].to(device)
    with autocast(enabled=cfg.mixed_precision):
        fake = gen(ir)
        _, g_log = criterion.generator_loss(
            torch.zeros(ir.shape[0], 1, 1, 1, device=device),
            fake, rgb
        )
    return g_log["loss/gen_total"], fake


# ─────────────────────────────────────────────
# Main trainer
# ─────────────────────────────────────────────

def train(cfg: TrainConfig, resume: Optional[str] = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # ── Setup ──────────────────────────────
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(cfg.log_dir)
    with open(Path(cfg.checkpoint_dir) / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # ── Data ───────────────────────────────
    loaders = get_dataloaders(
        cfg.manifest_path,
        batch_size  = cfg.batch_size,
        num_workers = cfg.num_workers,
        image_size  = cfg.image_size,
        enhance     = True,
    )

    # ── Models ─────────────────────────────
    gen  = ResidualUNetGenerator(in_channels=1, out_channels=3, base_ch=cfg.base_ch).to(device)
    disc = MultiScaleDiscriminator(in_channels=4, base_ch=cfg.base_ch, n_scales=cfg.n_scales).to(device)

    weights = LossWeights(
        gan=cfg.w_gan, l1=cfg.w_l1,
        perceptual=cfg.w_perceptual, ssim=cfg.w_ssim, edge=cfg.w_edge,
    )
    criterion = CombinedLoss(weights, device=device).to(device)

    # ── Optimizers ─────────────────────────
    opt_gen  = torch.optim.Adam(gen.parameters(),  lr=cfg.lr_gen,  betas=(cfg.beta1, cfg.beta2))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=cfg.lr_disc, betas=(cfg.beta1, cfg.beta2))

    lam         = build_lr_lambda(cfg)
    sched_gen   = torch.optim.lr_scheduler.LambdaLR(opt_gen,  lr_lambda=lam)
    sched_disc  = torch.optim.lr_scheduler.LambdaLR(opt_disc, lr_lambda=lam)

    scaler_gen  = GradScaler(enabled=cfg.mixed_precision)
    scaler_disc = GradScaler(enabled=cfg.mixed_precision)

    # ── Resume ─────────────────────────────
    start_epoch = 0
    best_val    = float("inf")
    if resume:
        start_epoch, best_val = load_checkpoint(
            resume, gen, disc, opt_gen, opt_disc, scaler_gen, scaler_disc, device
        )
        print(f"Resumed from {resume}  (epoch {start_epoch})")

    early_stop = EarlyStopping(cfg.patience, cfg.min_delta)

    # ── Grab fixed val batch for visualisation ──  
    fixed_val = None

    if len(loaders["val"].dataset) > 0:
        val_iter = iter(loaders["val"])
        fixed_val = next(val_iter)

    # ─────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────
    for epoch in range(start_epoch, cfg.n_epochs):
        gen.train();  disc.train()
        t0 = time.time()
        epoch_logs: dict = {}

        # Train
        for step, batch in enumerate(loaders["train"]):

            if step % 50 == 0:
                progress = 100 * step / len(loaders["train"])

                print(
            f"Epoch {epoch+1}/{cfg.n_epochs} | "
            f"Batch {step}/{len(loaders['train'])} | "
            f"{progress:.2f}%"
        )
            logs = _train_step(
                batch, gen, disc, criterion,
                opt_gen, opt_disc,
                scaler_gen, scaler_disc, cfg, device,
            )
            for k, v in logs.items():
                epoch_logs[k] = epoch_logs.get(k, 0.0) + v

        n_steps = len(loaders["train"])
        for k in epoch_logs:
            epoch_logs[k] /= n_steps

        # Val
        gen.eval();  disc.eval()
        val_losses = []
        with torch.no_grad():
            for batch in loaders["val"]:
                vl, _ = _val_step(batch, gen, criterion, device, cfg)
                val_losses.append(vl)
        if len(val_losses) > 0:
            val_loss = sum(val_losses) / len(val_losses)
        else:
            val_loss = epoch_logs.get("loss/gen_total", 0)

        sched_gen.step();  sched_disc.step()
        elapsed = time.time() - t0

        # Log to TensorBoard
        for k, v in epoch_logs.items():
            writer.add_scalar(k, v, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr/gen",  opt_gen.param_groups[0]["lr"], epoch)

        # Visualise fixed val batch
        if epoch % 5 == 0 and fixed_val is not None:
            with torch.no_grad():
                _, fakes = _val_step(fixed_val, gen, criterion, device, cfg)
            grid = vutils.make_grid(
                torch.cat([
                    fixed_val["ir"][:cfg.val_vis_n].expand(-1, 3, -1, -1),
                    fakes[:cfg.val_vis_n],
                    fixed_val["rgb"][:cfg.val_vis_n],
                ], dim=0),
                nrow=cfg.val_vis_n, normalize=True, value_range=(-1, 1),
            )
            writer.add_image("val/ir_fake_real", grid, epoch)

        # Console output
        print(
            f"Epoch [{epoch+1:04d}/{cfg.n_epochs}]  "
            f"D:{epoch_logs.get('loss/disc',0):.4f}  "
            f"G:{epoch_logs.get('loss/gen_total',0):.4f}  "
            f"Val:{val_loss:.4f}  "
            f"({elapsed:.1f}s)"
        )

        # Checkpoint
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
        if (epoch + 1) % cfg.save_every == 0 or is_best:
            save_checkpoint(
                epoch + 1, gen, disc, opt_gen, opt_disc,
                scaler_gen, scaler_disc, val_loss, is_best,
                cfg.checkpoint_dir,
            )

        # Early stopping
        if early_stop.step(val_loss):
            print(f"⏹  Early stopping at epoch {epoch+1} (no improvement for {cfg.patience} epochs).")
            break

    writer.close()
    print(f"\n✅ Training complete.  Best val loss: {best_val:.5f}")
    return gen


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",   default="dataset_patches/manifest.json")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--log_dir",    default="runs/ir_colorization")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch",      type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--resume",     default=None)
    args = parser.parse_args()

    cfg = TrainConfig(
        manifest_path  = args.manifest,
        checkpoint_dir = args.checkpoint_dir,
        log_dir        = args.log_dir,
        n_epochs       = args.epochs,
        batch_size     = args.batch,
        lr_gen         = args.lr,
        lr_disc        = args.lr,
    )
    train(cfg, resume=args.resume)
