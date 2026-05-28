"""
Space LLM Training Script
Trains a decoder-only transformer on space/astronomy data.
Designed for Google Colab with T4 GPU.
"""

import os
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from config import ModelConfig, TrainConfig, save_config
from model.transformer import SpaceLLM


class TokenDataset(Dataset):
    """Memory-mapped token dataset for efficient loading."""

    def __init__(self, data_path: str, seq_len: int):
        meta_path = Path(data_path) / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        self.data = np.memmap(
            Path(data_path) / "train.bin",
            dtype=np.uint16,
            mode="r",
            shape=meta["train_tokens"],
        )
        self.seq_len = seq_len
        self.total_tokens = len(self.data)

    def __len__(self):
        return (self.total_tokens - self.seq_len - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


class ValDataset(Dataset):
    """Validation dataset."""

    def __init__(self, data_path: str, seq_len: int):
        meta_path = Path(data_path) / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        self.data = np.memmap(
            Path(data_path) / "val.bin",
            dtype=np.uint16,
            mode="r",
            shape=meta["val_tokens"],
        )
        self.seq_len = seq_len
        self.total_tokens = len(self.data)

    def __len__(self):
        return max(1, (self.total_tokens - self.seq_len - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = min(start + self.seq_len + 1, self.total_tokens)
        chunk = self.data[start:end].astype(np.int64)
        if len(chunk) < self.seq_len + 1:
            chunk = np.pad(chunk, (0, self.seq_len + 1 - len(chunk)), constant_values=0)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return lr * step / warmup_steps
    if step > max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, val_loader, device, num_batches=20):
    """Estimate validation loss."""
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= num_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss, _ = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(model, optimizer, scheduler, step, val_loss, path):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "val_loss": val_loss,
        "config": model.config.to_dict(),
    }, path)
    print(f"  [Checkpoint] Saved to {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load training checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["step"], ckpt.get("val_loss", float("inf"))


def train(
    model_config: ModelConfig = None,
    train_config: TrainConfig = None,
    resume_from: str = None,
):
    """Main training loop."""
    if model_config is None:
        model_config = ModelConfig()
    if train_config is None:
        train_config = TrainConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Create model
    model = SpaceLLM(model_config).to(device)
    print(f"Parameters: {model.get_num_params():,}")

    # Datasets
    train_dataset = TokenDataset(train_config.data_dir, model_config.max_seq_len)
    val_dataset = ValDataset(train_config.data_dir, model_config.max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(0.9, 0.95),
    )

    # Resume
    start_step = 0
    best_val_loss = float("inf")
    if resume_from and os.path.exists(resume_from):
        start_step, best_val_loss = load_checkpoint(resume_from, model, optimizer)
        print(f"Resumed from step {start_step}, val_loss={best_val_loss:.4f}")

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=train_config.fp16)
    autocast_ctx = torch.amp.autocast("cuda", enabled=train_config.fp16)

    # Training state
    model.train()
    step = start_step
    epoch = 0
    total_loss = 0.0
    tokens_processed = 0
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"Training started at step {start_step}")
    print(f"Max steps: {train_config.max_steps}")
    print(f"Effective batch size: {train_config.batch_size * train_config.grad_accum_steps}")
    print(f"{'='*60}\n")

    while step < train_config.max_steps:
        epoch += 1
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= train_config.max_steps:
                break

            x, y = x.to(device), y.to(device)

            # Forward pass with mixed precision
            with autocast_ctx:
                _, loss, _ = model(x, targets=y)
                loss = loss / train_config.grad_accum_steps

            # Backward pass
            scaler.scale(loss).backward()
            total_loss += loss.item() * train_config.grad_accum_steps

            # Gradient accumulation
            if (batch_idx + 1) % train_config.grad_accum_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)

                # Learning rate schedule
                lr = get_lr(
                    step,
                    train_config.warmup_steps,
                    train_config.max_steps,
                    train_config.learning_rate,
                    train_config.min_lr,
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                step += 1
                tokens_processed += train_config.batch_size * model_config.max_seq_len

                # Logging
                if step % train_config.log_every == 0:
                    avg_loss = total_loss / train_config.log_every
                    elapsed = time.time() - start_time
                    tps = tokens_processed / elapsed
                    print(
                        f"Step {step:>6d}/{train_config.max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Tokens/s: {tps:,.0f} | "
                        f"Time: {elapsed:.0f}s"
                    )
                    total_loss = 0.0

                # Evaluation
                if step % train_config.eval_every == 0:
                    val_loss = estimate_loss(model, val_loader, device)
                    print(f"  [Eval] Step {step} | Val Loss: {val_loss:.4f} | Perplexity: {math.exp(val_loss):.2f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            model, optimizer, None, step, val_loss,
                            os.path.join(train_config.checkpoint_dir, "best.pt"),
                        )

                # Save checkpoint
                if step % train_config.save_every == 0:
                    save_checkpoint(
                        model, optimizer, None, step, best_val_loss,
                        os.path.join(train_config.checkpoint_dir, f"step_{step}.pt"),
                    )

    # Final save
    save_checkpoint(
        model, optimizer, None, step, best_val_loss,
        os.path.join(train_config.checkpoint_dir, "final.pt"),
    )

    # Save model config
    save_config(model_config, os.path.join(train_config.checkpoint_dir, "model_config.json"))

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete in {elapsed/3600:.1f} hours")
    print(f"Final val loss: {best_val_loss:.4f}")
    print(f"Final perplexity: {math.exp(best_val_loss):.2f}")
    print(f"Total tokens processed: {tokens_processed:,}")
    print(f"{'='*60}")

    return model


if __name__ == "__main__":
    train()
