"""
Training engine for ST-GCN action recognition.
"""

import csv
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils.checkpoint import save_checkpoint
from src.utils.visualize import visualize_batch

# ──────────────────────────────────────────────────────────────────────────────
# YOLO-style colour helpers (ANSI — works on most modern terminals)
# ──────────────────────────────────────────────────────────────────────────────
_BOLD  = "\033[1m"
_RESET = "\033[0m"
_GREEN = "\033[32m"
_CYAN  = "\033[36m"
_YELLOW = "\033[33m"
_BLUE  = "\033[34m"
_RED   = "\033[31m"


def _col(text, color):
    return f"{color}{text}{_RESET}"


def _header_line(epoch, num_epochs, num_classes):
    """Print the YOLO-style header row before each epoch."""
    tag = _col(f"ST-GCN", _BOLD + _BLUE)
    ep  = _col(f"Epoch {epoch}/{num_epochs}", _BOLD + _CYAN)
    sep = _col("─" * 70, _YELLOW)
    print(f"\n{sep}")
    print(f"  {tag}  {ep}  classes={num_classes}")
    print(_col("─" * 70, _YELLOW))


class Trainer:
    """End-to-end training and validation loop.

    Args:
        model: The ST-GCN model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: LR scheduler (or None).
        device: 'cuda' or 'cpu'.
        output_dir: Directory for checkpoints and TensorBoard logs.
        logger: Python logger instance.
        num_epochs: Total training epochs.
        save_every: Save a periodic checkpoint every N epochs.
        use_amp: Use automatic mixed precision (fp16).
    """

    def __init__(self,
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 criterion: nn.Module,
                 optimizer,
                 scheduler=None,
                 device: str = 'cuda',
                 output_dir: str = 'runs/exp',
                 logger=None,
                 num_epochs: int = 80,
                 save_every: int = 10,
                 use_amp: bool = False,
                 vis_every: int = 10,
                 class_names: list | None = None):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.save_every = save_every
        self.logger = logger

        # Mixed precision
        self.use_amp = use_amp and device != 'cpu'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # Visualisation
        self.vis_every   = vis_every
        self.class_names = class_names or []

        # Output
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'tensorboard'))

        # Tracking
        self.best_acc = 0.0
        self.start_epoch = 0
        self.history = []

        # History CSV
        self.history_path = self.output_dir / 'history.csv'
        self._init_history_csv()

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _init_history_csv(self):
        """Create or overwrite the history CSV with header."""
        with open(self.history_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'train_acc',
                'val_loss', 'val_acc', 'lr', 'time_sec', 'is_best'
            ])

    def _append_history(self, row: dict):
        """Append one epoch row to history list and CSV file."""
        self.history.append(row)
        with open(self.history_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                row['epoch'], f"{row['train_loss']:.4f}", f"{row['train_acc']:.2f}",
                f"{row['val_loss']:.4f}", f"{row['val_acc']:.2f}",
                f"{row['lr']:.6f}", f"{row['time_sec']:.1f}", row['is_best']
            ])

    def fit(self):
        """Run the full training loop."""
        # ── Startup banner ─────────────────────────────────────
        num_classes = getattr(self.model, 'num_class',
                              getattr(self.model, 'num_classes', '?'))
        n_train = len(self.train_loader.dataset)
        n_val   = len(self.val_loader.dataset)
        banner_items = [
            ("Device",   self.device),
            ("Train",    f"{n_train} samples ({len(self.train_loader)} batches)"),
            ("Val",      f"{n_val} samples ({len(self.val_loader)} batches)"),
            ("Epochs",   self.num_epochs),
            ("Output",   str(self.output_dir)),
            ("AMP",      self.use_amp),
        ]
        print(_col("═" * 70, _BOLD + _BLUE))
        print(_col(f"  ST-GCN Action Recognition — Training", _BOLD + _CYAN))
        print(_col("═" * 70, _BOLD + _BLUE))
        for k, v in banner_items:
            print(f"  {_col(k + ':',  _YELLOW):<22} {v}")
        print(_col("═" * 70, _BOLD + _BLUE))
        self._log(f"Starting training for {self.num_epochs} epochs on {self.device}")

        # ── Epoch-0 data sanity visualisation (no predictions) ────────
        self._visualize_samples(epoch=0)

        for epoch in range(self.start_epoch, self.num_epochs):
            epoch_start = time.time()

            # ── YOLO-style epoch header ────────────────────────
            _header_line(epoch + 1, self.num_epochs, num_classes)

            # ── Train ──────────────────────────────────────────
            train_loss, train_acc = self._train_one_epoch(epoch)

            # ── Validate ───────────────────────────────────────
            val_loss, val_acc = self._validate(epoch)

            # ── Scheduler step ─────────────────────────────────
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                # ReduceLROnPlateau needs metric
                if hasattr(self.scheduler, 'is_better'):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()

            elapsed = time.time() - epoch_start

            # ── YOLO-style epoch summary row ───────────────────
            is_best_now = val_acc > self.best_acc
            best_marker = _col(" ★ best", _GREEN + _BOLD) if is_best_now else ""
            print(
                f"  {_col('Results:', _BOLD + _YELLOW)}  "
                f"loss={_col(f'{train_loss:.4f}', _CYAN)}/"
                f"{_col(f'{val_loss:.4f}', _CYAN)}  "
                f"acc={_col(f'{train_acc:.1f}%', _GREEN)}/"
                f"{_col(f'{val_acc:.1f}%', _GREEN)}  "
                f"lr={_col(f'{current_lr:.2e}', _YELLOW)}  "
                f"time={_col(f'{elapsed:.0f}s', _BLUE)}"
                f"{best_marker}"
            )
            self._log(
                f"Epoch [{epoch+1}/{self.num_epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% | "
                f"lr={current_lr:.6f} | time={elapsed:.1f}s"
            )

            self.writer.add_scalars('Loss', {
                'train': train_loss, 'val': val_loss
            }, epoch)
            self.writer.add_scalars('Accuracy', {
                'train': train_acc, 'val': val_acc
            }, epoch)
            self.writer.add_scalar('LR', current_lr, epoch)

            # ── Checkpoint ─────────────────────────────────────
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
                self._log(f"  ★ New best val accuracy: {self.best_acc:.2f}%")
                print(_col(f"  ★ New best val accuracy: {self.best_acc:.2f}%", _GREEN + _BOLD))

            state = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_acc': self.best_acc,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }

            # Always save the latest state as checkpoint.pt
            save_checkpoint(state, str(self.output_dir / 'checkpoints'),
                            filename='checkpoint.pt')

            # Save a copy as best.pt if it's the best so far
            if is_best:
                save_checkpoint(state, str(self.output_dir / 'checkpoints'),
                                is_best=True)

            # ── History log ─────────────────────────────────────
            self._append_history({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': current_lr,
                'time_sec': elapsed,
                'is_best': is_best,
            })

            # ── Periodic visualisation with model predictions ─────────
            vis_epoch = epoch + 1  # 1-indexed for display
            if self.vis_every > 0 and vis_epoch % self.vis_every == 0:
                self._visualize_samples(epoch=vis_epoch, with_predictions=True)

        self.writer.close()
        self._log(f"Training complete. Best val accuracy: {self.best_acc:.2f}%")
        print(_col("═" * 70, _BOLD + _GREEN))
        print(_col(f"  Training complete! Best val accuracy: {self.best_acc:.2f}%", _BOLD + _GREEN))
        print(_col("═" * 70, _BOLD + _GREEN))
        return self.best_acc

    def _train_one_epoch(self, epoch: int):
        """Run one training epoch.

        Returns:
            (avg_loss, accuracy_percent)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.train_loader,
            desc=_col("  train", _BOLD + _CYAN),
            ncols=100,
            unit="batch",
            colour="cyan",
            dynamic_ncols=True,
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| "
                "{n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}] "
                "{postfix}"
            ),
            leave=False,
        )

        for batch_idx, (data, label) in enumerate(pbar):
            data = data.float().to(self.device)
            label = label.long().to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    output = self.model(data)
                    loss = self.criterion(output, label)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

            # Live metrics on the progress bar
            avg_loss_so_far = total_loss / total
            acc_so_far = 100.0 * correct / total
            pbar.set_postfix(
                loss=f"{avg_loss_so_far:.4f}",
                acc=f"{acc_so_far:.1f}%",
                refresh=False,
            )

        pbar.close()
        avg_loss = total_loss / total if total > 0 else 0
        accuracy = 100.0 * correct / total if total > 0 else 0
        return avg_loss, accuracy

    @torch.no_grad()
    def _validate(self, epoch: int):
        """Run validation.

        Returns:
            (avg_loss, accuracy_percent)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.val_loader,
            desc=_col("  val  ", _BOLD + _GREEN),
            ncols=100,
            unit="batch",
            colour="green",
            dynamic_ncols=True,
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| "
                "{n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}] "
                "{postfix}"
            ),
            leave=False,
        )

        for data, label in pbar:
            data = data.float().to(self.device)
            label = label.long().to(self.device)

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    output = self.model(data)
                    loss = self.criterion(output, label)
            else:
                output = self.model(data)
                loss = self.criterion(output, label)

            total_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

            avg_loss_so_far = total_loss / total
            acc_so_far = 100.0 * correct / total
            pbar.set_postfix(
                loss=f"{avg_loss_so_far:.4f}",
                acc=f"{acc_so_far:.1f}%",
                refresh=False,
            )

        pbar.close()
        avg_loss = total_loss / total if total > 0 else 0
        accuracy = 100.0 * correct / total if total > 0 else 0
        return avg_loss, accuracy

    # ──────────────────────────────────────────────────────────────────
    # Visualisation helpers
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _visualize_samples(
        self,
        epoch: int,
        n_samples: int = 16,
        n_cols: int = 4,
        with_predictions: bool = False,
    ):
        """Render a grid of skeleton samples and log to disk + TensorBoard.

        At epoch 0, only ground-truth labels are shown (data sanity check).
        When `with_predictions=True` the model is run in eval mode and
        predicted labels + confidence are overlaid on each cell.

        Args:
            epoch:            Epoch number used in filenames.
            n_samples:        Max samples to visualise per split.
            n_cols:           Grid column count.
            with_predictions: Whether to run the model for prediction overlay.
        """
        vis_dir = self.output_dir / 'samples'
        tag     = "epoch 0 — data check" if epoch == 0 else f"epoch {epoch}"
        print(_col(f"\n  [vis] Generating sample visualisation ({tag}) …", _CYAN))

        for split, loader in [("train", self.train_loader), ("val", self.val_loader)]:
            # Grab one batch
            data_batch, label_batch = next(iter(loader))
            data_np  = data_batch.numpy()                    # (N, C, T, V, M)
            label_np = label_batch.numpy()                   # (N,)

            pred_labels = pred_probs = None
            if with_predictions and len(self.class_names) > 0:
                self.model.eval()
                logits = self.model(data_batch.float().to(self.device))  # (N, K)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()      # (N, K)
                pred_labels = probs.argmax(axis=1)                        # (N,)
                pred_probs  = probs.max(axis=1)                           # (N,)

            save_path = visualize_batch(
                data       = data_np,
                labels     = label_np,
                class_names= self.class_names,
                out_path   = vis_dir,
                n_samples  = n_samples,
                n_cols     = n_cols,
                epoch      = epoch,
                split      = split,
                pred_labels= pred_labels,
                pred_probs = pred_probs,
            )

            # ── Log to TensorBoard (Images tab) ───────────────────────
            try:
                import cv2
                img_bgr = cv2.imread(str(save_path))
                if img_bgr is not None:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    # TensorBoard expects (C, H, W) in range [0, 1]
                    img_t   = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
                    tb_tag  = f"samples/{split}"
                    self.writer.add_image(tb_tag, img_t, global_step=epoch)
                    self.writer.flush()
            except Exception as e:
                self._log(f"[vis] TensorBoard image log failed: {e}")

            marker = _col("✓", _GREEN + _BOLD)
            print(f"  {marker} {_col(split, _BOLD)} → {_col(str(save_path), _CYAN)}")

        print()

    def resume(self, checkpoint_path: str):
        """Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to the .pt checkpoint file.
        """
        from src.utils.checkpoint import load_checkpoint
        info = load_checkpoint(checkpoint_path, self.model,
                               self.optimizer, device=self.device)
        self.start_epoch = info['epoch']
        self.best_acc = info['best_acc']
        self._log(f"Resumed from epoch {self.start_epoch}, best_acc={self.best_acc:.2f}%")
