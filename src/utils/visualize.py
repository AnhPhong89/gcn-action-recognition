"""
Skeleton sample visualization — YOLO-style training data inspection.

Generates a grid image of skeleton motion trails from a DataLoader batch,
with class labels and optional model prediction overlays (per-epoch check).

Outputs:
    runs/exp/samples/train_samples_epoch000.png
    runs/exp/samples/val_samples_epoch000.png
    runs/exp/samples/train_samples_epoch010.png  ← with model predictions
    ...
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# COCO-17 skeleton edges (matching graph.py 'coco' layout)
# ---------------------------------------------------------------------------
COCO_EDGES = [
    (0, 1), (0, 2),        # nose ↔ eyes
    (1, 3), (2, 4),        # eyes ↔ ears
    (0, 5), (0, 6),        # nose ↔ shoulders
    (5, 7), (7, 9),        # left arm
    (6, 8), (8, 10),       # right arm
    (5, 11), (6, 12),      # torso
    (11, 13), (13, 15),    # left leg
    (12, 14), (14, 16),    # right leg
    (5, 6), (11, 12),      # shoulder span + hip span
]

# ---------------------------------------------------------------------------
# Colour palette — one colour per class (RGB 0-255)
# Matches infer.py _PALETTE (converted from BGR to RGB)
# ---------------------------------------------------------------------------
_CLASS_COLORS_RGB = [
    (255, 200,   0),   # 0 – falling    orange
    (127, 255,   0),   # 1 – sitting    green
    (  0, 200, 255),   # 2 – standing   cyan
    (255, 100, 200),   # 3 – walking    purple
    ( 60,  60, 255),   # 4 – phone-walk red/blue
    (200, 200, 200),   # 5+ – fallback  grey
]

_BG_COLOR      = (18,  18,  28)   # dark navy background
_TRAIL_FRAMES  = 16               # max frames to sample for motion trail
_CELL_SIZE     = 224              # pixels per cell (square)
_FONT_SCALE    = 0.48
_FONT_THICK    = 1


def _class_color(idx: int):
    return _CLASS_COLORS_RGB[min(idx, len(_CLASS_COLORS_RGB) - 1)]


def _draw_skeleton_trail(
    canvas: "np.ndarray",
    skel: "np.ndarray",
    color_rgb: tuple,
    cell_w: int,
    cell_h: int,
):
    """Overlay skeleton motion trail onto a cell canvas (H, W, 3) uint8.

    Args:
        skel:  (C, T, V, M) float array. C[0]=x, C[1]=y. Values may be
               normalised (centred around 0) or raw pixel coords.
    """
    import cv2

    C, T, V, M = skel.shape
    if T == 0:
        return

    # Sample evenly-spaced frames up to _TRAIL_FRAMES
    indices = np.linspace(0, T - 1, min(T, _TRAIL_FRAMES), dtype=int)

    # Decide coordinate range — normalised coords are typically in [-3, 3]
    # Raw pixel coords are >> 1.  We rescale to fit the cell.
    xy = skel[:2, :, :, 0]                 # (2, T, V)  – first person only

    x_all = xy[0].flatten()
    y_all = xy[1].flatten()
    x_all = x_all[np.abs(x_all) > 1e-6]
    y_all = y_all[np.abs(y_all) > 1e-6]

    if len(x_all) == 0 or len(y_all) == 0:
        return

    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()
    span = max(x_max - x_min, y_max - y_min, 1e-4)

    # Padding so skeleton doesn't touch edges
    pad = 0.12

    def to_pixel(x, y):
        px = int((x - x_min) / span * cell_w * (1 - 2 * pad) + cell_w * pad)
        py = int((y - y_min) / span * cell_h * (1 - 2 * pad) + cell_h * pad)
        return px, py

    n_frames = len(indices)
    for fi, t in enumerate(indices):
        # Opacity: oldest frame most transparent → newest fully opaque
        alpha = 0.15 + 0.85 * (fi / max(n_frames - 1, 1))
        col = tuple(int(c * alpha) for c in color_rgb)

        kpts = skel[:2, t, :, 0]  # (2, V)
        pts  = [to_pixel(kpts[0, v], kpts[1, v]) for v in range(V)]

        # Draw edges
        for i, j in COCO_EDGES:
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            # Skip if either endpoint is at origin (missing keypoint)
            if skel[0, t, i, 0] == 0 and skel[1, t, i, 0] == 0:
                continue
            if skel[0, t, j, 0] == 0 and skel[1, t, j, 0] == 0:
                continue
            cv2.line(canvas, (x1, y1), (x2, y2), col, 1, cv2.LINE_AA)

        # Draw keypoints only for the last frame
        if fi == n_frames - 1:
            for v in range(V):
                if skel[0, t, v, 0] == 0 and skel[1, t, v, 0] == 0:
                    continue
                px, py = pts[v]
                cv2.circle(canvas, (px, py), 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, (px, py), 3, color_rgb,         1, cv2.LINE_AA)


def _draw_label_bar(
    canvas: "np.ndarray",
    true_name: str,
    color_rgb: tuple,
    pred_name: Optional[str] = None,
    pred_conf: Optional[float] = None,
    correct: Optional[bool] = None,
):
    """Draw class label (and optional prediction) on top of a cell."""
    import cv2

    h, w = canvas.shape[:2]

    # Top bar — true label
    bar_h = 22
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), color_rgb, -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    cv2.putText(canvas, true_name,
                (5, bar_h - 6), cv2.FONT_HERSHEY_SIMPLEX,
                _FONT_SCALE, (255, 255, 255), _FONT_THICK, cv2.LINE_AA)

    # Bottom bar — model prediction (optional)
    if pred_name is not None:
        tick_color = (50, 220, 50) if correct else (220, 50, 50)
        tick        = "✓" if correct else "✗"
        conf_str    = f"{pred_conf * 100:.0f}%" if pred_conf is not None else ""
        text        = f"{tick} {pred_name} {conf_str}"
        bot_y       = h - bar_h
        ov2 = canvas.copy()
        cv2.rectangle(ov2, (0, bot_y), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(ov2, 0.65, canvas, 0.35, 0, canvas)
        cv2.putText(canvas, text,
                    (5, h - 7), cv2.FONT_HERSHEY_SIMPLEX,
                    _FONT_SCALE, tick_color, _FONT_THICK, cv2.LINE_AA)


def visualize_batch(
    data: "np.ndarray",
    labels: "np.ndarray",
    class_names: list[str],
    out_path: str | Path,
    *,
    n_samples: int = 16,
    n_cols: int = 4,
    epoch: int = 0,
    split: str = "train",
    pred_labels: Optional["np.ndarray"] = None,
    pred_probs:  Optional["np.ndarray"] = None,
    cell_size: int = _CELL_SIZE,
) -> Path:
    """Build and save a grid of skeleton motion-trail images.

    Args:
        data:        (N, C, T, V, M) numpy array from DataLoader.
        labels:      (N,) integer class indices.
        class_names: List of class name strings.
        out_path:    Directory to save the PNG into.
        n_samples:   How many samples to show (capped to len(data)).
        n_cols:      Number of grid columns.
        epoch:       Current epoch number (used in filename).
        split:       'train' or 'val' (used in filename).
        pred_labels: (N,) predicted class indices (optional).
        pred_probs:  (N,) max softmax probability (optional).
        cell_size:   Pixel size of each grid cell.

    Returns:
        Path of the saved PNG.
    """
    import cv2

    n_samples = min(n_samples, len(data))
    n_rows    = (n_samples + n_cols - 1) // n_cols
    cw = ch   = cell_size

    grid_w = n_cols * cw
    grid_h = n_rows * ch
    grid   = np.full((grid_h, grid_w, 3), _BG_COLOR, dtype=np.uint8)

    for idx in range(n_samples):
        row = idx // n_cols
        col = idx %  n_cols
        y0, x0 = row * ch, col * cw

        cell = np.full((ch, cw, 3), _BG_COLOR, dtype=np.uint8)

        lbl_idx   = int(labels[idx])
        true_name = class_names[lbl_idx] if lbl_idx < len(class_names) else str(lbl_idx)
        color     = _class_color(lbl_idx)

        # Thin border matching class colour
        cv2.rectangle(cell, (0, 0), (cw - 1, ch - 1), color, 2)

        # Draw skeleton trail
        skel = data[idx]  # (C, T, V, M)
        _draw_skeleton_trail(cell, skel, color, cw, ch)

        # Labels
        pred_name = pred_conf = correct = None
        if pred_labels is not None:
            pi = int(pred_labels[idx])
            pred_name = class_names[pi] if pi < len(class_names) else str(pi)
            pred_conf = float(pred_probs[idx]) if pred_probs is not None else None
            correct   = (pi == lbl_idx)

        _draw_label_bar(cell, true_name, color,
                        pred_name=pred_name, pred_conf=pred_conf, correct=correct)

        grid[y0:y0 + ch, x0:x0 + cw] = cell

    # ── Header strip ──────────────────────────────────────────────────────
    header_h = 32
    header   = np.full((header_h, grid_w, 3), (15, 15, 25), dtype=np.uint8)
    tag      = f"ST-GCN  |  {split.upper()} samples  |  Epoch {epoch}"
    cv2.putText(header, tag,
                (10, header_h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (150, 200, 255), 1, cv2.LINE_AA)
    grid = np.vstack([header, grid])

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir  = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{split}_samples_epoch{epoch:03d}.png"
    save_path = out_dir / filename
    cv2.imwrite(str(save_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    return save_path
