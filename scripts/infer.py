"""
Real-time / offline inference for ST-GCN Action Recognition.

Supports two input modes:
  • Video file  : python scripts/infer.py --source video.mp4
  • Webcam      : python scripts/infer.py --source 0

Sliding-window logic
────────────────────
Frames are fed one-by-one to a ring buffer of size `window_size`
(=max_frames used during training, default 300).
Once the buffer is full an prediction is emitted every `stride` frames:

    frames  1 … 300  → predict
    frames  2 … 301  → predict (stride=1)
    …

Usage examples
──────────────
    # From a video file (shows overlay window + saves output)
    python scripts/infer.py --source data/test.mp4 --checkpoint runs/exp/checkpoints/best.pt

    # Webcam (index 0)
    python scripts/infer.py --source 0 --checkpoint runs/exp/checkpoints/best.pt

    # Custom window / stride
    python scripts/infer.py --source 0 --window 150 --stride 5

    # No display (headless / server), save output only
    python scripts/infer.py --source video.mp4 --no-display --save
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import STGCNModel, STGCNTwoStreamModel
from src.data.skeleton_extractor import SkeletonExtractor, NUM_KEYPOINTS
from src.inference.predictor import SlidingWindowPredictor


# ─────────────────────────────────────────────────────────────────────────────
# ANSI colours (same palette as trainer / preprocess)
# ─────────────────────────────────────────────────────────────────────────────
_B  = "\033[1m"
_R  = "\033[0m"
_G  = "\033[32m"
_C  = "\033[36m"
_Y  = "\033[33m"
_BL = "\033[34m"
_RD = "\033[31m"


def _col(text, color): return f"{color}{text}{_R}"

# ─────────────────────────────────────────────────────────────────────────────
# COCO skeleton connections for drawing
# ─────────────────────────────────────────────────────────────────────────────
COCO_EDGES = [
    (0,  1), (0,  2),             # nose → eyes
    (1,  3), (2,  4),             # eyes → ears
    (5,  6),                      # shoulders
    (5,  7), (7,  9),             # left arm
    (6,  8), (8, 10),             # right arm
    (5, 11), (6, 12),             # torso
    (11, 12),                     # hips
    (11, 13), (13, 15),           # left leg
    (12, 14), (14, 16),           # right leg
]

# Colour palette for different action classes (BGR)
_PALETTE = [
    (0,   200, 255),   # 0 – falling    orange-ish
    (0,   255, 127),   # 1 – sitting    green
    (255, 200,   0),   # 2 – standing   cyan
    (200, 100, 255),   # 3 – walking    purple
    (255,  60,  60),   # 4 – phone-walk red
    (200, 200, 200),   # 5+ – fallback  grey
]


def _class_color(idx: int):
    return _PALETTE[min(idx, len(_PALETTE) - 1)]


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ST-GCN Real-Time Inference (Sliding Window)")
    p.add_argument("--source",     type=str,   default="0",
                   help="Video file path or webcam index (default: 0)")
    p.add_argument("--checkpoint", type=str,   default=None,
                   help="Path to model checkpoint (.pt).  "
                        "Defaults to runs/exp/checkpoints/best.pt")
    p.add_argument("--config",     type=str,   default="configs/base.yaml",
                   help="Path to YAML config (default: configs/base.yaml)")
    p.add_argument("--window",     type=int,   default=None,
                   help="Sliding-window size in frames (overrides config max_frames)")
    p.add_argument("--stride",     type=int,   default=1,
                   help="Run inference every N frames (default: 1)")
    p.add_argument("--smooth",     type=float, default=0.5,
                   help="EMA smoothing factor for predictions 0‥1 (default: 0.5)")
    p.add_argument("--conf",       type=float, default=0.3,
                   help="YOLO pose confidence threshold (default: 0.3)")
    p.add_argument("--device",     type=str,   default="auto",
                   help="Inference device: auto | cuda | cpu (default: auto)")
    p.add_argument("--save",       action="store_true",
                   help="Save annotated output video")
    p.add_argument("--no-display", action="store_true", dest="no_display",
                   help="Suppress the OpenCV preview window")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, cfg: dict, device: str) -> torch.nn.Module:
    """Instantiate the model from config and load checkpoint weights."""
    model_cfg  = cfg["model"]
    num_classes = cfg["data"]["num_classes"]

    kwargs = {}
    if model_cfg.get("dropout", 0) > 0:
        kwargs["dropout"] = model_cfg["dropout"]

    model_type = model_cfg.get("type", "stgcn")
    if model_type == "stgcn":
        model = STGCNModel(
            in_channels=model_cfg["in_channels"],
            num_class=num_classes,
            graph_args=model_cfg["graph_args"],
            edge_importance_weighting=model_cfg["edge_importance_weighting"],
            **kwargs,
        )
    elif model_type == "stgcn_twostream":
        model = STGCNTwoStreamModel(
            in_channels=model_cfg["in_channels"],
            num_class=num_classes,
            graph_args=model_cfg["graph_args"],
            edge_importance_weighting=model_cfg["edge_importance_weighting"],
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, color, alpha=0.85):
    """Draw COCO-17 skeleton on frame in-place.

    Args:
        keypoints: (17, 3) array — (x, y, conf).
        color:     (B, G, R) tuple.
    """
    h, w = frame.shape[:2]
    # Draw edges
    for i, j in COCO_EDGES:
        x1, y1, c1 = keypoints[i]
        x2, y2, c2 = keypoints[j]
        if c1 < 0.1 or c2 < 0.1:
            continue
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)
    # Draw keypoints
    for i in range(NUM_KEYPOINTS):
        x, y, c = keypoints[i]
        if c < 0.1:
            continue
        cv2.circle(frame, (int(x), int(y)), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (int(x), int(y)), 4, color, 1,  cv2.LINE_AA)


def _draw_hud(
    frame:       np.ndarray,
    result:      dict,
    class_names: list,
    fps:         float,
    window_size: int,
):
    """Draw the YOLO-style heads-up display on the frame."""
    h, w = frame.shape[:2]
    ready      = result["ready"]
    label      = result["label"]
    confidence = result["confidence"]
    probs      = result["probs"]
    frame_idx  = result["frame_idx"]
    pred_idx   = class_names.index(label) if label in class_names else 0

    color = _class_color(pred_idx)

    # ── Top-left panel ────────────────────────────────────────────────────
    panel_w, panel_h = 280, 130
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Model name
    cv2.putText(frame, "ST-GCN  Action Recognition",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (180, 180, 180), 1, cv2.LINE_AA)

    # Frame / buffer info
    buf_filled = min(frame_idx, window_size)
    buf_pct    = buf_filled / window_size
    cv2.putText(frame,
                f"Buffer: {buf_filled}/{window_size}",
                (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (180, 180, 180), 1, cv2.LINE_AA)

    # Buffer fill bar
    bar_x, bar_y, bar_w, bar_h = 10, 70, panel_w - 20, 6
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + int(bar_w * buf_pct), bar_y + bar_h),
                  (0, 200, 120) if ready else (0, 140, 255), -1)

    if not ready:
        cv2.putText(frame, "Filling buffer…",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (0, 200, 255), 1, cv2.LINE_AA)
        return

    # ── Action label ──────────────────────────────────────────────────────
    label_disp = label.replace("_", " ").title()
    cv2.putText(frame, f"{label_disp}",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{confidence*100:.1f}%",
                (10, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (200, 200, 200), 1, cv2.LINE_AA)

    # ── Per-class probability bars (right side) ───────────────────────────
    if probs is not None:
        bx, by = w - 210, 10
        for i, (cls, p) in enumerate(zip(class_names, probs)):
            bar_len = int(180 * p)
            is_top  = (i == pred_idx)
            bcolor  = _class_color(i)
            cv2.rectangle(frame, (bx, by + i*22),
                          (bx + bar_len, by + i*22 + 14), bcolor, -1)
            cv2.putText(frame,
                        f"{cls[:18]:<18}  {p*100:5.1f}%",
                        (bx - 5, by + i*22 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 255, 255) if is_top else (170, 170, 170),
                        1 if not is_top else 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Main inference loop
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    # ── Config ────────────────────────────────────────────────────────────
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    class_names = cfg["data"].get("class_names", [])
    window_size = args.window or cfg["data"].get("max_frames", 300)
    checkpoint  = args.checkpoint or str(
        Path(cfg["output"]["dir"]) / "checkpoints" / "best.pt"
    )

    # ── Device ────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # ── Startup banner ────────────────────────────────────────────────────
    print(_col("═" * 70, _B + _BL))
    print(_col("  ST-GCN  Real-Time Inference  (Sliding Window)", _B + _C))
    print(_col("═" * 70, _B + _BL))
    items = [
        ("Source",      args.source),
        ("Checkpoint",  checkpoint),
        ("Device",      device),
        ("Window",      f"{window_size} frames"),
        ("Stride",      f"every {args.stride} frame(s)"),
        ("Smoothing",   f"EMA α={args.smooth}"),
        ("Classes",     ", ".join(class_names)),
    ]
    for k, v in items:
        print(f"  {_col(k + ':', _Y):<22} {v}")
    print(_col("═" * 70, _B + _BL))

    # ── Load models ───────────────────────────────────────────────────────
    print(_col("  Loading ST-GCN checkpoint …", _C))
    model = load_model(checkpoint, cfg, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(_col(f"  ST-GCN ready  ({total_params:,} params)", _G + _B))

    print(_col("  Loading YOLOv11n-pose …", _C))
    extractor = SkeletonExtractor(conf_threshold=args.conf, device=device)
    print(_col("  YOLOv11n-pose ready", _G + _B))

    # ── Predictor ─────────────────────────────────────────────────────────
    predictor = SlidingWindowPredictor(
        model=model,
        class_names=class_names,
        window_size=window_size,
        stride=args.stride,
        smooth_alpha=args.smooth,
        device=device,
        normalize=cfg["dataloader"].get("normalize", True),
        conf_threshold=args.conf,
    )

    # ── Video source ──────────────────────────────────────────────────────
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(_col(f"[ERROR] Cannot open source: {args.source}", _RD + _B))
        return

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(_col(f"  Source: {W}×{H} @ {fps_src:.1f} fps", _C))
    print()
    print(_col("  Press  q  to quit", _Y))
    print()

    # ── Video writer (optional) ───────────────────────────────────────────
    writer = None
    if args.save:
        out_name = f"infer_{Path(str(args.source)).stem}.mp4"
        out_path = Path("runs") / "infer" / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_src, (W, H))
        print(_col(f"  Saving output to {out_path}", _C))

    # ── FPS tracking ──────────────────────────────────────────────────────
    fps_times: deque = deque(maxlen=30)
    display_fps = 0.0

    # ── Main loop ─────────────────────────────────────────────────────────
    try:
        while True:
            t0 = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                # Video file ended — loop or exit
                if isinstance(src, str):
                    break
                continue

            # ── YOLO pose estimation ─────────────────────────────────────
            results = extractor.model.predict(
                frame,
                device=extractor.device,
                conf=extractor.conf_threshold,
                verbose=False,
            )
            keypoints = extractor._pick_best_person(results[0])  # (17, 3)

            # ── Predictor ────────────────────────────────────────────────
            result = predictor.push_frame(keypoints)

            # ── Draw ─────────────────────────────────────────────────────
            pred_idx = (class_names.index(result["label"])
                        if result["label"] in class_names else 0)
            skel_color = _class_color(pred_idx)
            _draw_skeleton(frame, keypoints, color=skel_color)
            _draw_hud(frame, result, class_names, display_fps, window_size)

            # ── FPS ──────────────────────────────────────────────────────
            fps_times.append(time.perf_counter() - t0)
            display_fps = 1.0 / (sum(fps_times) / len(fps_times))

            # Terminal log
            if result["ready"] and result["frame_idx"] % 30 == 0:
                label = result["label"].replace("_", " ")
                conf  = result["confidence"] * 100
                print(
                    f"  frame {result['frame_idx']:6d}  "
                    f"{_col(f'{label:<26}', _G + _B)}  "
                    f"conf={_col(f'{conf:5.1f}%', _C)}  "
                    f"fps={_col(f'{display_fps:5.1f}', _Y)}"
                )

            # ── Save / Display ───────────────────────────────────────────
            if writer is not None:
                writer.write(frame)

            if not args.no_display:
                cv2.imshow("ST-GCN Inference", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n" + _col("  Interrupted by user.", _Y))

    # ── Cleanup ───────────────────────────────────────────────────────────
    cap.release()
    if writer is not None:
        writer.release()
        print(_col(f"\n  Saved output to {out_path}", _G))
    if not args.no_display:
        cv2.destroyAllWindows()

    print(_col("\n  Done.", _G + _B))


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(parse_args())
