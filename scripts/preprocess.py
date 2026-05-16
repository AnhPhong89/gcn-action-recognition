import argparse
import os
import pickle
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
import time

import sys
# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.skeleton_extractor import SkeletonExtractor

# ──────────────────────────────────────────────────────────────────────────────
# ANSI colour helpers (same palette as YOLO / trainer.py)
# ──────────────────────────────────────────────────────────────────────────────
_BOLD   = "\033[1m"
_RESET  = "\033[0m"
_GREEN  = "\033[32m"
_CYAN   = "\033[36m"
_YELLOW = "\033[33m"
_BLUE   = "\033[34m"
_RED    = "\033[31m"


def _col(text, color):
    return f"{color}{text}{_RESET}"


def parse_args():
    parser = argparse.ArgumentParser(description="Extract skeletons and prepare dataset for ST-GCN")
    parser.add_argument('--raw_dir', type=str, default='data/raw', help='Path to raw dataset directory (should contain train/ and val/ subfolders)')
    parser.add_argument('--out_dir', type=str, default='data/processed', help='Path to output processed data')
    parser.add_argument('--max_frames', type=int, default=300, help='Maximum frames to keep per video via padding/truncating')
    parser.add_argument('--yolo', type=str, default='yolo11m-pose.pt', help='Path to YOLO pose model')
    parser.add_argument('--config', type=str, default='configs/base.yaml', help='Path to YAML config file for class names')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold for YOLO pose')
    parser.add_argument('--device', type=str, default='auto', help='Device for YOLO inference (auto, cuda, cpu)')
    # Add an optional limit for debugging/quick runs
    parser.add_argument('--limit', type=int, default=-1, help='Limit number of videos per class (for debugging)')
    return parser.parse_args()


def pad_or_truncate(skeleton: np.ndarray, max_frames: int) -> np.ndarray:
    """
    Pad or truncate skeleton sequence to max_frames along the T axis.
    skeleton shape: (C, T, V, M)
    """
    C, T, V, M = skeleton.shape
    
    if T == max_frames:
        return skeleton
        
    if T > max_frames:
        # Truncate: take the first max_frames.
        return skeleton[:, :max_frames, :, :]
        
    # Pad with zeros at the end
    padded = np.zeros((C, max_frames, V, M), dtype=np.float32)
    padded[:, :T, :, :] = skeleton
    return padded


def process_split(split_name: str, split_dir: Path, class_to_idx: dict, extractor, args) -> tuple:
    """Process a single split (train or val) and return (data, labels, samples)."""
    print(_col(f"\n  Processing '{split_name}' split …", _BOLD + _BLUE))
    
    samples = []   # Tên file/mẫu
    labels = []    # Label index
    skeletons = [] # Numpy arrays (C, T_max, V, M)
    
    total_videos = 0
    
    for cls_name, label_idx in class_to_idx.items():
        cls_dir = split_dir / cls_name
        if not cls_dir.exists():
            continue
            
        video_paths = list(cls_dir.glob("*.mp4")) + list(cls_dir.glob("*.avi"))
        if args.limit > 0:
            video_paths = video_paths[:args.limit]
            
        total_videos += len(video_paths)
        if len(video_paths) == 0:
            continue
            
        ok_count = 0
        err_count = 0

        pbar = tqdm(
            video_paths,
            desc=f"    {_col(cls_name, _CYAN):<28}",
            unit="vid",
            ncols=100,
            colour="cyan",
            dynamic_ncols=True,
            bar_format=(
                "{desc} {percentage:3.0f}%|{bar}| "
                "{n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}] "
                "{postfix}"
            ),
        )

        for p in pbar:
            try:
                # Extract skeleton: (C, T, V, M)
                skel = extractor.extract_from_video(p)
                
                # Cắt hoặc bù frames
                skel_fixed = pad_or_truncate(skel, args.max_frames)
                
                samples.append(p.stem)
                labels.append(label_idx)
                skeletons.append(skel_fixed)
                ok_count += 1
                
            except Exception as e:
                err_count += 1
                tqdm.write(_col(f"  [WARN] {p.name}: {e}", _YELLOW))

            pbar.set_postfix(
                ok=_col(ok_count, _GREEN),
                err=_col(err_count, _RED) if err_count else 0,
                refresh=False,
            )

        pbar.close()
        
    if len(skeletons) == 0:
        return None, None, None
        
    data = np.stack(skeletons, axis=0)
    return data, labels, samples


def main():
    args = parse_args()
    t0 = time.time()
    
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_dir = raw_dir / "train"
    val_dir = raw_dir / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        print(_col(f"[ERROR] Raw directory must contain 'train' and 'val' subfolders.", _RED + _BOLD))
        print(f"Looked in: {raw_dir}")
        return

    # ── Startup banner ──────────────────────────────────────────────────────
    print(_col("═" * 70, _BOLD + _BLUE))
    print(_col("  ST-GCN  Skeleton Preprocessing", _BOLD + _CYAN))
    print(_col("═" * 70, _BOLD + _BLUE))
    cfg_items = [
        ("Raw dir",     raw_dir),
        ("Output dir",  out_dir),
        ("Max frames",  args.max_frames),
        ("Confidence",  args.conf),
        ("Device",      args.device),
    ]
    for k, v in cfg_items:
        print(f"  {_col(k + ':', _YELLOW):<22} {v}")
    print(_col("═" * 70, _BOLD + _BLUE))

    # Load config to get allowed classes
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    allowed_classes = cfg.get('data', {}).get('class_names', [])

    # Lấy danh sách các class từ thư mục train, LỌC theo allowed_classes
    classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
    if allowed_classes:
        classes = [c for c in classes if c in allowed_classes]
        
    classes.sort()  # Sort alphabetically to ensure consistent label indices
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    print(f"\n  {_col('Classes found:', _YELLOW)} {_col(len(classes), _BOLD)} → {classes}")
    print()

    # Load layout from config
    layout = cfg.get('model', {}).get('graph_args', {}).get('layout', 'openpose')
    
    # Khởi tạo extractor
    print(_col(f"  Loading {args.yolo} model (layout: {layout}) …", _CYAN))
    extractor = SkeletonExtractor(model_path=args.yolo, conf_threshold=args.conf, device=args.device, layout=layout)
    print(_col(f"  {args.yolo} ready.", _GREEN + _BOLD))
    
    # ── Process Train Split ────────────────────────────────────────────────
    train_data, train_labels, train_samples = process_split("train", train_dir, class_to_idx, extractor, args)
    
    # ── Process Val Split ──────────────────────────────────────────────────
    val_data, val_labels, val_samples = process_split("val", val_dir, class_to_idx, extractor, args)
    
    # ── Save Results ───────────────────────────────────────────────────────
    print(_col(f"\n  Saving to {out_dir} …", _CYAN))
    
    # Save Train
    if train_data is not None:
        np.save(out_dir / "train_data.npy", train_data)
        with open(out_dir / "train_label.pkl", 'wb') as f:
            pickle.dump((train_samples, list(train_labels)), f)
        print(f"  {_col('Train:', _GREEN)} {len(train_labels)} samples saved.")
    else:
        print(f"  {_col('Train:', _RED)} No data found!")
        
    # Save Val
    if val_data is not None:
        np.save(out_dir / "val_data.npy", val_data)
        with open(out_dir / "val_label.pkl", 'wb') as f:
            pickle.dump((val_samples, list(val_labels)), f)
        print(f"  {_col('Val:', _GREEN)} {len(val_labels)} samples saved.")
    else:
        print(f"  {_col('Val:', _RED)} No data found!")
        
    # Lưu label mapping
    with open(out_dir / "label_map.yaml", 'w') as f:
        yaml.dump(class_to_idx, f)

    elapsed = time.time() - t0
    print(_col("═" * 70, _BOLD + _GREEN))
    train_len = len(train_labels) if train_labels else 0
    val_len = len(val_labels) if val_labels else 0
    print(f"  {_col('Done!', _BOLD + _GREEN)}  "
          f"{_col(train_len, _BOLD)} train + {_col(val_len, _BOLD)} val samples  "
          f"saved to {_col(out_dir, _CYAN)}")
    print(f"  {_col('Time:', _YELLOW)} {elapsed:.1f}s")
    print(_col("═" * 70, _BOLD + _GREEN))


if __name__ == "__main__":
    main()
