import argparse
import os
import pickle
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
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
    parser.add_argument('--raw_dir', type=str, default='data/raw', help='Path to raw videos directory')
    parser.add_argument('--out_dir', type=str, default='data/processed', help='Path to output processed data')
    parser.add_argument('--max_frames', type=int, default=300, help='Maximum frames to keep per video via padding/truncating')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold for YOLO pose')
    parser.add_argument('--test_size', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for splitting')
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
        # Truncate: take the middle part, or just the first max_frames.
        # Here we take the first max_frames.
        return skeleton[:, :max_frames, :, :]
        
    # Pad with zeros at the end
    padded = np.zeros((C, max_frames, V, M), dtype=np.float32)
    padded[:, :T, :, :] = skeleton
    return padded


def main():
    args = parse_args()
    t0 = time.time()
    
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not raw_dir.exists():
        print(_col(f"[ERROR] Raw directory {raw_dir} does not exist.", _RED + _BOLD))
        return

    # ── Startup banner ──────────────────────────────────────────────────────
    print(_col("═" * 70, _BOLD + _BLUE))
    print(_col("  ST-GCN  Skeleton Preprocessing", _BOLD + _CYAN))
    print(_col("═" * 70, _BOLD + _BLUE))
    cfg_items = [
        ("Raw dir",     raw_dir),
        ("Output dir",  out_dir),
        ("Max frames",  args.max_frames),
        ("Test split",  args.test_size),
        ("Confidence",  args.conf),
        ("Device",      args.device),
        ("Seed",        args.seed),
    ]
    for k, v in cfg_items:
        print(f"  {_col(k + ':', _YELLOW):<22} {v}")
    print(_col("═" * 70, _BOLD + _BLUE))

    # Khởi tạo extractor
    print(_col(f"  Loading YOLOv11n-pose model …", _CYAN))
    extractor = SkeletonExtractor(conf_threshold=args.conf, device=args.device)
    print(_col(f"  YOLOv11n-pose ready.", _GREEN + _BOLD))
    
    # Lấy danh sách các class (thư mục con)
    classes = [d.name for d in raw_dir.iterdir() if d.is_dir()]
    classes.sort()  # Sort alphabetically to ensure consistent label indices
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    print(f"\n  {_col('Classes found:', _YELLOW)} {_col(len(classes), _BOLD)} → {classes}")
    print()
    
    samples = []   # Tên file/mẫu
    labels = []    # Label index
    skeletons = [] # Numpy arrays (C, T_max, V, M)
    
    # Duyệt qua các thư mục class
    total_videos = 0
    for cls_name in classes:
        cls_dir = raw_dir / cls_name
        video_paths = list(cls_dir.glob("*.mp4")) + list(cls_dir.glob("*.avi"))
        
        if args.limit > 0:
            video_paths = video_paths[:args.limit]
        
        total_videos += len(video_paths)
        label_idx = class_to_idx[cls_name]
        ok_count = 0
        err_count = 0

        pbar = tqdm(
            video_paths,
            desc=f"  {_col(cls_name, _BOLD + _CYAN):<30}",
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
        status = _col("✓", _GREEN + _BOLD) if err_count == 0 else _col("!", _YELLOW + _BOLD)
        tqdm.write(
            f"  {status} {_col(cls_name, _BOLD):<20}  "
            f"{_col(ok_count, _GREEN)} ok  /  "
            f"{_col(err_count, _RED) if err_count else err_count} errors"
        )
                
    if len(skeletons) == 0:
        print(_col("No valid videos processed. Exiting.", _RED + _BOLD))
        return
        
    print(f"\n  {_col('Extracted:', _YELLOW)} {_col(len(skeletons), _BOLD + _GREEN)} / {total_videos} videos")
    
    # Gộp thành array lớn: (N, C, T, V, M)
    print(_col("  Stacking arrays …", _CYAN))
    data = np.stack(skeletons, axis=0)
    print(f"  {_col('Data shape:', _YELLOW)} {data.shape}")
    
    # Split train / val
    print(_col(f"  Splitting (train={1-args.test_size:.0%} / val={args.test_size:.0%}) …", _CYAN))
    X_train, X_val, y_train, y_val, names_train, names_val = train_test_split(
        data, labels, samples,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels
    )
    
    # Lưu files
    print(_col(f"  Saving to {out_dir} …", _CYAN))
    
    # Train
    np.save(out_dir / "train_data.npy", X_train)
    with open(out_dir / "train_label.pkl", 'wb') as f:
        pickle.dump((names_train, list(y_train)), f)
        
    # Val
    np.save(out_dir / "val_data.npy", X_val)
    with open(out_dir / "val_label.pkl", 'wb') as f:
        pickle.dump((names_val, list(y_val)), f)
        
    # Lưu label mapping
    with open(out_dir / "label_map.yaml", 'w') as f:
        yaml.dump(class_to_idx, f)

    elapsed = time.time() - t0
    print(_col("═" * 70, _BOLD + _GREEN))
    print(f"  {_col('Done!', _BOLD + _GREEN)}  "
          f"{_col(len(y_train), _BOLD)} train + {_col(len(y_val), _BOLD)} val samples  "
          f"saved to {_col(out_dir, _CYAN)}")
    print(f"  {_col('Time:', _YELLOW)} {elapsed:.1f}s")
    print(_col("═" * 70, _BOLD + _GREEN))


if __name__ == "__main__":
    main()
