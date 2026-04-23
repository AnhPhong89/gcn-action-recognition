import argparse
import os
import pickle
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import sys
# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.skeleton_extractor import SkeletonExtractor


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
    
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not raw_dir.exists():
        print(f"Error: Raw directory {raw_dir} does not exist.")
        return

    # Khởi tạo extractor
    print(f"Initializing YOLOv11n-pose (device={args.device}, conf={args.conf})...")
    extractor = SkeletonExtractor(conf_threshold=args.conf, device=args.device)
    
    # Lấy danh sách các class (thư mục con)
    classes = [d.name for d in raw_dir.iterdir() if d.is_dir()]
    classes.sort()  # Sort alphabetically to ensure consistent label indices
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    print(f"Found {len(classes)} classes: {classes}")
    
    samples = []   # Tên file/mẫu
    labels = []    # Label index
    skeletons = [] # Numpy arrays (C, T_max, V, M)
    
    # Duyệt qua các thư mục class
    for cls_name in classes:
        cls_dir = raw_dir / cls_name
        video_paths = list(cls_dir.glob("*.mp4")) + list(cls_dir.glob("*.avi"))
        
        if args.limit > 0:
            video_paths = video_paths[:args.limit]
            
        print(f"\nProcessing class '{cls_name}' ({len(video_paths)} videos)...")
        label_idx = class_to_idx[cls_name]
        
        for p in tqdm(video_paths, desc=cls_name):
            try:
                # Extract skeleton: (C, T, V, M)
                skel = extractor.extract_from_video(p)
                
                # Cắt hoặc bù frames
                skel_fixed = pad_or_truncate(skel, args.max_frames)
                
                samples.append(p.stem) # use filename as sample name
                labels.append(label_idx)
                skeletons.append(skel_fixed)
                
            except Exception as e:
                print(f"Error processing {p.name}: {e}")
                
    if len(skeletons) == 0:
        print("No valid videos processed. Exiting.")
        return
        
    print(f"\nExtracted features from {len(skeletons)} videos.")
    
    # Gộp thành array lớn: (N, C, T, V, M)
    data = np.stack(skeletons, axis=0)
    print(f"Full data shape: {data.shape}")
    
    # Split train / val
    print(f"Splitting data (train: {1-args.test_size}, val: {args.test_size})...")
    X_train, X_val, y_train, y_val, names_train, names_val = train_test_split(
        data, labels, samples,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels
    )
    
    # Lưu files
    print(f"Saving to {out_dir}...")
    
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
        
    print(f"Done! Saved {len(y_train)} train samples and {len(y_val)} val samples.")

if __name__ == "__main__":
    main()
