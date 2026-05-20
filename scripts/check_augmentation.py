import sys
import numpy as np
from pathlib import Path

# Đảm bảo có thể import các module từ src/
sys.path.append(str(Path('.').absolute()))

from src.data.dataset import SkeletonDataset
from src.utils.visualize import visualize_batch

def main():
    # Cấu hình đường dẫn dữ liệu dựa theo base.yaml
    data_path = "data/processed/train_data.npy"
    label_path = "data/processed/train_label.pkl"
    class_names = ["Fall Down", "Lying Down", "Walking"]
    
    # Nơi lưu ảnh kết quả
    out_dir = Path("runs/visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Đang load dữ liệu...")
    # 1. Dataset GỐC (Tắt toàn bộ tiền xử lý)
    dataset_raw = SkeletonDataset(
        data_path, label_path,
        normalize=False,
        random_choose=False, random_shift=False, random_move=False,
        mmap=True # Dùng mmap để tiết kiệm RAM
    )

    # 2. Dataset ĐÃ XỬ LÝ (Bật Normalize và Augmentations)
    dataset_aug = SkeletonDataset(
        data_path, label_path,
        normalize=True,
        window_size=50,
        random_choose=True, random_shift=True, random_move=True,
        mmap=True
    )

    # Chọn 8 video (mẫu) đầu tiên để quan sát
    n_samples = 8
    indices = list(range(n_samples))
    
    # Lấy dữ liệu
    raw_data = [dataset_raw[i][0] for i in indices]
    raw_labels = [dataset_raw[i][1] for i in indices]
    
    aug_data = [dataset_aug[i][0] for i in indices]
    aug_labels = [dataset_aug[i][1] for i in indices]

    # Stack list thành mảng NumPy (N, C, T, V, M)
    raw_data_np = np.stack(raw_data)
    raw_labels_np = np.array(raw_labels)
    
    aug_data_np = np.stack(aug_data)
    aug_labels_np = np.array(aug_labels)

    # Trực quan hóa dữ liệu RAW
    raw_out = visualize_batch(
        raw_data_np, raw_labels_np, class_names, 
        out_path=out_dir, 
        n_samples=n_samples, 
        n_cols=4, 
        split="raw", # Sẽ tạo ra tên file là raw_samples_epoch000.png
        epoch=0
    )
    print(f"📸 Đã lưu ảnh GỐC (không xử lý) tại: {raw_out}")

    # Trực quan hóa dữ liệu đã AUGMENTED
    aug_out = visualize_batch(
        aug_data_np, aug_labels_np, class_names, 
        out_path=out_dir, 
        n_samples=n_samples, 
        n_cols=4, 
        split="augmented", # Sẽ tạo ra tên file là augmented_samples_epoch000.png
        epoch=0
    )
    print(f"📸 Đã lưu ảnh ĐÃ XỬ LÝ (Augmented + Normalized) tại: {aug_out}")

if __name__ == "__main__":
    main()
