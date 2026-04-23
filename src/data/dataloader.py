"""
DataLoader factory for skeleton-based action recognition.

Provides `build_dataloader()` to create train/val DataLoaders from
preprocessed .npy + .pkl files, and `build_dataloaders()` as a
convenience wrapper that returns both at once.
"""

from pathlib import Path
from torch.utils.data import DataLoader
from .dataset import SkeletonDataset


def build_dataloader(data_path: str,
                     label_path: str,
                     batch_size: int = 16,
                     shuffle: bool = True,
                     num_workers: int = 2,
                     pin_memory: bool = True,
                     drop_last: bool = False,
                     random_choose: bool = False,
                     random_shift: bool = False,
                     random_move: bool = False,
                     window_size: int = -1,
                     normalize: bool = True,
                     debug: bool = False,
                     mmap: bool = True) -> DataLoader:
    """Create a single DataLoader from .npy data and .pkl labels.

    Args:
        data_path:  Path to the .npy file (N, C, T, V, M).
        label_path: Path to the .pkl label file (sample_names, labels).
        batch_size: Samples per batch.
        shuffle:    Shuffle every epoch.
        num_workers: Parallel data-loading workers.
        pin_memory: Pin memory for faster GPU transfer.
        drop_last:  Drop the last incomplete batch.
        random_choose: Randomly crop temporal window (augmentation).
        random_shift:  Randomly shift the sequence in time (augmentation).
        random_move:   Random spatial jitter (augmentation).
        window_size: Target temporal length when random_choose is True.
        normalize:  Center-normalize around hip (recommended).
        debug:      Use only first 100 samples.
        mmap:       Memory-map the .npy file for large datasets.

    Returns:
        A torch DataLoader yielding (data, label) batches.
    """
    dataset = SkeletonDataset(
        data_path=str(data_path),
        label_path=str(label_path),
        random_choose=random_choose,
        random_shift=random_shift,
        random_move=random_move,
        window_size=window_size,
        normalize=normalize,
        debug=debug,
        mmap=mmap,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return loader


def build_dataloaders(processed_dir: str,
                      batch_size: int = 16,
                      num_workers: int = 2,
                      pin_memory: bool = True,
                      normalize: bool = True,
                      window_size: int = -1,
                      debug: bool = False) -> tuple:
    """Convenience function to create both train and val DataLoaders.

    Expects the following files inside `processed_dir`:
        - train_data.npy, train_label.pkl
        - val_data.npy,   val_label.pkl

    Args:
        processed_dir: Path to the directory with preprocessed files.
        batch_size: Samples per batch.
        num_workers: Parallel data-loading workers.
        pin_memory: Pin memory for faster GPU transfer.
        normalize:  Center-normalize around hip.
        window_size: Target temporal length (-1 = use full sequence).
        debug:      Use only first 100 samples.

    Returns:
        (train_loader, val_loader) tuple of DataLoaders.
    """
    processed_dir = Path(processed_dir)

    train_loader = build_dataloader(
        data_path=processed_dir / 'train_data.npy',
        label_path=processed_dir / 'train_label.pkl',
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,          # consistent batch sizes during training
        random_choose=window_size > 0,
        random_shift=True,       # temporal augmentation for training
        random_move=True,        # spatial augmentation for training
        window_size=window_size,
        normalize=normalize,
        debug=debug,
    )

    val_loader = build_dataloader(
        data_path=processed_dir / 'val_data.npy',
        label_path=processed_dir / 'val_label.pkl',
        batch_size=batch_size,
        shuffle=False,           # no shuffle for validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        random_choose=False,     # no augmentation for validation
        random_shift=False,
        random_move=False,
        window_size=window_size,
        normalize=normalize,
        debug=debug,
    )

    return train_loader, val_loader
