"""
Skeleton dataset for ST-GCN action recognition.

Loads preprocessed .npy data (N, C, T, V, M) and .pkl labels,
applies augmentations, and returns (data_tensor, label) pairs.
"""

import numpy as np
import pickle
import torch
from . import transforms as tools


class SkeletonDataset(torch.utils.data.Dataset):
    """Feeder for skeleton-based action recognition.

    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label pickle file containing (sample_names, labels)
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        random_move: If true, apply spatial augmentations via random move
        window_size: The length of the output sequence
        normalize: If true, center-normalize coordinates around hip center
        debug: If true, only use the first 100 samples
        mmap: If true, use memory-mapped file for data loading
    """

    def __init__(self,
                 data_path,
                 label_path,
                 random_choose=False,
                 random_shift=False,
                 random_move=False,
                 window_size=-1,
                 normalize=True,
                 debug=False,
                 mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalize = normalize

        self.load_data(mmap)

    def load_data(self, mmap):
        """Load data and labels from disk."""
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data: (N, C, T, V, M)
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data — copy so augmentations don't modify the mmap'd source
        data_numpy = np.array(self.data[index])  # (C, T, V, M)
        label = self.label[index]

        # ── Normalization ──────────────────────────────────────
        if self.normalize:
            data_numpy = self._normalize(data_numpy)

        # ── Temporal augmentations ─────────────────────────────
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)

        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)

        # ── Spatial augmentations ──────────────────────────────
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy.astype(np.float32), label

    @staticmethod
    def _normalize(data_numpy):
        """Center-normalize skeleton coordinates.

        For COCO layout, uses the midpoint of left_hip (11) and
        right_hip (12) as the body center. Each frame's x,y coordinates
        are shifted so the center joint is at (0, 0), then rescaled
        by the torso length (shoulder→hip distance).

        Args:
            data_numpy: (C, T, V, M) array. C[0]=x, C[1]=y, C[2]=conf.

        Returns:
            Normalized array with same shape.
        """
        C, T, V, M = data_numpy.shape
        result = data_numpy.copy()

        # Only normalize x (ch0) and y (ch1), leave confidence (ch2) intact
        for m in range(M):
            # Hip center: midpoint of left_hip(11) and right_hip(12)
            hip_center_x = (result[0, :, 11, m] + result[0, :, 12, m]) / 2.0
            hip_center_y = (result[1, :, 11, m] + result[1, :, 12, m]) / 2.0

            # Shoulder center: midpoint of left_shoulder(5) and right_shoulder(6)
            shoulder_center_x = (result[0, :, 5, m] + result[0, :, 6, m]) / 2.0
            shoulder_center_y = (result[1, :, 5, m] + result[1, :, 6, m]) / 2.0

            # Torso length per frame (shoulder_center → hip_center)
            torso_len = np.sqrt(
                (shoulder_center_x - hip_center_x) ** 2 +
                (shoulder_center_y - hip_center_y) ** 2
            )
            # Avoid division by zero; use mean of valid torso lengths
            valid_torso = torso_len[torso_len > 1e-4]
            scale = valid_torso.mean() if len(valid_torso) > 0 else 1.0

            # Subtract hip center, then divide by scale
            for t in range(T):
                # Skip padded (all-zero) frames
                if result[0, t, :, m].sum() == 0 and result[1, t, :, m].sum() == 0:
                    continue
                result[0, t, :, m] = (result[0, t, :, m] - hip_center_x[t]) / scale
                result[1, t, :, m] = (result[1, t, :, m] - hip_center_y[t]) / scale

        return result
