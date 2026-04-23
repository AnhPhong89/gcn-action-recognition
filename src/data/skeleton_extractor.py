"""
Skeleton extraction from video using YOLOv11n-pose.

Extracts COCO-17 keypoints per frame and returns a numpy array
shaped (C, T, V, M) suitable for ST-GCN training.

C = 3  (x, y, confidence)
T = number of frames
V = 17 (COCO keypoints)
M = 1  (single person — highest confidence)
"""

import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO


# COCO 17 keypoint names (for reference / visualization)
COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]

NUM_KEYPOINTS = 17


class SkeletonExtractor:
    """Extract skeleton keypoints from video using YOLOv11n-pose.

    Args:
        model_path: Path to YOLO pose model weights.
                    Defaults to 'yolo11n-pose.pt' (auto-download).
        conf_threshold: Minimum person detection confidence.
        device: Device to run inference on ('cuda', 'cpu', or 'auto').
    """

    def __init__(self,
                 model_path: str = 'yolo11n-pose.pt',
                 conf_threshold: float = 0.3,
                 device: str = 'auto'):
        if device == 'auto':
            import torch
            device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.conf_threshold = conf_threshold
        self.model = YOLO(model_path)

    def extract_from_video(self, video_path: str) -> np.ndarray:
        """Extract skeleton sequence from a single video file.

        Args:
            video_path: Path to the video file.

        Returns:
            np.ndarray of shape (C=3, T, V=17, M=1).
            If no person is detected in a frame, that frame is filled with zeros.
        """
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        frames_keypoints = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO pose estimation (single image, no verbose)
            results = self.model.predict(
                frame,
                device=self.device,
                conf=self.conf_threshold,
                verbose=False,
            )

            keypoints = self._pick_best_person(results[0])
            frames_keypoints.append(keypoints)

        cap.release()

        if len(frames_keypoints) == 0:
            raise ValueError(f"No frames read from video: {video_path}")

        # Stack: (T, V, C) → transpose to (C, T, V)
        skeleton = np.stack(frames_keypoints, axis=0)  # (T, V, 3)
        skeleton = skeleton.transpose(2, 0, 1)          # (3, T, V)
        skeleton = skeleton[:, :, :, np.newaxis]         # (3, T, V, 1) = (C, T, V, M)

        # Interpolate missing keypoints
        skeleton = self._interpolate_missing(skeleton)

        return skeleton.astype(np.float32)

    def _pick_best_person(self, result) -> np.ndarray:
        """From a YOLO result, pick the person with highest confidence.

        Returns:
            np.ndarray of shape (V=17, 3) — (x, y, conf) per keypoint.
            Returns zeros if no person detected.
        """
        if result.keypoints is None or len(result.keypoints) == 0:
            return np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)

        kpts = result.keypoints  # Keypoints object

        # kpts.data shape: (num_persons, 17, 3) — x, y, conf
        data = kpts.data.cpu().numpy()

        if data.shape[0] == 0:
            return np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)

        # Pick person with highest average keypoint confidence
        mean_confs = data[:, :, 2].mean(axis=1)
        best_idx = mean_confs.argmax()
        best_kpts = data[best_idx]  # (17, 3)

        return best_kpts.astype(np.float32)

    @staticmethod
    def _interpolate_missing(skeleton: np.ndarray) -> np.ndarray:
        """Linearly interpolate frames where keypoints are all zeros.

        Args:
            skeleton: (C, T, V, M) array.

        Returns:
            Interpolated skeleton array with same shape.
        """
        C, T, V, M = skeleton.shape
        result = skeleton.copy()

        for v in range(V):
            for m in range(M):
                # Check which frames have valid data (non-zero x or y)
                valid = (result[0, :, v, m] != 0) | (result[1, :, v, m] != 0)

                if valid.sum() == 0 or valid.sum() == T:
                    continue  # All missing or all valid — skip

                valid_indices = np.where(valid)[0]
                missing_indices = np.where(~valid)[0]

                for c in range(C):
                    result[c, missing_indices, v, m] = np.interp(
                        missing_indices,
                        valid_indices,
                        result[c, valid_indices, v, m],
                    )

        return result
