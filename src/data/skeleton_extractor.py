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
        layout: 'coco' (17 kpts) or 'openpose' (18 kpts).
    """

    def __init__(self,
                 model_path: str = 'yolo11n-pose.pt',
                 conf_threshold: float = 0.3,
                 device: str = 'auto',
                 layout: str = 'openpose'):
        if device == 'auto':
            import torch
            device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.conf_threshold = conf_threshold
        self.layout = layout
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

        if self.layout == 'openpose':
            skeleton = self._convert_to_openpose(skeleton)

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

    @staticmethod
    def _convert_to_openpose(skeleton_coco: np.ndarray) -> np.ndarray:
        """Convert COCO 17 keypoints to OpenPose 18 keypoints layout.
        
        Args:
            skeleton_coco: np.ndarray shape (C, T, 17, M)
            
        Returns:
            np.ndarray shape (C, T, 18, M)
        """
        C, T, V, M = skeleton_coco.shape
        if V != 17:
            raise ValueError(f"Expected 17 keypoints for COCO, got {V}")
            
        skeleton_op = np.zeros((C, T, 18, M), dtype=np.float32)
        
        # Calculate Neck as midpoint of LShoulder (5) and RShoulder (6)
        neck = (skeleton_coco[:, :, 5, :] + skeleton_coco[:, :, 6, :]) / 2.0
        
        # Mapping from COCO (+ Neck at index 17) to OpenPose
        # COCO: 0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar, 5:LShoulder, 6:RShoulder, 7:LElbow, 8:RElbow,
        #       9:LWrist, 10:RWrist, 11:LHip, 12:RHip, 13:LKnee, 14:RKnee, 15:LAnkle, 16:RAnkle
        # OpenPose: 0:Nose, 1:Neck, 2:RShoulder, 3:RElbow, 4:RWrist, 5:LShoulder, 6:LElbow, 7:LWrist, 
        #       8:RHip, 9:RKnee, 10:RAnkle, 11:LHip, 12:LKnee, 13:LAnkle, 14:REye, 15:LEye, 16:REar, 17:LEar
        
        map_indices = [
            0,  # 0: Nose -> Nose (0)
            17, # 1: Neck -> Neck (17)
            6,  # 2: RShoulder -> RShoulder (6)
            8,  # 3: RElbow -> RElbow (8)
            10, # 4: RWrist -> RWrist (10)
            5,  # 5: LShoulder -> LShoulder (5)
            7,  # 6: LElbow -> LElbow (7)
            9,  # 7: LWrist -> LWrist (9)
            12, # 8: RHip -> RHip (12)
            14, # 9: RKnee -> RKnee (14)
            16, # 10: RAnkle -> RAnkle (16)
            11, # 11: LHip -> LHip (11)
            13, # 12: LKnee -> LKnee (13)
            15, # 13: LAnkle -> LAnkle (15)
            2,  # 14: REye -> REye (2)
            1,  # 15: LEye -> LEye (1)
            4,  # 16: REar -> REar (4)
            3   # 17: LEar -> LEar (3)
        ]
        
        # We append neck to the end of COCO features to easily index it
        skeleton_coco_ext = np.concatenate([skeleton_coco, neck[:, :, np.newaxis, :]], axis=2) # (C, T, 18, M)
        
        for op_idx, coco_idx in enumerate(map_indices):
            skeleton_op[:, :, op_idx, :] = skeleton_coco_ext[:, :, coco_idx, :]
            
        # Optional: Adjust confidence for the calculated Neck point
        # Since it's an average, the confidence is averaged too. We can leave it as is.
        
        return skeleton_op

    @staticmethod
    def _convert_frame_to_openpose(keypoints_coco: np.ndarray) -> np.ndarray:
        """Convert a single frame COCO keypoints (17, 3) to OpenPose (18, 3)."""
        if keypoints_coco.shape != (17, 3):
            raise ValueError(f"Expected (17, 3) shape, got {keypoints_coco.shape}")
            
        keypoints_op = np.zeros((18, 3), dtype=np.float32)
        neck = (keypoints_coco[5, :] + keypoints_coco[6, :]) / 2.0
        
        map_indices = [
            0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3
        ]
        
        kpts_ext = np.vstack([keypoints_coco, neck]) # (18, 3)
        for op_idx, coco_idx in enumerate(map_indices):
            keypoints_op[op_idx] = kpts_ext[coco_idx]
            
        return keypoints_op
