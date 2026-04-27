"""
Sliding-window real-time predictor for ST-GCN action recognition.

The predictor maintains a ring buffer of `window_size` skeleton frames.
Every time a new frame arrives it is appended to the buffer (discarding the
oldest frame once the buffer is full).  When the buffer contains at least
`window_size` frames the model runs inference on the whole window → one
action label per frame cadence.

                  frame 1 … 300  → predict
                  frame 2 … 301  → predict
                  frame 3 … 302  → predict
                  …

Design goals
────────────
• No coupling to cv2 or any display library — the predictor only handles
  skeleton → action logic; video capture / drawing lives in the script layer.
• Normalization identical to training (hip-centre + torso-scale).
• Smooth output via a configurable temporal smoothing window on the raw
  class-probability vectors.
"""

from __future__ import annotations

import collections
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Helper: normalize one (C, T, V, M) clip — same logic as SkeletonDataset
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_clip(data: np.ndarray) -> np.ndarray:
    """Center-normalize a (C, T, V, M) skeleton clip.

    Mirrors ``SkeletonDataset._normalize`` so inference matches training exactly.
    """
    C, T, V, M = data.shape
    result = data.copy()
    for m in range(M):
        hip_x = (result[0, :, 11, m] + result[0, :, 12, m]) / 2.0
        hip_y = (result[1, :, 11, m] + result[1, :, 12, m]) / 2.0
        s_x   = (result[0, :, 5,  m] + result[0, :, 6,  m]) / 2.0
        s_y   = (result[1, :, 5,  m] + result[1, :, 6,  m]) / 2.0
        torso = np.sqrt((s_x - hip_x) ** 2 + (s_y - hip_y) ** 2)
        valid = torso[torso > 1e-4]
        scale = valid.mean() if len(valid) > 0 else 1.0
        for t in range(T):
            if result[0, t, :, m].sum() == 0 and result[1, t, :, m].sum() == 0:
                continue
            result[0, t, :, m] = (result[0, t, :, m] - hip_x[t]) / scale
            result[1, t, :, m] = (result[1, t, :, m] - hip_y[t]) / scale
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SlidingWindowPredictor
# ─────────────────────────────────────────────────────────────────────────────

class SlidingWindowPredictor:
    """Real-time sliding-window action predictor.

    Args:
        model:        Loaded ST-GCN model (already on the right device).
        class_names:  List of class label strings ordered by class index.
        window_size:  Number of frames per inference window (must match
                      ``max_frames`` used during pre-processing / training).
        stride:       Run inference every ``stride`` frames.
                      stride=1  → inference on every frame (slowest, smoothest).
                      stride=N  → same prediction repeated for N frames.
        smooth_alpha: EMA smoothing factor for probability vectors.
                      0 = no smoothing, higher = more temporal smoothing.
        device:       'cuda' or 'cpu'.
        normalize:    Apply hip-center normalization (should match training).
        conf_threshold: Minimum mean keypoint confidence to consider a frame
                        as "person detected".  Frames below this threshold are
                        treated as zero-padded.
    """

    NUM_KEYPOINTS: int = 17
    NUM_CHANNELS:  int = 3   # x, y, conf
    NUM_PERSONS:   int = 1

    def __init__(
        self,
        model:           nn.Module,
        class_names:     List[str],
        window_size:     int = 300,
        stride:          int = 1,
        smooth_alpha:    float = 0.5,
        device:          str = "cpu",
        normalize:       bool = True,
        conf_threshold:  float = 0.1,
    ):
        self.model          = model.to(device).eval()
        self.class_names    = class_names
        self.window_size    = window_size
        self.stride         = max(1, stride)
        self.smooth_alpha   = smooth_alpha
        self.device         = device
        self.normalize      = normalize
        self.conf_threshold = conf_threshold

        self.num_classes = len(class_names)

        # Ring buffer: each element is (V=17, C=3) keypoint array for one frame
        self._buffer: collections.deque = collections.deque(
            maxlen=window_size
        )

        # Running EMA probabilities (num_classes,)
        self._smooth_probs: Optional[np.ndarray] = None

        # Internal counters
        self._frame_count: int = 0   # total frames pushed
        self._infer_count: int = 0   # total inference runs

        # Last prediction (label_str, confidence, probs)
        self.last_label:      str            = "–"
        self.last_confidence: float          = 0.0
        self.last_probs:      Optional[np.ndarray] = None

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def push_frame(self, keypoints: np.ndarray) -> dict:
        """Feed one frame of keypoints and return the current prediction.

        Args:
            keypoints: (V=17, 3) array — (x, y, conf) per keypoint in
                       **pixel coordinates** (same as YOLO output).
                       Pass ``np.zeros((17, 3))`` if no person was detected.

        Returns:
            dict with keys:
                label      – predicted action string  ('' if buffer not full)
                confidence – probability of predicted class  [0, 1]
                probs      – (num_classes,) softmax probabilities
                frame_idx  – total frames seen so far
                ready      – True once the buffer has been filled once
        """
        self._buffer.append(keypoints.astype(np.float32))
        self._frame_count += 1

        ready = len(self._buffer) == self.window_size

        # Run inference every `stride` frames once the buffer is full
        if ready and (self._frame_count % self.stride == 0):
            self._run_inference()

        return {
            "label":      self.last_label,
            "confidence": self.last_confidence,
            "probs":      self.last_probs,
            "frame_idx":  self._frame_count,
            "ready":      ready,
        }

    def reset(self):
        """Clear the buffer and all cached predictions."""
        self._buffer.clear()
        self._smooth_probs = None
        self._frame_count  = 0
        self._infer_count  = 0
        self.last_label      = "–"
        self.last_confidence = 0.0
        self.last_probs      = None

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _build_clip(self) -> np.ndarray:
        """Convert the current buffer to a (C, T, V, M) float32 array."""
        # Stack buffer → (T, V, C) then transpose → (C, T, V, 1)
        frames = np.stack(list(self._buffer), axis=0)        # (T, V, 3)
        clip   = frames.transpose(2, 0, 1)[:, :, :, np.newaxis]  # (C, T, V, 1)
        return clip.astype(np.float32)

    @torch.no_grad()
    def _run_inference(self):
        """Build clip, optionally normalize, run model, update smoothed probs."""
        clip = self._build_clip()   # (C, T, V, M)

        if self.normalize:
            clip = _normalize_clip(clip)

        # (1, C, T, V, M) → model
        tensor = torch.from_numpy(clip).unsqueeze(0).to(self.device)

        logits = self.model(tensor)           # (1, num_classes)
        probs  = torch.softmax(logits, dim=1) \
                      .squeeze(0)             \
                      .cpu()                  \
                      .numpy()                # (num_classes,)

        # Temporal EMA smoothing
        if self._smooth_probs is None or self.smooth_alpha == 0:
            self._smooth_probs = probs
        else:
            a = self.smooth_alpha
            self._smooth_probs = a * self._smooth_probs + (1 - a) * probs

        pred_idx = int(self._smooth_probs.argmax())
        self.last_label      = self.class_names[pred_idx]
        self.last_confidence = float(self._smooth_probs[pred_idx])
        self.last_probs      = self._smooth_probs.copy()
        self._infer_count   += 1
