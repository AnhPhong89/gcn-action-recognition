"""
Pydantic schemas cho FastAPI endpoints.
"""
from typing import List, Optional
from pydantic import BaseModel


class KeypointFrame(BaseModel):
    """Một frame keypoints từ YOLO pose detection.

    keypoints: list of 17 keypoints, mỗi keypoint là [x, y, conf]
               Shape: (17, 3)
    """
    keypoints: List[List[float]]  # (17, 3)


class PredictRequest(BaseModel):
    """Request body gửi lên /predict."""
    frame: KeypointFrame


class PredictResponse(BaseModel):
    """Response từ /predict."""
    label: str
    confidence: float
    probs: List[float]
    frame_idx: int
    ready: bool


class ResetResponse(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    class_names: List[str]
    window_size: int


class BatchPredictRequest(BaseModel):
    """Gửi nhiều frame keypoints cùng lúc (dùng cho video file).

    frames: danh sách các frame, mỗi frame là (17, 3)
    """
    frames: List[KeypointFrame]


class FrameResult(BaseModel):
    """Kết quả dự đoán cho 1 frame trong batch."""
    label: str
    confidence: float
    probs: List[float]
    frame_idx: int
    ready: bool


class BatchPredictResponse(BaseModel):
    """Response từ /predict_batch."""
    results: List[FrameResult]
    total_frames: int
