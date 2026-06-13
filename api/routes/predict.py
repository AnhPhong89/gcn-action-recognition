"""
Prediction routes:
  POST /predict         — 1 frame tại một thời điểm (webcam)
  POST /predict_batch   — nhiều frame cùng lúc (video file)
  POST /reset           — reset sliding window buffer

Lưu ý: YOLO output COCO-17 keypoints, nhưng model được train với OpenPose-18.
Hàm coco17_to_openpose18() thực hiện convert tự động phía server.
"""
import numpy as np
from fastapi import APIRouter, HTTPException

from api.schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse, FrameResult,
    ResetResponse,
)
from api.state import app_state

router = APIRouter()

# ── COCO-17 → OpenPose-18 mapping ───────────────────────────────────────────
# COCO-17:  0=nose, 1=l_eye, 2=r_eye, 3=l_ear, 4=r_ear,
#           5=l_shl, 6=r_shl, 7=l_elb, 8=r_elb, 9=l_wri, 10=r_wri,
#           11=l_hip, 12=r_hip, 13=l_kne, 14=r_kne, 15=l_ank, 16=r_ank
#
# OpenPose-18: 0=nose, 1=neck, 2=r_shl, 3=r_elb, 4=r_wri,
#              5=l_shl, 6=l_elb, 7=l_wri, 8=r_hip, 9=r_kne, 10=r_ank,
#              11=l_hip, 12=l_kne, 13=l_ank, 14=r_eye, 15=l_eye,
#              16=r_ear, 17=l_ear

_COCO_TO_OP = [
    0,   # OP-0  nose       ← COCO-0  nose
    -1,  # OP-1  neck       ← mean(COCO-5, COCO-6) shoulders
    6,   # OP-2  r_shoulder ← COCO-6
    8,   # OP-3  r_elbow    ← COCO-8
    10,  # OP-4  r_wrist    ← COCO-10
    5,   # OP-5  l_shoulder ← COCO-5
    7,   # OP-6  l_elbow    ← COCO-7
    9,   # OP-7  l_wrist    ← COCO-9
    12,  # OP-8  r_hip      ← COCO-12
    14,  # OP-9  r_knee     ← COCO-14
    16,  # OP-10 r_ankle    ← COCO-16
    11,  # OP-11 l_hip      ← COCO-11
    13,  # OP-12 l_knee     ← COCO-13
    15,  # OP-13 l_ankle    ← COCO-15
    2,   # OP-14 r_eye      ← COCO-2
    1,   # OP-15 l_eye      ← COCO-1
    4,   # OP-16 r_ear      ← COCO-4
    3,   # OP-17 l_ear      ← COCO-3
]


def coco17_to_openpose18(kpts_coco: np.ndarray) -> np.ndarray:
    """Chuyển đổi COCO-17 keypoints sang OpenPose-18 format.

    Args:
        kpts_coco: (17, 3) float32 — x, y, conf (YOLO COCO output)

    Returns:
        (18, 3) float32 — x, y, conf (OpenPose)
    """
    op = np.zeros((18, 3), dtype=np.float32)
    for op_idx, coco_idx in enumerate(_COCO_TO_OP):
        if coco_idx == -1:
            # Neck = midpoint of left & right shoulder
            l_shl = kpts_coco[5]
            r_shl = kpts_coco[6]
            if l_shl[2] > 0.1 or r_shl[2] > 0.1:
                op[op_idx, 0] = (l_shl[0] + r_shl[0]) / 2.0
                op[op_idx, 1] = (l_shl[1] + r_shl[1]) / 2.0
                op[op_idx, 2] = (l_shl[2] + r_shl[2]) / 2.0
        else:
            op[op_idx] = kpts_coco[coco_idx]
    return op


def _run_predict(predictor, kpts_coco: np.ndarray) -> dict:
    """Convert COCO-17 → OpenPose-18 rồi feed vào predictor."""
    kpts_op = coco17_to_openpose18(kpts_coco)
    return predictor.push_frame(kpts_op)


def _make_frame_result(r: dict) -> FrameResult:
    return FrameResult(
        label=r["label"],
        confidence=float(r["confidence"]),
        probs=[float(p) for p in r["probs"]] if r["probs"] is not None else [],
        frame_idx=r["frame_idx"],
        ready=r["ready"],
    )


# ── Routes ───────────────────────────────────────────────────────────────────
@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    """Feed 1 frame keypoints (COCO-17) vào predictor và trả kết quả."""
    predictor = app_state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model chưa được load")

    kpts = np.array(body.frame.keypoints, dtype=np.float32)
    if kpts.shape != (17, 3):
        raise HTTPException(
            status_code=422,
            detail=f"keypoints phải có shape (17, 3), nhận được {kpts.shape}",
        )

    r = _run_predict(predictor, kpts)
    fr = _make_frame_result(r)
    return PredictResponse(
        label=fr.label, confidence=fr.confidence,
        probs=fr.probs, frame_idx=fr.frame_idx, ready=fr.ready,
    )


@router.post("/predict_batch", response_model=BatchPredictResponse)
def predict_batch(body: BatchPredictRequest):
    """Xử lý nhiều frame cùng lúc — dành cho video file.

    Gửi tất cả keypoints của video lên 1 lần thay vì gọi /predict từng frame.
    Nhanh hơn nhiều vì tránh HTTP overhead mỗi frame.
    """
    predictor = app_state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model chưa được load")

    if not body.frames:
        raise HTTPException(status_code=422, detail="frames không được rỗng")

    predictor.reset()

    results: list[FrameResult] = []
    for frame in body.frames:
        kpts = np.array(frame.keypoints, dtype=np.float32)
        if kpts.shape != (17, 3):
            kpts = np.zeros((17, 3), dtype=np.float32)
        r = _run_predict(predictor, kpts)
        results.append(_make_frame_result(r))

    return BatchPredictResponse(results=results, total_frames=len(results))


@router.post("/reset", response_model=ResetResponse)
def reset():
    """Reset sliding window buffer."""
    predictor = app_state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model chưa được load")
    predictor.reset()
    return ResetResponse(message="Buffer đã được reset")
