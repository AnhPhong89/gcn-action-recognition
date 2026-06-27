import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ultralytics import YOLO
from pathlib import Path
import asyncio

from api.state import app_state
from api.routes.predict import coco17_to_openpose18

router = APIRouter()

YOLO_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "yolo11m-pose.pt"
yolo_model = None

def get_yolo():
    global yolo_model
    if yolo_model is None:
        print(f"[WS] Loading YOLO from {YOLO_MODEL_PATH}")
        yolo_model = YOLO(str(YOLO_MODEL_PATH))
    return yolo_model

@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected")
    
    yolo = get_yolo()
    predictor = app_state.get("predictor")
    
    if predictor is None:
        await websocket.close(code=1011, reason="Predictor not loaded")
        return
        
    # Reset buffer khi có connection mới
    predictor.reset()
    
    try:
        while True:
            # Receive image bytes from client
            data = await websocket.receive_bytes()
            
            # Decode frame
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
                
            # Run YOLO
            results = yolo(frame, conf=0.3, verbose=False)
            kpts = np.zeros((17, 3), dtype=np.float32)
            if results and results[0].keypoints is not None:
                kp = results[0].keypoints
                if kp.xy is not None and len(kp.xy) > 0:
                    kpts[:, :2] = kp.xy[0].cpu().numpy()
                    kpts[:, 2]  = kp.conf[0].cpu().numpy()

            # Predict Action
            kpts_op = coco17_to_openpose18(kpts)
            r = predictor.push_frame(kpts_op)

            # Phản hồi lại client
            await websocket.send_json({
                "label": r["label"],
                "confidence": float(r["confidence"]),
                "ready": r["ready"],
                "keypoints": kpts.tolist() # Gửi lại keypoints để JS vẽ lên video
            })

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
        try:
            await websocket.close()
        except:
            pass
