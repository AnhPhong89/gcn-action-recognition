"""
CLI Webcam Inference — không cần Streamlit, chạy trực tiếp với OpenCV window.

Dùng khi muốn test nhanh mà không cần UI:
    python demo/webcam_inference.py

Cần FastAPI đang chạy: uvicorn api.main:app --port 8000
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import time
import cv2
import numpy as np
import requests
from ultralytics import YOLO

API_URL    = "http://localhost:8000"
YOLO_MODEL = str(ROOT / "yolo11m-pose.pt")

COCO_EDGES = [
    (0,1),(0,2),(1,3),(2,4),(0,5),(0,6),
    (5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,13),(13,15),(12,14),(14,16),
    (5,6),(11,12),
]

ACTION_COLORS_BGR = {
    "Fall Down":   (0,  80,  255),
    "Lying Down":  (0,  200, 255),
    "Walking":     (0,  220, 80),
}


def call_predict(keypoints):
    try:
        r = requests.post(f"{API_URL}/predict",
                          json={"frame": {"keypoints": keypoints.tolist()}},
                          timeout=0.5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def draw(frame, keypoints, label, confidence, ready, fps):
    h, w = frame.shape[:2]
    pts = [(int(k[0]), int(k[1])) for k in keypoints]
    color = ACTION_COLORS_BGR.get(label, (200,200,200))

    # skeleton
    for i, j in COCO_EDGES:
        if keypoints[i][2] > 0.1 and keypoints[j][2] > 0.1:
            cv2.line(frame, pts[i], pts[j], (0,230,100), 2, cv2.LINE_AA)
    for idx, (x, y, c) in enumerate(keypoints):
        if c > 0.1:
            cv2.circle(frame, (int(x), int(y)), 4, (255,255,255), -1)

    # label bar
    if ready:
        cv2.rectangle(frame, (0,0), (w, 50), (20,20,20), -1)
        cv2.putText(frame, f"{label}  {confidence*100:.1f}%",
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Warming up...", (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150,150,150), 1, cv2.LINE_AA)

    cv2.putText(frame, f"FPS: {fps:.1f}", (w-110, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,200,255), 1, cv2.LINE_AA)


def main():
    print(f"[demo] Loading YOLO từ {YOLO_MODEL}")
    yolo = YOLO(YOLO_MODEL)

    print(f"[demo] Kiểm tra API tại {API_URL}")
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        h = r.json()
        print(f"[demo] API OK | classes={h['class_names']} | device={h['device']}")
    except Exception as e:
        print(f"[demo] ❌ Không kết nối được API: {e}")
        print("        Hãy chạy: uvicorn api.main:app --port 8000")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    prev = time.time()
    label, confidence, ready = "–", 0.0, False

    print("[demo] Nhấn Q để thoát")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        results = yolo(frame, conf=0.3, verbose=False)
        kpts = np.zeros((17, 3), dtype=np.float32)
        if results and results[0].keypoints is not None:
            kp = results[0].keypoints
            if kp.xy is not None and len(kp.xy) > 0:
                kpts[:, :2] = kp.xy[0].cpu().numpy()
                kpts[:, 2]  = kp.conf[0].cpu().numpy()

        result = call_predict(kpts)
        if result:
            label      = result["label"]
            confidence = result["confidence"]
            ready      = result["ready"]

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now

        draw(frame, kpts, label, confidence, ready, fps)
        cv2.imshow("GCN Action Recognition — [Q] thoát", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
