# 🤸 GCN Action Recognition

Nhận diện hành động con người theo thời gian thực bằng **ST-GCN** (Spatial Temporal Graph Convolutional Network).

Pipeline: **Webcam / Video file → YOLO Pose → keypoints → ST-GCN → Action label**

---

## 📁 Cấu trúc dự án

```
gcn-action-recognition/
├── api/                    # FastAPI backend
│   ├── main.py             # Entry point, load model lúc startup
│   ├── state.py            # Shared state (predictor)
│   ├── schemas.py          # Pydantic request/response models
│   └── routes/
│       ├── health.py       # GET /health
│       └── predict.py      # POST /predict, POST /reset
├── demo/
│   ├── app.py              # Streamlit frontend (webcam + video file)
│   └── webcam_inference.py # CLI demo thuần OpenCV (không cần Streamlit)
├── src/
│   ├── inference/
│   │   └── predictor.py    # SlidingWindowPredictor
│   ├── models/
│   │   └── st_gcn.py       # ST-GCN model
│   └── utils/
├── configs/
│   └── base.yaml           # Cấu hình model, classes, window_size
├── runs/exp/checkpoints/
│   └── best.pt             # Model đã train
└── yolo11m-pose.pt         # YOLO pose detection model
```

---

## ⚙️ Yêu cầu

- Python ≥ 3.9
- Webcam kết nối với máy tính *(chỉ cần nếu dùng chế độ webcam)*
- (Tùy chọn) GPU CUDA để chạy nhanh hơn

---

## 🚀 Cách chạy

### Bước 1 — Cài dependencies

```bash
pip install fastapi uvicorn[standard] streamlit ultralytics requests pydantic omegaconf torch torchvision
```

Hoặc cài toàn bộ từ `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

### Bước 2 — Khởi động FastAPI Backend

> **Quan trọng:** Phải chạy bước này **trước** khi mở Streamlit.

Mở **Terminal 1**, chạy từ thư mục gốc dự án:

```bash
cd e:\gcn-action-recognition

uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Khi thấy log như sau là backend đã sẵn sàng:

```
[API] Model loaded từ runs/exp/checkpoints/best.pt | device=cpu
[API] Predictor sẵn sàng | classes=['Fall Down', 'Lying Down', 'Walking'] | window=50
INFO:     Uvicorn running on http://0.0.0.0:8000
```

> **❓ Vào http://localhost:8000 chỉ thấy trang status?**  
> Đó là bình thường — FastAPI là REST API, không có giao diện tại `/`.  
> Trang root sẽ tự redirect sau 3 giây. Để xem đầy đủ API, truy cập:  
> 👉 **http://localhost:8000/docs** — Swagger UI (thử API trực tiếp trên trình duyệt)

---

### Bước 3 — Mở Streamlit Frontend

Mở **Terminal 2** (để Terminal 1 vẫn chạy), chạy từ thư mục gốc:

```bash
cd e:\gcn-action-recognition

streamlit run demo/app.py
```

Trình duyệt sẽ tự động mở tại: **http://localhost:8501**

Nếu không tự mở, hãy copy URL đó vào trình duyệt thủ công.

> ⚠️ Nếu trang hiện thông báo **"Không kết nối được API"** → kiểm tra Terminal 1 đang chạy bình thường.

---

### Bước 4 — Sử dụng

1. Trong sidebar trái, kiểm tra **✅ API đang chạy**
2. Chọn **nguồn đầu vào**:

**Chế độ Webcam** `📷 Webcam`
- Nhấn **▶ Bắt đầu** để mở webcam
- Đứng trước camera, thực hiện các hành động:
  - 🚶 **Walking** — đi bộ
  - 🤸 **Fall Down** — ngã xuống
  - 🛋️ **Lying Down** — nằm xuống
- Đợi khoảng **2–3 giây** (model cần đủ 50 frame trong buffer trước khi dự đoán)
- Nhấn **⏹ Dừng** để tắt webcam

**Chế độ Video file** `🎬 Video file`
- Upload file video (`.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`) qua nút **Browse files**
- Chọn **tốc độ phát** (1×, 2×, 4×, 8×)
- Nhấn **▶ Bắt đầu** — video sẽ phát và nhận dạng từng frame
- Thanh progress bar màu xanh ở đáy video cho biết tiến độ

> 💡 **Tip:** Nhấn **🔄 Reset buffer** khi muốn bắt đầu phiên nhận dạng mới (buffer sẽ tự reset mỗi lần nhấn ▶ Bắt đầu)

---

## 🖥️ Tùy chọn CLI (không cần Streamlit)

Nếu muốn test nhanh bằng cửa sổ OpenCV thay vì Streamlit:

```bash
# Đảm bảo FastAPI vẫn đang chạy ở Terminal 1
python demo/webcam_inference.py
```

Nhấn **Q** để thoát.

---

## 🌐 API Endpoints

| Method | Endpoint    | Mô tả |
|--------|-------------|-------|
| `GET`  | `/health`   | Kiểm tra trạng thái server & model |
| `POST` | `/predict`  | Gửi 1 frame keypoints, nhận action label |
| `POST` | `/reset`    | Reset sliding window buffer |
| `GET`  | `/docs`     | Swagger UI — thử API trực tiếp trên browser |

### Ví dụ gọi `/predict` bằng curl:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"frame": {"keypoints": [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]}}'
```

---

## 🐛 Xử lý sự cố

### Trang Streamlit không hiển thị / trắng trang
- Thử nhấn **F5** hoặc **Ctrl+Shift+R** để hard reload trang
- Đảm bảo FastAPI đang chạy ở Terminal 1

### Lỗi "Cannot connect to API"
```
❌ Không kết nối được API
```
→ Chạy lại Terminal 1: `uvicorn api.main:app --port 8000`

### Lỗi "Không mở được webcam"
- Kiểm tra webcam đang kết nối
- Đóng các ứng dụng khác đang dùng webcam (Zoom, Teams, OBS...)
- Thử đổi device index: sửa `cv2.VideoCapture(0)` thành `cv2.VideoCapture(1)` trong `demo/app.py`

### Model dự đoán sai / chưa hiển thị kết quả
- Chờ đủ **50 frame** (khoảng 2–3 giây) để buffer đầy — lúc này hiện "Warming up..."
- Nhấn **🔄 Reset buffer** rồi thử lại
- Đảm bảo người đứng đủ gần camera để YOLO detect được keypoints

### Chạy chậm / FPS thấp
- Mặc định giới hạn **20 FPS** để tránh quá tải
- Nếu có GPU, FastAPI sẽ tự dùng CUDA (log `device=cuda`)
- Tắt skeleton overlay trong sidebar để giảm tải Streamlit

---

## 🔧 Cấu hình

Chỉnh sửa `configs/base.yaml` để thay đổi:

```yaml
data:
  num_classes: 3
  max_frames: 50          # ← kích thước sliding window
  class_names:
    - Fall Down
    - Lying Down
    - Walking

model:
  type: stgcn
  in_channels: 3          # x, y, confidence
```

---

## 📊 Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit (port 8501)                    │
│                                                             │
│  Webcam → OpenCV → YOLO Pose → keypoints (17×3)            │
│                                     │                       │
│                             HTTP POST /predict              │
└─────────────────────────────────────┼───────────────────────┘
                                      │
┌─────────────────────────────────────▼───────────────────────┐
│                    FastAPI (port 8000)                      │
│                                                             │
│  keypoints → SlidingWindowPredictor → ST-GCN → label       │
│              (buffer 50 frames, EMA smoothing)              │
└─────────────────────────────────────────────────────────────┘
```

**Lý do tách FastAPI + Streamlit:**
- ST-GCN model (~24MB) chỉ load **1 lần** trong FastAPI — không bị reload mỗi khi Streamlit rerun
- Streamlit chỉ lo phần UI nhẹ, gọi HTTP request để lấy kết quả
- Có thể chạy nhiều client Streamlit cùng 1 backend
