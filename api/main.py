"""
FastAPI entry point — GCN Action Recognition API.

Khởi động:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Sau đó mở Streamlit:
    streamlit run demo/app.py
"""
import sys
from pathlib import Path

# Đảm bảo project root trong sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import OmegaConf

from src.models import STGCNModel
from src.inference import SlidingWindowPredictor
from src.utils.checkpoint import load_checkpoint
from api.state import app_state
from api.routes import health, predict as predict_router
from api.routes import websocket_stream

# ── Config mặc định (có thể override qua env var) ──────────────────────────
CONFIG_PATH   = ROOT / "configs" / "base.yaml"
CHECKPOINT    = ROOT / "runs" / "exp" / "checkpoints" / "best.pt"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE   = 50   # khớp với max_frames trong config


def _load_model_and_predictor():
    """Load ST-GCN model và khởi tạo SlidingWindowPredictor."""
    cfg = OmegaConf.load(CONFIG_PATH)

    class_names = list(cfg.data.class_names)
    num_classes  = cfg.data.num_classes
    graph_args   = OmegaConf.to_container(cfg.model.graph_args, resolve=True)

    model = STGCNModel(
        in_channels=cfg.model.in_channels,
        num_class=num_classes,
        graph_args=graph_args,
        edge_importance_weighting=cfg.model.edge_importance_weighting,
        dropout=cfg.model.dropout,
    )

    if CHECKPOINT.exists():
        load_checkpoint(str(CHECKPOINT), model, device=DEVICE)
        print(f"[API] Model loaded từ {CHECKPOINT} | device={DEVICE}")
    else:
        print(f"[API] ⚠️  Không tìm thấy checkpoint tại {CHECKPOINT} — chạy với weight ngẫu nhiên")

    predictor = SlidingWindowPredictor(
        model=model,
        class_names=class_names,
        window_size=WINDOW_SIZE,
        stride=1,
        smooth_alpha=0.5,
        device=DEVICE,
        normalize=True,
    )

    app_state["predictor"]  = predictor
    app_state["device"]     = DEVICE
    app_state["config"]     = {
        "class_names": class_names,
        "window_size": WINDOW_SIZE,
        "num_classes": num_classes,
    }
    print(f"[API] Predictor sẵn sàng | classes={class_names} | window={WINDOW_SIZE}")


# ── Lifespan (thay thế @app.on_event deprecated) ───────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model_and_predictor()
    yield
    app_state.clear()
    print("[API] Shutdown — state cleared")


# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="GCN Action Recognition API",
    description="Real-time skeleton-based action recognition với ST-GCN",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(predict_router.router, tags=["predict"])
app.include_router(websocket_stream.router, tags=["websocket"])


@app.get("/", include_in_schema=False)
def root():
    from fastapi.responses import HTMLResponse
    html = """
<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="3;url=/docs">
  <title>GCN Action Recognition API</title>
  <style>
    body { font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0;
           display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
    .card { background: #1e293b; border-radius: 12px; padding: 40px 48px; text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4); max-width: 480px; }
    h1 { font-size: 2rem; margin: 0 0 8px; }
    p  { color: #94a3b8; margin: 8px 0; }
    a  { display: inline-block; margin-top: 20px; padding: 12px 28px;
         background: #3b82f6; color: #fff; border-radius: 8px; text-decoration: none;
         font-weight: 600; transition: background .2s; }
    a:hover { background: #2563eb; }
    .badge { display: inline-block; background: #10b981; color: #fff;
             border-radius: 999px; padding: 2px 12px; font-size: .8rem; margin-bottom: 16px; }
  </style>
</head>
<body>
  <div class="card">
    <div class="badge">✅ Đang chạy</div>
    <h1>🤸 GCN Action Recognition</h1>
    <p>FastAPI backend đang hoạt động bình thường.</p>
    <p style="font-size:.85rem">Tự động chuyển tới <strong>/docs</strong> sau 3 giây...</p>
    <a href="/docs">📖 Mở Swagger UI / Docs</a>
  </div>
</body>
</html>
"""
    return HTMLResponse(content=html)
