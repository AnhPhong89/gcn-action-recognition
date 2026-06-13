"""
Health check route.
"""
from fastapi import APIRouter
from api.schemas import HealthResponse
from api.state import app_state

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    """Kiểm tra trạng thái server và model."""
    predictor = app_state.get("predictor")
    cfg = app_state.get("config", {})
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        device=app_state.get("device", "cpu"),
        class_names=cfg.get("class_names", []),
        window_size=cfg.get("window_size", 0),
    )
