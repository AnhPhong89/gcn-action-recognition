"""
Shared mutable state cho FastAPI app.
Dùng dict đơn giản thay vì global variables để dễ test.
"""
from typing import Any, Dict

app_state: Dict[str, Any] = {}
