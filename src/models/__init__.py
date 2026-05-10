"""
Model zoo for this repository.

We intentionally keep each model self-contained under `src/models/*` so that
new architectures can be added without changing existing training/inference
code paths.
"""

from .st_gcn import STGCNModel
from .st_gcn_twostream import STGCNTwoStreamModel

__all__ = ["STGCNModel", "STGCNTwoStreamModel"]
