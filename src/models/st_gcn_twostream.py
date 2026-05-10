import torch
import torch.nn as nn

from .st_gcn import STGCNModel as ST_GCN

class STGCNTwoStreamModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.origin_stream = ST_GCN(*args, **kwargs)
        self.motion_stream = ST_GCN(*args, **kwargs)

    def forward(self, x):
        N, C, T, V, M = x.size()
        
        # Calculate motion features (m) from temporal difference
        # We use torch.zeros(..., device=x.device) instead of hardcoding cuda.FloatTensor
        # to ensure it works properly on both GPU and CPU.
        m = torch.cat((
            torch.zeros(N, C, 1, V, M, device=x.device),
            x[:, :, 1:-1] - 0.5 * x[:, :, 2:] - 0.5 * x[:, :, :-2],
            torch.zeros(N, C, 1, V, M, device=x.device)
        ), 2)

        res = self.origin_stream(x) + self.motion_stream(m)
        return res
