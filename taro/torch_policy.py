"""Optional PyTorch implementation of the Phase 2 GNN actor-critic.

The module is import-safe without torch. Install torch to use ``TAROGNNPolicy``.
"""
from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = object


if torch is not None:
    class EdgeAwareGraphSAGE(nn.Module):
        def __init__(self, node_dim=6, edge_dim=6, hidden=64):
            super().__init__()
            self.node = nn.Linear(node_dim, hidden)
            self.edge = nn.Linear(edge_dim, hidden)
            self.update = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

        def forward(self, x, edge_index, edge_attr):
            h = torch.relu(self.node(x))
            src, dst = edge_index
            msg = h[src] + self.edge(edge_attr)
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, msg)
            deg = torch.zeros(h.shape[0], device=h.device)
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
            agg = agg / deg.clamp_min(1).unsqueeze(-1)
            return self.update(torch.cat([h, agg], dim=-1))


    class TAROGNNPolicy(nn.Module):
        def __init__(self, node_dim=6, edge_dim=6, hidden=64):
            super().__init__()
            self.encoder = EdgeAwareGraphSAGE(node_dim, edge_dim, hidden)
            self.actor = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

        def forward(self, node_x, edge_index, edge_attr, destination):
            h = self.encoder(node_x, edge_index, edge_attr)
            pooled = h.mean(dim=0, keepdim=True)
            logits = []
            for node in range(h.shape[0]):
                if node == destination:
                    continue
                logits.append(self.actor(torch.cat([h[node], h[destination]], dim=-1)))
            return torch.cat(logits) if logits else torch.empty(0, device=node_x.device), self.critic(pooled)
else:
    class TAROGNNPolicy:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ImportError("Install torch to use TAROGNNPolicy")
