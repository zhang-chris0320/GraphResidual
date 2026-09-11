from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class AdapterStats:
    gate_mean: float
    gate_std: float
    alpha: float
    residual_norm_ratio: float
    preservation_cosine: float


class GraphResidualAdapter(nn.Module):
    """External residual fusion; it never owns or mutates PTM-Mamba."""

    def __init__(
        self,
        graph_dim: int = 64,
        hidden_dim: int = 768,
        projection_dim: int = 256,
        dropout: float = 0.1,
        node_types: Iterable[str] = ("mutation", "ptm", "protein"),
        alpha_init: float = 0.05,
        max_residual_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        self.graph_dim = int(graph_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_residual_ratio = float(max_residual_ratio)
        self.projectors = nn.ModuleDict(
            {
                kind: nn.Sequential(
                    nn.Linear(self.graph_dim, projection_dim),
                    nn.LayerNorm(projection_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(projection_dim, self.hidden_dim),
                )
                for kind in node_types
            }
        )
        self.gates = nn.ModuleDict(
            {kind: nn.Linear(self.hidden_dim * 2, self.hidden_dim) for kind in node_types}
        )
        init_logit = torch.logit(torch.tensor(float(alpha_init)).clamp(1e-4, 1 - 1e-4))
        self.alpha_logits = nn.ParameterDict(
            {kind: nn.Parameter(init_logit.clone()) for kind in node_types}
        )
        self.output_norms = nn.ModuleDict({kind: nn.LayerNorm(self.hidden_dim) for kind in node_types})

    def alpha(self, node_type: str) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logits[node_type])

    def forward(self, h: torch.Tensor, graph: torch.Tensor, node_type: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if h.shape[-1] != self.hidden_dim or graph.shape[-1] != self.graph_dim:
            raise ValueError(f"dimension mismatch h={tuple(h.shape)} graph={tuple(graph.shape)}")
        projected = self.projectors[node_type](graph)
        gate = torch.sigmoid(self.gates[node_type](torch.cat((h, projected), dim=-1)))
        raw_residual = self.alpha(node_type) * gate * projected
        h_norm = h.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        residual_norm = raw_residual.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        scale = torch.clamp(self.max_residual_ratio * h_norm / residual_norm, max=1.0)
        residual = raw_residual * scale
        z = self.output_norms[node_type](h + residual)
        aux = {
            "projected_graph": projected,
            "gate": gate,
            "alpha": self.alpha(node_type),
            "residual": residual,
            "residual_norm_ratio": residual.norm(dim=-1) / h_norm.squeeze(-1),
            "preservation_cosine": F.cosine_similarity(z, h, dim=-1),
        }
        return z, aux


class AdapterWithHead(nn.Module):
    def __init__(self, adapter: GraphResidualAdapter, head: nn.Module) -> None:
        super().__init__()
        self.adapter = adapter
        self.head = head


def preservation_loss(z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(z, h, dim=-1)).mean()

