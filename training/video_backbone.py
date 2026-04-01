from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn


@dataclass
class BackboneOutput:
    shared: torch.Tensor
    timing_logits: torch.Tensor
    strategy_logits: torch.Tensor | None = None
    template_logits: torch.Tensor | None = None
    state_pred: torch.Tensor | None = None


class TemporalPersonaBackbone(nn.Module):
    def __init__(
        self,
        sequence_dim: int,
        context_dim: int,
        hidden_dim: int,
        timing_dim: int,
        state_dim: int | None = None,
        strategy_dim: int | None = None,
        template_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.sequence_encoder = nn.Sequential(
            nn.Conv1d(sequence_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.timing_head = nn.Linear(hidden_dim, timing_dim)
        self.state_head = nn.Linear(hidden_dim, state_dim) if state_dim is not None else None
        self.strategy_head = nn.Linear(hidden_dim, strategy_dim) if strategy_dim is not None else None
        self.template_head = nn.Linear(hidden_dim, template_dim) if template_dim is not None else None

    def forward(self, sequence: torch.Tensor, context: torch.Tensor) -> BackboneOutput:
        seq = sequence.transpose(1, 2)
        seq_feat = self.sequence_encoder(seq).squeeze(-1)
        ctx_feat = self.context_encoder(context)
        shared = self.fusion(torch.cat([seq_feat, ctx_feat], dim=-1))
        return BackboneOutput(
            shared=shared,
            timing_logits=self.timing_head(shared),
            strategy_logits=self.strategy_head(shared) if self.strategy_head is not None else None,
            template_logits=self.template_head(shared) if self.template_head is not None else None,
            state_pred=self.state_head(shared) if self.state_head is not None else None,
        )
