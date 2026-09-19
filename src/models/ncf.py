"""Pretrained Neural Collaborative Filtering model."""

from __future__ import annotations

import torch
from torch import nn

from .gmf import GMF
from .mlp import MLP


def mlp_tower(predictive_factor: int) -> tuple[int, ...]:
    """Return hidden widths for a 4f input tower: 4f -> 2f -> f."""
    if predictive_factor not in (8, 16, 32, 64):
        raise ValueError("predictive_factor must be one of 8, 16, 32 or 64")
    return (2 * predictive_factor, predictive_factor)


class NCF(nn.Module):
    """NCF fusion of separately pretrained GMF and MLP predictors."""

    def __init__(self, num_users: int, num_items: int, predictive_factor: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.predictive_factor = int(predictive_factor)
        self.gmf = GMF(num_users, num_items, embedding_dim=predictive_factor)
        self.mlp = MLP(
            num_users, num_items, embedding_dim=2 * predictive_factor,
            hidden_layers=mlp_tower(predictive_factor), dropout=dropout,
        )
        self.output = nn.Linear(2 * predictive_factor, 1)
        self.alpha = 0.5
        nn.init.zeros_(self.output.bias)

    def initialize_from_pretrained(self, gmf: GMF, mlp: MLP, alpha: float = 0.5) -> None:
        """Copy branch parameters and combine pretrained output weights."""
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.gmf.load_state_dict(gmf.state_dict())
        self.mlp.load_state_dict(mlp.state_dict())
        with torch.no_grad():
            self.output.weight[:, : self.predictive_factor].copy_(alpha * gmf.output.weight)
            self.output.weight[:, self.predictive_factor :].copy_((1.0 - alpha) * mlp.output.weight)
            self.output.bias.copy_(alpha * gmf.output.bias + (1.0 - alpha) * mlp.output.bias)
        self.alpha = float(alpha)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        if user_idx.shape != item_idx.shape:
            raise ValueError("user_idx and item_idx must have the same shape.")
        gmf_features = self.gmf.features_from_embeddings(
            self.gmf.user_embedding(user_idx), self.gmf.item_embedding(item_idx)
        )
        mlp_features = self.mlp.features_from_embeddings(
            self.mlp.user_embedding(user_idx), self.mlp.item_embedding(item_idx)
        )
        return self.output(torch.cat([gmf_features, mlp_features], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(user_idx, item_idx))
