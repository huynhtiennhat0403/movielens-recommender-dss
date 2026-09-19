"""Generalized Matrix Factorization used for NCF pretraining."""

from __future__ import annotations

import torch
from torch import nn


class GMF(nn.Module):
    """Independent GMF predictor with its own user and item tables."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        output_bias: bool = True,
    ) -> None:
        super().__init__()

        if num_users <= 0 or num_items <= 0 or embedding_dim <= 0:
            raise ValueError("num_users, num_items and embedding_dim must be > 0.")

        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.embedding_dim = int(embedding_dim)
        self.output_bias = bool(output_bias)

        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)
        self.output = nn.Linear(self.embedding_dim, 1, bias=self.output_bias)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        if user_idx.shape != item_idx.shape:
            raise ValueError("user_idx and item_idx must have the same shape.")

        user_vec = self.user_embedding(user_idx)
        item_vec = self.item_embedding(item_idx)

        return self.logits_from_features(self.features_from_embeddings(user_vec, item_vec))

    def features_from_embeddings(self, user_vec: torch.Tensor, item_vec: torch.Tensor) -> torch.Tensor:
        """Return the GMF interaction vector for use by NCF."""
        return user_vec * item_vec

    def logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(features).squeeze(-1)
