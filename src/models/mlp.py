"""MLP branch used for NCF pretraining."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class MLP(nn.Module):
    """Independent MLP predictor with a paper-style halving tower."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 16,
        hidden_layers: Sequence[int] = (16, 8),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if num_users <= 0 or num_items <= 0 or embedding_dim <= 0:
            raise ValueError("num_users, num_items and embedding_dim must be > 0.")
        if not hidden_layers:
            raise ValueError("hidden_layers must contain at least one layer.")
        if any(int(size) <= 0 for size in hidden_layers):
            raise ValueError("All hidden layer sizes must be > 0.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.embedding_dim = int(embedding_dim)
        self.hidden_layers = tuple(int(x) for x in hidden_layers)
        self.dropout = float(dropout)

        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)

        layers = []
        input_dim = self.embedding_dim * 2

        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            input_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.output_dim = self.hidden_layers[-1]
        self.output = nn.Linear(self.output_dim, 1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)

        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

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
        x = torch.cat([user_vec, item_vec], dim=-1)
        return self.network(x)

    def logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(features).squeeze(-1)
