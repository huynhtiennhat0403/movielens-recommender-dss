"""Matrix Factorization baseline for implicit-feedback recommendation."""

from __future__ import annotations

import torch
from torch import nn


class MatrixFactorization(nn.Module):
    """
    Simple implicit-feedback Matrix Factorization model.

    The model learns:
    - one embedding vector for each user;
    - one embedding vector for each item;
    - optional scalar user/item biases;
    - one global bias.

    The forward pass returns raw logits. Use
    ``torch.nn.BCEWithLogitsLoss`` during training.

    Parameters
    ----------
    num_users:
        Number of encoded users.
    num_items:
        Number of encoded items.
    embedding_dim:
        Latent dimension for user/item embeddings.
    use_bias:
        Whether to learn user and item biases.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        use_bias: bool = True,
    ) -> None:
        super().__init__()

        if num_users <= 0:
            raise ValueError("num_users must be > 0.")
        if num_items <= 0:
            raise ValueError("num_items must be > 0.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0.")

        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.embedding_dim = int(embedding_dim)
        self.use_bias = bool(use_bias)

        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)

        if self.use_bias:
            self.user_bias = nn.Embedding(self.num_users, 1)
            self.item_bias = nn.Embedding(self.num_items, 1)
        else:
            self.user_bias = None
            self.item_bias = None

        self.global_bias = nn.Parameter(torch.zeros(1))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize model parameters."""
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)

        if self.use_bias:
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)

        nn.init.zeros_(self.global_bias)

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute preference logits for pairs of users and items.

        Parameters
        ----------
        user_idx:
            Tensor containing encoded user ids.
        item_idx:
            Tensor containing encoded item ids.

        Returns
        -------
        torch.Tensor
            One logit per user-item pair.
        """
        if user_idx.shape != item_idx.shape:
            raise ValueError(
                "user_idx and item_idx must have the same shape. "
                f"Got {user_idx.shape} and {item_idx.shape}."
            )

        user_vec = self.user_embedding(user_idx)
        item_vec = self.item_embedding(item_idx)

        logits = (user_vec * item_vec).sum(dim=-1)
        logits = logits + self.global_bias

        if self.use_bias:
            logits = logits + self.user_bias(user_idx).squeeze(-1)
            logits = logits + self.item_bias(item_idx).squeeze(-1)

        return logits

    @torch.no_grad()
    def predict_proba(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Return sigmoid probabilities for user-item pairs."""
        return torch.sigmoid(self.forward(user_idx, item_idx))

    def extra_repr(self) -> str:
        return (
            f"num_users={self.num_users}, "
            f"num_items={self.num_items}, "
            f"embedding_dim={self.embedding_dim}, "
            f"use_bias={self.use_bias}"
        )
