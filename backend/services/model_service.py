from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn
from src.models import NCF


@dataclass
class AdaptedUserVectors:
    gmf: torch.Tensor
    mlp: torch.Tensor


class ModelService:
    def __init__(self, project_root: Path, checkpoint_name: str = "ncf_best.pth") -> None:
        self.project_root = Path(project_root)
        self.processed_dir = self.project_root / "data" / "processed"
        self.models_dir = self.project_root / "models"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.user_mapping = pd.read_csv(self.processed_dir / "user_mapping.csv")
        self.movie_mapping = pd.read_csv(self.processed_dir / "movie_mapping.csv")
        self.train_positive = pd.read_csv(self.processed_dir / "train.csv")

        self.user_id_to_idx = dict(zip(
            self.user_mapping["user_id"].astype(int),
            self.user_mapping["user_idx"].astype(int),
        ))
        self.movie_id_to_idx = dict(zip(
            self.movie_mapping["movie_id"].astype(int),
            self.movie_mapping["movie_idx"].astype(int),
        ))
        self.train_seen_movie_ids = set(self.train_positive["movie_id"].astype(int).unique())

        checkpoint_path = self.models_dir / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"NCF checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint["config"]

        self.model = NCF(
            num_users=int(checkpoint["num_users"]),
            num_items=int(checkpoint["num_items"]),
            predictive_factor=int(config.get("predictive_factor", 8)),
            dropout=float(config.get("dropout", 0.0)),
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.adaptation_steps = 40
        self.adaptation_lr = 0.05
        self.adaptation_negatives_per_positive = 4

    def validate_user_id(self, user_id: int) -> None:
        if int(user_id) not in self.user_id_to_idx:
            raise KeyError(f"Unknown existing user_id: {user_id}")

    def validate_movie_id(self, movie_id: int) -> None:
        if int(movie_id) not in self.movie_id_to_idx:
            raise KeyError(f"Movie {movie_id} is unavailable to the trained model.")

    def get_movie_metadata(self, movie_id: int) -> Dict:
        self.validate_movie_id(movie_id)
        row = self.movie_mapping[
            self.movie_mapping["movie_id"].astype(int) == int(movie_id)
        ].iloc[0]
        return {
            "movie_id": int(movie_id),
            "title": str(row["title"]),
            "genres": str(row["genres"]).split("|") if pd.notna(row["genres"]) else [],
        }

    @torch.no_grad()
    def score_existing_user(self, user_id: int, movie_ids: Sequence[int]) -> np.ndarray:
        self.validate_user_id(user_id)
        user_idx = self.user_id_to_idx[int(user_id)]
        movie_indices = [self.movie_id_to_idx[int(mid)] for mid in movie_ids]
        users = torch.full((len(movie_indices),), user_idx, dtype=torch.long, device=self.device)
        items = torch.tensor(movie_indices, dtype=torch.long, device=self.device)
        logits = self.model(users, items)
        return torch.sigmoid(logits).cpu().numpy()

    def _forward_with_user_vectors(self, gmf_user, mlp_user, movie_indices):
        item_gmf = self.model.gmf.item_embedding(movie_indices)
        item_mlp = self.model.mlp.item_embedding(movie_indices)

        gmf_batch = gmf_user.unsqueeze(0).expand(len(movie_indices), -1)
        mlp_batch = mlp_user.unsqueeze(0).expand(len(movie_indices), -1)

        gmf_features = gmf_batch * item_gmf
        mlp_features = self.model.mlp.network(torch.cat([mlp_batch, item_mlp], dim=-1))
        fused = torch.cat([gmf_features, mlp_features], dim=-1)
        return self.model.output(fused).squeeze(-1)

    def _initial_vectors_for_existing_user(self, user_id: int):
        self.validate_user_id(user_id)
        idx = self.user_id_to_idx[int(user_id)]
        return (
            self.model.gmf.user_embedding.weight[idx].detach().clone(),
            self.model.mlp.user_embedding.weight[idx].detach().clone(),
        )

    def _initial_vectors_for_new_user(self, ratings: Mapping[int, int]):
        pos = [int(mid) for mid in ratings if int(mid) in self.movie_id_to_idx]
        gmf_dim = self.model.gmf.embedding_dim
        mlp_dim = self.model.mlp.embedding_dim

        if not pos:
            return (
                torch.zeros(gmf_dim, device=self.device),
                torch.zeros(mlp_dim, device=self.device),
            )

        idx = torch.tensor([self.movie_id_to_idx[mid] for mid in pos], dtype=torch.long, device=self.device)
        with torch.no_grad():
            return (
                self.model.gmf.item_embedding(idx).mean(dim=0).clone(),
                self.model.mlp.item_embedding(idx).mean(dim=0).clone(),
            )

    def _build_adaptation_examples(self, ratings: Mapping[int, int], seed: int = 42):
        rated_ids = {int(mid) for mid in ratings if int(mid) in self.movie_id_to_idx}
        movie_ids, labels = [], []
        positive_count = 0

        for mid, rating in ratings.items():
            mid = int(mid)
            if mid not in self.movie_id_to_idx:
                continue
            movie_ids.append(mid); labels.append(1.0); positive_count += 1

        rng = np.random.default_rng(seed)
        available = np.array(sorted(self.train_seen_movie_ids - rated_ids), dtype=np.int64)
        n_sampled = positive_count * self.adaptation_negatives_per_positive

        if n_sampled > 0 and len(available) > 0:
            sampled = rng.choice(available, size=n_sampled, replace=n_sampled > len(available))
            movie_ids.extend(int(x) for x in sampled)
            labels.extend([0.0] * len(sampled))

        if not movie_ids:
            raise ValueError("Adaptation needs at least one valid movie rating.")

        return movie_ids, labels

    def adapt_user(self, ratings: Mapping[int, int], existing_user_id: int | None = None, seed: int = 42):
        for mid in ratings:
            self.validate_movie_id(int(mid))

        if existing_user_id is None:
            gmf_init, mlp_init = self._initial_vectors_for_new_user(ratings)
        else:
            gmf_init, mlp_init = self._initial_vectors_for_existing_user(existing_user_id)

        gmf_user = nn.Parameter(gmf_init)
        mlp_user = nn.Parameter(mlp_init)

        optimizer = torch.optim.Adam([gmf_user, mlp_user], lr=self.adaptation_lr)
        criterion = nn.BCEWithLogitsLoss()

        movie_ids, labels = self._build_adaptation_examples(ratings, seed)
        movie_indices = torch.tensor(
            [self.movie_id_to_idx[mid] for mid in movie_ids],
            dtype=torch.long,
            device=self.device,
        )
        targets = torch.tensor(labels, dtype=torch.float32, device=self.device)

        for _ in range(self.adaptation_steps):
            optimizer.zero_grad(set_to_none=True)
            logits = self._forward_with_user_vectors(gmf_user, mlp_user, movie_indices)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        return AdaptedUserVectors(gmf=gmf_user.detach(), mlp=mlp_user.detach())

    @torch.no_grad()
    def score_with_adapted_vectors(self, vectors: AdaptedUserVectors, movie_ids: Sequence[int]) -> np.ndarray:
        movie_indices = torch.tensor(
            [self.movie_id_to_idx[int(mid)] for mid in movie_ids],
            dtype=torch.long,
            device=self.device,
        )
        logits = self._forward_with_user_vectors(vectors.gmf, vectors.mlp, movie_indices)
        return torch.sigmoid(logits).cpu().numpy()
