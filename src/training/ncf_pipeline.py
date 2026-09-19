"""Reproducible dynamic-sampling training for MF, GMF, MLP and NCF."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.negative_sampling import sample_training_negatives
from src.evaluation.metrics import hit_rate_at_k, ndcg_at_k
from src.models import GMF, MLP, NCF
from src.models.ncf import mlp_tower


def _loader(frame: pd.DataFrame, batch_size: int, device: torch.device) -> DataLoader:
    tensors = TensorDataset(
        torch.as_tensor(frame.user_idx.to_numpy(dtype=np.int64, copy=True), dtype=torch.long, device=device),
        torch.as_tensor(frame.movie_idx.to_numpy(dtype=np.int64, copy=True), dtype=torch.long, device=device),
        torch.as_tensor(frame.label.to_numpy(dtype=np.float32, copy=True), dtype=torch.float32, device=device),
    )
    return DataLoader(tensors, batch_size=batch_size, shuffle=True)


def _loss_for_frame(model: nn.Module, frame: pd.DataFrame, batch_size: int, device: torch.device) -> float:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total = 0.0
    count = 0
    with torch.no_grad():
        for users, items, labels in _loader(frame, batch_size, device):
            loss = criterion(model(users, items), labels)
            total += float(loss) * len(labels)
            count += len(labels)
    return total / max(count, 1)


def evaluate_candidates(model: nn.Module, candidates: pd.DataFrame, device: torch.device) -> dict[str, float]:
    """Evaluate deterministic candidate sets containing one positive first."""
    model.eval()
    metrics = {f"HR@{k}": [] for k in (5, 10, 20)}
    metrics.update({f"NDCG@{k}": [] for k in (5, 10, 20)})
    with torch.no_grad():
        for _, group in candidates.groupby("user_id", sort=True):
            users = torch.as_tensor(group.user_idx.to_numpy(dtype=np.int64, copy=True), dtype=torch.long, device=device)
            items = torch.as_tensor(group.movie_idx.to_numpy(dtype=np.int64, copy=True), dtype=torch.long, device=device)
            scores = model(users, items).detach().cpu().numpy()
            order = np.argsort(-scores)
            ranked_item_ids = group.movie_idx.to_numpy()[order].tolist()
            positive_item_ids = group.loc[group.is_positive.astype(bool), "movie_idx"].tolist()
            for k in (5, 10, 20):
                metrics[f"HR@{k}"].append(hit_rate_at_k(ranked_item_ids, positive_item_ids, k))
                metrics[f"NDCG@{k}"].append(ndcg_at_k(ranked_item_ids, positive_item_ids, k))
    return {key: float(np.mean(values)) if values else float("nan") for key, values in metrics.items()}


def fit_model(
    model: nn.Module,
    train_positive: pd.DataFrame,
    raw_history: Mapping[int, set[int]],
    candidate_movie_ids: list[int],
    movie_id_to_idx: Mapping[int, int],
    validation: pd.DataFrame,
    config: dict,
    checkpoint_path: Path,
    history_path: Path,
    optimizer_name: str = "adam",
    validation_candidates: pd.DataFrame | None = None,
) -> tuple[nn.Module, pd.DataFrame]:
    """Fit one model with fresh 4:1 negatives and ranking-based selection."""
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    optimizer_cls = torch.optim.SGD if optimizer_name.lower() == "sgd" else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config.get("weight_decay", 0.0)))
    best_score = float("-inf")
    stale = 0
    rows = []
    for epoch in range(1, int(config.get("epochs", 40)) + 1):
        sampled = sample_training_negatives(
            train_positive, raw_history, candidate_movie_ids, movie_id_to_idx,
            n_negatives=int(config.get("train_negatives", 4)), seed=int(config.get("seed", 42)) + epoch,
        )
        model.train()
        criterion = nn.BCEWithLogitsLoss()
        train_loss = 0.0
        seen = 0
        batches = _loader(sampled, int(config["batch_size"]), device)
        progress = tqdm(
            batches,
            desc=f"{model.__class__.__name__} epoch {epoch}/{config.get('epochs', 40)}",
            leave=False,
            unit="batch",
        )
        for users, items, labels in progress:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(users, items), labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss) * len(labels)
            seen += len(labels)
            progress.set_postfix(loss=f"{float(loss):.4f}")
        if validation_candidates is None:
            val_frame = validation.assign(label=1.0)
            val_loss = _loss_for_frame(model, val_frame, int(config["batch_size"]), device)
            val_metrics = {"HR@10": float("nan"), "NDCG@10": float("nan")}
            score = -val_loss
            selection_metric = "negative_val_loss"
        else:
            val_frame = validation_candidates.rename(columns={"is_positive": "label"}).copy()
            val_loss = _loss_for_frame(model, val_frame, int(config["batch_size"]), device)
            val_metrics = evaluate_candidates(model, validation_candidates, device)
            score = val_metrics["NDCG@10"]
            selection_metric = "NDCG@10"
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(seen, 1),
            "val_loss": val_loss,
            "val_hr10": val_metrics["HR@10"],
            "val_ndcg10": val_metrics["NDCG@10"],
        }
        rows.append(row)
        tqdm.write(
            f"{model.__class__.__name__} epoch {epoch}: "
            f"train_loss={row['train_loss']:.4f}, val_loss={val_loss:.4f}, "
            f"val_NDCG@10={row['val_ndcg10']:.4f}"
        )
        if score > best_score:
            best_score, stale = score, 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_config = {**config, "selection_metric": selection_metric, "best_score": best_score}
            torch.save({"model_state_dict": model.state_dict(), "config": checkpoint_config, "num_users": model.num_users, "num_items": model.num_items}, checkpoint_path)
        else:
            stale += 1
            if stale >= int(config.get("patience", 5)):
                tqdm.write(f"Early stopping at epoch {epoch}; best {selection_metric}={best_score:.4f}.")
                break
    history = pd.DataFrame(rows)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(history_path, index=False)
    best = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best["model_state_dict"])
    return model, history


def build_pretrained_ncf(gmf_checkpoint: Path, mlp_checkpoint: Path, ncf: NCF, device: torch.device, alpha: float = 0.5) -> NCF:
    """Load GMF/MLP checkpoints into NCF before SGD fine-tuning."""
    gmf_state = torch.load(gmf_checkpoint, map_location=device)
    mlp_state = torch.load(mlp_checkpoint, map_location=device)
    factor = ncf.predictive_factor
    gmf = GMF(ncf.num_users, ncf.num_items, embedding_dim=factor).to(device)
    mlp = MLP(ncf.num_users, ncf.num_items, embedding_dim=2 * factor, hidden_layers=mlp_tower(factor)).to(device)
    gmf.load_state_dict(gmf_state["model_state_dict"])
    mlp.load_state_dict(mlp_state["model_state_dict"])
    ncf.initialize_from_pretrained(gmf, mlp, alpha=alpha)
    return ncf
