"""Leakage-safe MovieLens implicit-feedback preparation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Set

import numpy as np
import pandas as pd


def load_movielens_ratings(path: Path) -> pd.DataFrame:
    """Load the native MovieLens 1M ``ratings.dat`` format."""
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )


def build_temporal_implicit_splits(
    ratings: pd.DataFrame,
    min_interactions: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, validation, test and the eligible raw interaction table.

    Every observed rating becomes label 1. For each eligible user, the last
    interaction is test, the previous one is validation, and older rows are
    train. Raw ``rating`` is retained for UI/history purposes.
    """
    required = {"user_id", "movie_id", "rating"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = ratings.copy()
    if "timestamp" not in frame.columns:
        frame["timestamp"] = np.arange(len(frame), dtype=np.int64)
    frame = frame.sort_values(["user_id", "timestamp", "movie_id"]).reset_index(drop=True)
    counts = frame.groupby("user_id").size()
    eligible_users = counts[counts >= int(min_interactions)].index
    eligible = frame[frame["user_id"].isin(eligible_users)].copy()
    eligible["label"] = 1.0
    eligible["rank_from_end"] = eligible.groupby("user_id").cumcount(ascending=False)
    test = eligible[eligible["rank_from_end"] == 0].copy()
    validation = eligible[eligible["rank_from_end"] == 1].copy()
    train = eligible[eligible["rank_from_end"] >= 2].copy()
    keep = [c for c in eligible.columns if c != "rank_from_end"]
    return train[keep], validation[keep], test[keep], eligible[keep]


def encode_splits(
    splits: Iterable[pd.DataFrame],
    user_mapping: pd.DataFrame,
    movie_mapping: pd.DataFrame,
) -> list[pd.DataFrame]:
    """Attach stable user_idx/movie_idx values to split dataframes."""
    user_map = dict(zip(user_mapping.user_id.astype(int), user_mapping.user_idx.astype(int)))
    movie_map = dict(zip(movie_mapping.movie_id.astype(int), movie_mapping.movie_idx.astype(int)))
    encoded = []
    for frame in splits:
        result = frame.copy()
        result["user_idx"] = result["user_id"].map(user_map)
        result["movie_idx"] = result["movie_id"].map(movie_map)
        result = result.dropna(subset=["user_idx", "movie_idx"]).copy()
        result[["user_idx", "movie_idx"]] = result[["user_idx", "movie_idx"]].astype(np.int64)
        encoded.append(result)
    return encoded


def build_user_history(ratings: pd.DataFrame) -> Dict[int, Set[int]]:
    """Map every raw user to every movie they have ever rated."""
    return (
        ratings.groupby("user_id")["movie_id"]
        .apply(lambda values: set(values.astype(int)))
        .to_dict()
    )


def build_eval_candidates(
    heldout: pd.DataFrame,
    raw_history: Mapping[int, Set[int]],
    candidate_movie_ids: Iterable[int],
    movie_id_to_idx: Mapping[int, int],
    negatives_per_user: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Create deterministic 1-positive + 100-unobserved-negative sets."""
    rng = np.random.default_rng(seed)
    universe_set = set(map(int, candidate_movie_ids))
    universe = np.asarray(sorted(universe_set), dtype=np.int64)
    rows = []
    for row in heldout.sort_values("user_id").itertuples(index=False):
        user_id = int(row.user_id)
        positive = int(row.movie_id)
        if positive not in universe_set:
            continue
        available = np.asarray(sorted(set(universe.tolist()) - raw_history.get(user_id, set())), dtype=np.int64)
        available = available[available != positive]
        if len(available) == 0:
            continue
        sampled = rng.choice(available, size=negatives_per_user, replace=len(available) < negatives_per_user)
        item_ids = np.concatenate(([positive], sampled))
        rows.extend({
            "user_id": user_id,
            "movie_id": int(movie_id),
            "user_idx": int(row.user_idx),
            "movie_idx": int(movie_id_to_idx[int(movie_id)]),
            "is_positive": int(i == 0),
            "candidate_rank": i,
        } for i, movie_id in enumerate(item_ids))
    return pd.DataFrame(rows)


def save_official_splits(
    raw_ratings_path: Path,
    processed_dir: Path,
    min_interactions: int = 3,
) -> dict[str, int]:
    """Build official implicit splits from a MovieLens-style ratings CSV."""
    ratings = load_movielens_ratings(raw_ratings_path) if raw_ratings_path.suffix == ".dat" else pd.read_csv(raw_ratings_path)
    train, validation, test, eligible = build_temporal_implicit_splits(ratings, min_interactions)
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train), ("validation", validation), ("test", test), ("eligible_interactions", eligible)):
        frame.to_csv(processed_dir / f"{name}.csv", index=False)
    return {
        "eligible_users": int(eligible.user_id.nunique()),
        "train_interactions": int(len(train)),
        "validation_interactions": int(len(validation)),
        "test_interactions": int(len(test)),
        "train_movies": int(train.movie_id.nunique()),
        "validation_cold_start_items": int((~validation.movie_id.isin(train.movie_id)).sum()),
        "test_cold_start_items": int((~test.movie_id.isin(train.movie_id)).sum()),
    }
