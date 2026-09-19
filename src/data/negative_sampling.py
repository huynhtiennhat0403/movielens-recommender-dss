"""Negative-sampling helpers for implicit-feedback recommendation."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Set

import numpy as np
import pandas as pd


def build_user_rated_movies(
    ratings_df: pd.DataFrame,
) -> Dict[int, Set[int]]:
    """
    Return all movie ids each user has ever rated.

    This is intentionally based on *all* ratings, not only positives, so
    an explicit 1-3 star interaction is not sampled later as "unseen".
    """
    required = {"user_id", "movie_id"}
    missing = required - set(ratings_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    return (
        ratings_df
        .groupby("user_id")["movie_id"]
        .apply(lambda x: set(x.astype(int)))
        .to_dict()
    )


def sample_training_negatives(
    train_positive_df: pd.DataFrame,
    user_rated_movies: Mapping[int, Set[int]],
    candidate_movie_ids: Iterable[int],
    movie_id_to_idx: Mapping[int, int],
    n_negatives: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Re-sample training negatives for one epoch.

    Negative candidates:
    - must be in the train-seen movie universe;
    - must never have been rated by the user in raw MovieLens data.

    Returns a shuffled dataframe with:
    user_id, movie_id, user_idx, movie_idx, label
    """
    if n_negatives <= 0:
        raise ValueError("n_negatives must be > 0.")

    required = {"user_id", "movie_id", "user_idx", "movie_idx"}
    missing = required - set(train_positive_df.columns)
    if missing:
        raise ValueError(f"train_positive_df missing columns: {sorted(missing)}")

    rng = np.random.default_rng(seed)

    candidate_movie_ids = np.asarray(
        sorted(set(int(x) for x in candidate_movie_ids)),
        dtype=np.int64
    )
    candidate_set = set(candidate_movie_ids.tolist())

    positive = train_positive_df[
        ["user_id", "movie_id", "user_idx", "movie_idx"]
    ].copy()
    positive["label"] = 1.0

    negative_frames = []

    for user_id, group in positive.groupby("user_id", sort=False):
        user_id = int(user_id)
        rated = set(int(x) for x in user_rated_movies.get(user_id, set()))

        available = np.asarray(
            sorted(candidate_set - rated),
            dtype=np.int64,
        )

        needed = len(group) * n_negatives

        if len(available) == 0:
            raise ValueError(f"User {user_id} has no available negative candidates.")

        sampled_ids = rng.choice(
            available,
            size=needed,
            replace=(needed > len(available))
        )

        user_idx = int(group["user_idx"].iloc[0])

        neg = pd.DataFrame({
            "user_id": np.full(needed, user_id, dtype=np.int64),
            "movie_id": sampled_ids,
            "user_idx": np.full(needed, user_idx, dtype=np.int64),
            "movie_idx": [
                int(movie_id_to_idx[int(mid)])
                for mid in sampled_ids
            ],
            "label": np.zeros(needed, dtype=np.float32),
        })

        negative_frames.append(neg)

    negatives = pd.concat(negative_frames, ignore_index=True)

    combined = pd.concat(
        [positive, negatives],
        ignore_index=True
    )

    combined = combined.sample(
        frac=1.0,
        random_state=seed
    ).reset_index(drop=True)

    return combined
