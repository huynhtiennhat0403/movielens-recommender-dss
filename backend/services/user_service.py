from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import pandas as pd
from .model_service import ModelService
from .profile_store import ProfileStore
from .tmdb_service import TMDBService


class UserService:
    def __init__(self, project_root: Path, model_service: ModelService, store: ProfileStore, tmdb: TMDBService) -> None:
        self.project_root = Path(project_root)
        self.model = model_service
        self.store = store
        self.tmdb = tmdb

        raw_dir = self.project_root / "data" / "raw" / "ml-1m"
        self.raw_ratings = pd.read_csv(
            raw_dir / "ratings.dat",
            sep="::", engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"],
            encoding="latin-1",
        )
        self.raw_users = pd.read_csv(
            raw_dir / "users.dat",
            sep="::", engine="python",
            names=["user_id", "gender", "age", "occupation", "zip_code"],
            encoding="latin-1",
        )

    def existing_user_summary(self, user_id: int) -> Dict:
        self.model.validate_user_id(user_id)
        row = self.raw_users[self.raw_users["user_id"] == int(user_id)]
        profile = row.iloc[0].to_dict() if not row.empty else {}
        return {
            "user_id": int(user_id),
            "gender": profile.get("gender"),
            "age": int(profile["age"]) if profile.get("age") is not None else None,
            "occupation": int(profile["occupation"]) if profile.get("occupation") is not None else None,
            "zip_code": str(profile["zip_code"]) if profile.get("zip_code") is not None else None,
            "base_rating_count": int((self.raw_ratings["user_id"] == int(user_id)).sum()),
            "custom_rating_count": len(self.store.get_existing_user_custom_ratings(user_id)),
        }

    def existing_user_history(self, user_id: int) -> List[Dict]:
        self.model.validate_user_id(user_id)
        base = self.raw_ratings[self.raw_ratings["user_id"] == int(user_id)][["movie_id", "rating"]]
        base_map = {int(r.movie_id): int(r.rating) for r in base.itertuples(index=False)}
        custom = self.store.get_existing_user_custom_ratings(user_id)
        merged = dict(base_map)
        merged.update(custom)

        items = []
        for movie_id, rating in merged.items():
            if movie_id not in self.model.movie_id_to_idx:
                continue
            items.append({
                **self.model.get_movie_metadata(movie_id),
                "rating": int(rating),
                "source": "custom" if movie_id in custom else "movielens",
            })
        items.sort(key=lambda x: (x["source"] == "custom", x["rating"]), reverse=True)
        return self.tmdb.enrich_movies(items)

    def new_profile_history(self, profile_id: str) -> List[Dict]:
        ratings = self.store.get_new_profile_ratings(profile_id)
        items = [
            {
                **self.model.get_movie_metadata(movie_id),
                "rating": int(rating),
                "source": "custom",
            }
            for movie_id, rating in ratings.items()
        ]
        return self.tmdb.enrich_movies(items)
