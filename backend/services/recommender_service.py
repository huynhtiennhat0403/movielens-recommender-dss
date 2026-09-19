from __future__ import annotations
from typing import Dict, Iterable, List, Sequence
from .model_service import ModelService
from .profile_store import ProfileStore
from .tmdb_service import TMDBService


class RecommenderService:
    def __init__(self, model_service: ModelService, store: ProfileStore, tmdb: TMDBService) -> None:
        self.model = model_service
        self.store = store
        self.tmdb = tmdb

    def _top_k_payload(self, movie_ids: Sequence[int], scores, top_k: int) -> List[Dict]:
        pairs = sorted(zip(movie_ids, scores), key=lambda x: float(x[1]), reverse=True)[:top_k]
        items = []
        for rank, (movie_id, score) in enumerate(pairs, start=1):
            items.append({
                **self.model.get_movie_metadata(int(movie_id)),
                "score": float(score),
                "rank": rank,
            })
        return self.tmdb.enrich_movies(items)

    def existing_user_recommendations(self, user_id: int, base_rated_movie_ids: Iterable[int], top_k: int = 10) -> Dict:
        self.model.validate_user_id(user_id)
        custom_ratings = self.store.get_existing_user_custom_ratings(user_id)

        excluded = set(int(x) for x in base_rated_movie_ids)
        excluded.update(custom_ratings.keys())
        candidates = sorted(self.model.train_seen_movie_ids - excluded)

        if custom_ratings:
            vectors = self.model.adapt_user(custom_ratings, existing_user_id=user_id, seed=user_id)
            scores = self.model.score_with_adapted_vectors(vectors, candidates)
            adapted = True
        else:
            scores = self.model.score_existing_user(user_id, candidates)
            adapted = False

        return {
            "subject_id": str(user_id),
            "subject_type": "existing_user",
            "top_k": top_k,
            "adapted": adapted,
            "recommendations": self._top_k_payload(candidates, scores, top_k),
        }

    def new_profile_recommendations(self, profile_id: str, top_k: int = 10) -> Dict:
        ratings = self.store.get_new_profile_ratings(profile_id)
        if len(ratings) < 5:
            raise ValueError("A new profile needs at least 5 ratings before recommendation.")

        vectors = self.model.adapt_user(
            ratings,
            existing_user_id=None,
            seed=abs(hash(profile_id)) % (2**32),
        )
        candidates = sorted(self.model.train_seen_movie_ids - set(ratings.keys()))
        scores = self.model.score_with_adapted_vectors(vectors, candidates)

        return {
            "subject_id": profile_id,
            "subject_type": "new_profile",
            "top_k": top_k,
            "adapted": True,
            "recommendations": self._top_k_payload(candidates, scores, top_k),
        }
