from functools import lru_cache
from pathlib import Path
from .services.model_service import ModelService
from .services.profile_store import ProfileStore
from .services.recommender_service import RecommenderService
from .services.tmdb_service import TMDBService
from .services.user_service import UserService


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_runtime_db_path() -> Path:
    return get_project_root() / "data" / "runtime" / "profiles.db"


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    return ModelService(get_project_root())


@lru_cache(maxsize=1)
def get_profile_store() -> ProfileStore:
    return ProfileStore(get_runtime_db_path())


@lru_cache(maxsize=1)
def get_tmdb_service() -> TMDBService:
    return TMDBService(get_project_root(), get_runtime_db_path())


@lru_cache(maxsize=1)
def get_recommender_service() -> RecommenderService:
    return RecommenderService(get_model_service(), get_profile_store(), get_tmdb_service())


@lru_cache(maxsize=1)
def get_user_service() -> UserService:
    return UserService(get_project_root(), get_model_service(), get_profile_store(), get_tmdb_service())
