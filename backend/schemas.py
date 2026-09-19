from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class MovieRatingInput(BaseModel):
    movie_id: int = Field(gt=0)
    rating: int = Field(ge=1, le=5)


class RatingCreateRequest(MovieRatingInput):
    pass


class NewProfileCreateRequest(BaseModel):
    ratings: List[MovieRatingInput] = Field(min_length=5)

    @field_validator("ratings")
    @classmethod
    def validate_unique_movies(cls, ratings: List[MovieRatingInput]):
        movie_ids = [r.movie_id for r in ratings]
        if len(movie_ids) != len(set(movie_ids)):
            raise ValueError("Each movie_id may appear only once.")
        return ratings


class MovieItem(BaseModel):
    movie_id: int
    title: str
    genres: List[str]
    tmdb_id: Optional[int] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    overview: Optional[str] = None


class RatingItem(MovieItem):
    rating: int
    source: str


class RecommendationItem(MovieItem):
    score: float
    rank: int


class NewProfileResponse(BaseModel):
    profile_id: str
    rating_count: int


class HistoryResponse(BaseModel):
    subject_id: str
    subject_type: str
    ratings: List[RatingItem]


class RecommendationResponse(BaseModel):
    subject_id: str
    subject_type: str
    top_k: int
    adapted: bool
    recommendations: List[RecommendationItem]


class ExistingUserSummary(BaseModel):
    user_id: int
    gender: Optional[str] = None
    age: Optional[int] = None
    occupation: Optional[int] = None
    zip_code: Optional[str] = None
    base_rating_count: int
    custom_rating_count: int


class HealthResponse(BaseModel):
    status: str
    device: str
    model: str
    tmdb_enabled: bool
