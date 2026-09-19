from __future__ import annotations

import re
from typing import List, Optional

from fastapi import APIRouter, Query

from backend.dependencies import get_model_service, get_tmdb_service
from backend.schemas import MovieItem

router = APIRouter(prefix="/movies", tags=["Movies"])


def _extract_year(title: str) -> Optional[int]:
    match = re.search(r"\((\d{4})\)\s*$", str(title))
    return int(match.group(1)) if match else None


@router.get("", response_model=list[MovieItem])
def search_movies(
    query: str = Query(default="", max_length=100),
    genres: Optional[List[str]] = Query(default=None),
    year: Optional[int] = Query(default=None, ge=1900, le=2003),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    model = get_model_service()
    df = model.movie_mapping.copy()

    if query.strip():
        df = df[
            df["title"].str.contains(
                query.strip(),
                case=False,
                na=False,
                regex=False,
            )
        ]

    if year is not None:
        years = df["title"].astype(str).map(_extract_year)
        df = df[years == year]

    if genres:
        requested = [g.strip() for g in genres if g and g.strip()]

        # AND semantics: a movie must contain every selected genre.
        for genre in requested:
            df = df[
                df["genres"].astype(str).str.split("|").apply(
                    lambda values: genre in values
                )
            ]

    df = df.iloc[offset: offset + limit]

    items = [
        model.get_movie_metadata(int(mid))
        for mid in df["movie_id"].astype(int).tolist()
    ]
    return get_tmdb_service().enrich_movies(items)


@router.get("/genres", response_model=list[str])
def list_genres():
    model = get_model_service()

    genre_set = set()
    for value in model.movie_mapping["genres"].dropna().astype(str):
        for genre in value.split("|"):
            genre = genre.strip()
            if genre:
                genre_set.add(genre)

    return sorted(genre_set)
