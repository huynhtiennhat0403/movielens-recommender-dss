from __future__ import annotations
import os
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import requests
from dotenv import load_dotenv


class TMDBService:
    API_BASE = "https://api.themoviedb.org/3"
    IMAGE_BASE = "https://image.tmdb.org/t/p"

    def __init__(self, project_root: Path, db_path: Path) -> None:
        self.project_root = Path(project_root)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        load_dotenv(self.project_root / ".env")
        self.access_token = os.getenv("TMDB_ACCESS_TOKEN", "").strip()
        self.enabled = bool(self.access_token)

        self.session = requests.Session()
        if self.enabled:
            self.session.headers.update({
                "Authorization": f"Bearer {self.access_token}",
                "accept": "application/json",
            })

        self._init_cache()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_cache(self) -> None:
        with self._connect() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS movie_metadata_cache (
                    movie_id INTEGER PRIMARY KEY,
                    tmdb_id INTEGER,
                    poster_path TEXT,
                    backdrop_path TEXT,
                    overview TEXT,
                    fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

    @staticmethod
    def _parse_title(title: str):
        value = str(title).strip()
        m = re.match(r"^(.*)\s+\((\d{4})\)$", value)
        if m:
            raw_title, year = m.group(1).strip(), int(m.group(2))
        else:
            raw_title, year = value, None

        article = re.match(r"^(.*),\s+(The|A|An)$", raw_title, flags=re.IGNORECASE)
        if article:
            raw_title = f"{article.group(2)} {article.group(1)}"
        return raw_title, year

    def _image_url(self, path: Optional[str], size: str) -> Optional[str]:
        return f"{self.IMAGE_BASE}/{size}{path}" if path else None

    def _cached(self, movie_id: int) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT tmdb_id, poster_path, backdrop_path, overview FROM movie_metadata_cache WHERE movie_id = ?",
                (int(movie_id),),
            ).fetchone()
        if row is None:
            return None
        return {
            "tmdb_id": int(row["tmdb_id"]) if row["tmdb_id"] is not None else None,
            "poster_url": self._image_url(row["poster_path"], "w500"),
            "backdrop_url": self._image_url(row["backdrop_path"], "w1280"),
            "overview": row["overview"] or None,
        }

    def _save(self, movie_id: int, result: Dict) -> None:
        with self._connect() as conn:
            conn.execute('''
                INSERT INTO movie_metadata_cache(
                    movie_id, tmdb_id, poster_path, backdrop_path, overview, fetched_at
                )
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(movie_id)
                DO UPDATE SET
                    tmdb_id = excluded.tmdb_id,
                    poster_path = excluded.poster_path,
                    backdrop_path = excluded.backdrop_path,
                    overview = excluded.overview,
                    fetched_at = CURRENT_TIMESTAMP
            ''', (
                int(movie_id),
                int(result["id"]),
                result.get("poster_path"),
                result.get("backdrop_path"),
                result.get("overview"),
            ))

    def get_movie_metadata(self, movie_id: int, title: str) -> Dict:
        cached = self._cached(movie_id)
        if cached is not None:
            return cached

        empty = {
            "tmdb_id": None,
            "poster_url": None,
            "backdrop_url": None,
            "overview": None,
        }
        if not self.enabled:
            return empty

        query, year = self._parse_title(title)
        params = {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": 1,
        }
        if year is not None:
            params["year"] = year

        try:
            r = self.session.get(
                f"{self.API_BASE}/search/movie",
                params=params,
                timeout=8,
            )
            r.raise_for_status()
            results = r.json().get("results", [])

            if not results and year is not None:
                params.pop("year", None)
                r = self.session.get(
                    f"{self.API_BASE}/search/movie",
                    params=params,
                    timeout=8,
                )
                r.raise_for_status()
                results = r.json().get("results", [])

            if not results:
                return empty

            result = results[0]
            self._save(movie_id, result)

            return {
                "tmdb_id": int(result["id"]),
                "poster_url": self._image_url(result.get("poster_path"), "w500"),
                "backdrop_url": self._image_url(result.get("backdrop_path"), "w1280"),
                "overview": result.get("overview") or None,
            }

        except requests.RequestException:
            return empty

    def enrich_movie(self, movie: Dict) -> Dict:
        return {
            **movie,
            **self.get_movie_metadata(movie["movie_id"], movie["title"]),
        }

    def enrich_movies(self, movies: Iterable[Dict], max_workers: int = 6) -> List[Dict]:
        items = list(movies)
        if not items:
            return []

        if not self.enabled:
            return [
                {
                    **m,
                    "tmdb_id": None,
                    "poster_url": None,
                    "backdrop_url": None,
                    "overview": None,
                }
                for m in items
            ]

        out = [None] * len(items)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            fut_map = {
                executor.submit(self.enrich_movie, movie): i
                for i, movie in enumerate(items)
            }
            for fut in as_completed(fut_map):
                i = fut_map[fut]
                try:
                    out[i] = fut.result()
                except Exception:
                    out[i] = {
                        **items[i],
                        "tmdb_id": None,
                        "poster_url": None,
                        "backdrop_url": None,
                        "overview": None,
                    }
        return out
