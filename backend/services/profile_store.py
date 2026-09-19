from __future__ import annotations
import sqlite3
import uuid
from pathlib import Path
from typing import Dict, List, Tuple


class ProfileStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS new_profiles (
                    profile_id TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS new_profile_ratings (
                    profile_id TEXT NOT NULL,
                    movie_id INTEGER NOT NULL,
                    rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(profile_id, movie_id)
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS existing_user_ratings (
                    user_id INTEGER NOT NULL,
                    movie_id INTEGER NOT NULL,
                    rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(user_id, movie_id)
                )
            ''')

    def create_profile(self, ratings: List[Tuple[int, int]]) -> str:
        profile_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute("INSERT INTO new_profiles(profile_id) VALUES (?)", (profile_id,))
            conn.executemany(
                "INSERT INTO new_profile_ratings(profile_id, movie_id, rating) VALUES (?, ?, ?)",
                [(profile_id, m, r) for m, r in ratings],
            )
        return profile_id

    def profile_exists(self, profile_id: str) -> bool:
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM new_profiles WHERE profile_id = ?",
                (profile_id,),
            ).fetchone() is not None

    def upsert_new_profile_rating(self, profile_id: str, movie_id: int, rating: int) -> None:
        if not self.profile_exists(profile_id):
            raise KeyError(f"Unknown profile_id: {profile_id}")
        with self._connect() as conn:
            conn.execute('''
                INSERT INTO new_profile_ratings(profile_id, movie_id, rating)
                VALUES (?, ?, ?)
                ON CONFLICT(profile_id, movie_id)
                DO UPDATE SET rating = excluded.rating, updated_at = CURRENT_TIMESTAMP
            ''', (profile_id, movie_id, rating))

    def get_new_profile_ratings(self, profile_id: str) -> Dict[int, int]:
        if not self.profile_exists(profile_id):
            raise KeyError(f"Unknown profile_id: {profile_id}")
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT movie_id, rating FROM new_profile_ratings WHERE profile_id = ?",
                (profile_id,),
            ).fetchall()
        return {int(r["movie_id"]): int(r["rating"]) for r in rows}

    def upsert_existing_user_rating(self, user_id: int, movie_id: int, rating: int) -> None:
        with self._connect() as conn:
            conn.execute('''
                INSERT INTO existing_user_ratings(user_id, movie_id, rating)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id, movie_id)
                DO UPDATE SET rating = excluded.rating, updated_at = CURRENT_TIMESTAMP
            ''', (user_id, movie_id, rating))

    def get_existing_user_custom_ratings(self, user_id: int) -> Dict[int, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT movie_id, rating FROM existing_user_ratings WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        return {int(r["movie_id"]): int(r["rating"]) for r in rows}
