"""SQLite persistence for profile snapshots and progress records."""
from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

RETENTION_SECONDS = 365 * 24 * 60 * 60
TOKEN_SECONDS = 30 * 24 * 60 * 60


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def initialize(db_path: Path) -> None:
    with connect(db_path) as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                phone TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL DEFAULT '',
                avatar_emoji TEXT NOT NULL DEFAULT '🙂',
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS otp_challenges (
                id TEXT PRIMARY KEY,
                phone TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_otp_phone_created ON otp_challenges(phone, created_at);
            CREATE TABLE IF NOT EXISTS otp_request_history (
                id TEXT PRIMARY KEY,
                phone TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_otp_request_phone_created ON otp_request_history(phone, created_at);
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS snapshots (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                answers_json TEXT NOT NULL,
                profile_json TEXT NOT NULL,
                estimate_json TEXT NOT NULL,
                outputs_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_snapshots_user_created ON snapshots(user_id, created_at DESC);
            CREATE TABLE IF NOT EXISTS career_goals (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('active', 'paused', 'completed')),
                gaps_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                expires_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_goals_user_updated ON career_goals(user_id, updated_at DESC);
            CREATE TABLE IF NOT EXISTS actions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                detail TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('not_started', 'in_progress', 'completed')),
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                expires_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_actions_user_updated ON actions(user_id, updated_at DESC);
        """)
        snapshot_columns = {row["name"] for row in con.execute("PRAGMA table_info(snapshots)")}
        if "outputs_json" not in snapshot_columns:
            con.execute("ALTER TABLE snapshots ADD COLUMN outputs_json TEXT NOT NULL DEFAULT '{}'")
        user_columns = {row["name"] for row in con.execute("PRAGMA table_info(users)")}
        if "display_name" not in user_columns:
            con.execute("ALTER TABLE users ADD COLUMN display_name TEXT NOT NULL DEFAULT ''")
        if "avatar_emoji" not in user_columns:
            con.execute("ALTER TABLE users ADD COLUMN avatar_emoji TEXT NOT NULL DEFAULT '🙂'")


def purge_expired(db_path: Path, now: float | None = None) -> None:
    current = now if now is not None else time.time()
    with connect(db_path) as con:
        con.execute("DELETE FROM users WHERE expires_at <= ?", (current,))
        con.execute("DELETE FROM otp_challenges WHERE expires_at <= ?", (current,))
        con.execute("DELETE FROM otp_request_history WHERE created_at <= ?", (current - 24 * 60 * 60,))
        con.execute("DELETE FROM sessions WHERE expires_at <= ?", (current,))
        con.execute("DELETE FROM snapshots WHERE expires_at <= ?", (current,))
        con.execute("DELETE FROM career_goals WHERE expires_at <= ?", (current,))
        con.execute("DELETE FROM actions WHERE expires_at <= ?", (current,))


def hash_secret(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def create_user(db_path: Path, phone: str, now: float | None = None) -> str:
    current = now if now is not None else time.time()
    user_id = secrets.token_urlsafe(18)
    with connect(db_path) as con:
        con.execute("INSERT INTO users(id, phone, created_at, expires_at) VALUES (?, ?, ?, ?)",
                    (user_id, phone, current, current + RETENTION_SECONDS))
    return user_id


def create_session(db_path: Path, user_id: str, now: float | None = None) -> str:
    current = now if now is not None else time.time()
    token = secrets.token_urlsafe(32)
    with connect(db_path) as con:
        con.execute("INSERT INTO sessions(token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                    (hash_secret(token), user_id, current, current + TOKEN_SECONDS))
        con.execute("UPDATE users SET expires_at = ? WHERE id = ?",
                    (current + RETENTION_SECONDS, user_id))
    return token


def json_field(value: str):
    return json.loads(value)
