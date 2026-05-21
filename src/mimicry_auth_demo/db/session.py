"""Database engine and session factory."""
from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

_engine: Engine | None = None
_SessionFactory: sessionmaker | None = None


_SQLITE_FALLBACK = str(Path.home() / ".mimicry_auth" / "mimicry.db")


def _fix_pg_url(url: str) -> str:
    """postgresql:// → postgresql+psycopg:// (psycopg v3, которая у нас установлена)."""
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg" + url[len(prefix) - 3:]
    return url


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = os.environ.get("DATABASE_URL")
        if url:
            _engine = create_engine(_fix_pg_url(url), pool_pre_ping=True)
        else:
            # SQLite fallback для локальной разработки без Docker
            Path(_SQLITE_FALLBACK).parent.mkdir(parents=True, exist_ok=True)
            _engine = create_engine(f"sqlite:///{_SQLITE_FALLBACK}", connect_args={"check_same_thread": False})
    return _engine


def get_session() -> Session:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionFactory()


def init_db() -> None:
    """Создать все таблицы (идемпотентно)."""
    Base.metadata.create_all(get_engine())
