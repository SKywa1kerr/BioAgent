"""SQLite database layer for BioAgent Desktop."""
from __future__ import annotations
import os
from pathlib import Path
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

class Base(DeclarativeBase):
    pass

_engine = None
_SessionFactory = None


def _default_db_path() -> str:
    """Store database in user's app data directory."""
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path.home() / ".local" / "share"
    app_dir = base / "BioAgent"
    app_dir.mkdir(parents=True, exist_ok=True)
    return str(app_dir / "bioagent.db")


def get_engine(db_url: str | None = None):
    global _engine
    if _engine is None:
        url = db_url or f"sqlite:///{_default_db_path()}"
        _engine = create_engine(url, connect_args={"check_same_thread": False})
    return _engine


def init_db(db_url: str | None = None):
    from .db_models import Analysis, Sample  # noqa: ensure registered
    engine = get_engine(db_url)
    Base.metadata.create_all(engine)


def get_session_factory():
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine())
    return _SessionFactory


@contextmanager
def db_session():
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
