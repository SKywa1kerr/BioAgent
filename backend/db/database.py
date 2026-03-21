from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from backend.config import get_config

class Base(DeclarativeBase):
    pass

_engine = None
_SessionFactory = None

def get_engine():
    global _engine
    if _engine is None:
        cfg = get_config()
        db_url = cfg.get("database", {}).get("url", "sqlite:///./bioagent.db")
        connect_args = {}
        if db_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        _engine = create_engine(db_url, connect_args=connect_args)
    return _engine

def get_session_factory():
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine())
    return _SessionFactory

def init_db():
    from backend.db.models import Analysis, Sample  # noqa: ensure models registered
    Base.metadata.create_all(get_engine())
