from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

class Base(DeclarativeBase):
    pass

_engine = None
_SessionFactory = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine("sqlite:///./bioagent.db", connect_args={"check_same_thread": False})
    return _engine

def get_session_factory():
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine())
    return _SessionFactory

def init_db():
    from backend.db.models import Analysis, Sample  # noqa: ensure models registered
    Base.metadata.create_all(get_engine())
