from sqlalchemy import text
from backend.db.database import init_db, db_session

def test_db_session_context_manager():
    init_db()
    with db_session() as session:
        result = session.execute(text("SELECT 1")).scalar()
        assert result == 1

def test_db_session_rollback_on_error():
    init_db()
    try:
        with db_session() as session:
            raise ValueError("test error")
    except ValueError:
        pass
    with db_session() as session:
        result = session.execute(text("SELECT 1")).scalar()
        assert result == 1
