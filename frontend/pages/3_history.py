# frontend/pages/3_history.py
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.db.database import init_db, get_session_factory
from backend.db.models import Analysis

st.set_page_config(page_title="历史记录", page_icon="📋", layout="wide")
st.title("历史记录")

init_db()
Session = get_session_factory()
session = Session()
analyses = session.query(Analysis).order_by(Analysis.created_at.desc()).all()
session.close()

if not analyses:
    st.info("暂无分析记录")
    st.stop()

for a in analyses:
    with st.container(border=True):
        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 2])
        col1.write(f"**{a.name}**")
        col2.write(f"✅ {a.ok_count}")
        col3.write(f"❌ {a.wrong_count}")
        col4.write(f"❓ {a.uncertain_count}")
        col5.write(f"{a.created_at}")
