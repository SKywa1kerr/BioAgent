# frontend/pages/3_history.py
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.db.database import init_db, db_session
from backend.db.models import Analysis
from frontend.components.styles import inject_global_css, render_header, section_title

st.set_page_config(page_title="历史记录", page_icon="📋", layout="wide")

inject_global_css()
render_header("📋 历史记录", "浏览所有分析记录，追溯数据质量趋势")

init_db()
with db_session() as session:
    analyses_orm = session.query(Analysis).order_by(Analysis.created_at.desc()).all()
    # Extract data from ORM objects before session closes to avoid DetachedInstanceError
    analyses = []
    for a in analyses_orm:
        analyses.append({
            "name": a.name,
            "created_at": a.created_at.strftime("%Y-%m-%d %H:%M") if a.created_at else "",
            "total": a.total or 0,
            "ok_count": a.ok_count,
            "wrong_count": a.wrong_count,
            "uncertain_count": a.uncertain_count,
        })

if not analyses:
    st.info("暂无分析记录。请先在「新建分析」页面运行分析。")
    st.stop()

st.markdown(f"共 **{len(analyses)}** 条分析记录")
st.markdown("")

for a in analyses:
    total = a["total"]
    ok_pct = f"{a['ok_count'] / total * 100:.0f}%" if total > 0 else "-"

    st.markdown(f"""
    <div class="history-card">
        <div>
            <div class="hc-name">{a['name']}</div>
            <div class="hc-date">{a['created_at']} &nbsp;|&nbsp; {total} 个样本 &nbsp;|&nbsp; 通过率 {ok_pct}</div>
        </div>
        <div class="history-stats">
            <span class="hs-pill hs-ok">OK {a['ok_count']}</span>
            <span class="hs-pill hs-wrong">Wrong {a['wrong_count']}</span>
            <span class="hs-pill hs-uncertain">Uncertain {a['uncertain_count']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
