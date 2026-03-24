# frontend/pages/3_history.py
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.db.database import init_db, db_session
from backend.db.models import Analysis, Sample
from frontend.components.styles import inject_global_css, render_header, section_title

st.set_page_config(page_title="历史记录", page_icon="📋", layout="wide")

inject_global_css()
render_header("📋 历史记录", "浏览所有分析记录，追溯数据质量趋势")

init_db()

# Handle delete action
if "delete_id" in st.session_state and st.session_state.delete_id:
    del_id = st.session_state.delete_id
    st.session_state.delete_id = None
    with db_session() as session:
        analysis = session.get(Analysis, del_id)
        if analysis:
            session.query(Sample).filter(Sample.analysis_id == del_id).delete()
            session.delete(analysis)
    st.success("已删除分析记录")
    st.rerun()

with db_session() as session:
    analyses_orm = session.query(Analysis).order_by(Analysis.created_at.desc()).all()
    analyses = []
    for a in analyses_orm:
        analyses.append({
            "id": a.id,
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

    col_card, col_btn = st.columns([9, 1])

    with col_card:
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

    with col_btn:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        if st.button("🗑️", key=f"del_{a['id']}", help="删除此记录"):
            st.session_state[f"confirm_{a['id']}"] = True

    # Confirmation dialog
    if st.session_state.get(f"confirm_{a['id']}", False):
        with st.container():
            st.warning(f"确定要删除 **{a['name']}** 吗？此操作不可撤销。")
            c1, c2, _ = st.columns([1, 1, 6])
            with c1:
                if st.button("确认删除", key=f"yes_{a['id']}", type="primary"):
                    st.session_state[f"confirm_{a['id']}"] = False
                    st.session_state["delete_id"] = a["id"]
                    st.rerun()
            with c2:
                if st.button("取消", key=f"no_{a['id']}"):
                    st.session_state[f"confirm_{a['id']}"] = False
                    st.rerun()
