# frontend/pages/2_results.py
import streamlit as st
import pandas as pd
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.db.database import init_db, get_session_factory
from backend.db.models import Analysis, Sample
from frontend.components.charts import identity_distribution, coverage_distribution
from frontend.components.alignment_viewer import render_alignment

st.set_page_config(page_title="分析结果", page_icon="📊", layout="wide")
st.title("分析结果")

init_db()
Session = get_session_factory()
session = Session()

# Get latest analysis or user-selected
analyses = session.query(Analysis).order_by(Analysis.created_at.desc()).limit(10).all()
if not analyses:
    st.info("暂无分析记录。请先在「新建分析」页面运行分析。")
    session.close()
    st.stop()

options = {a.id: f"{a.name} ({a.status}, {a.total} samples)" for a in analyses}
selected_id = st.selectbox("选择分析记录", options.keys(), format_func=lambda x: options[x])

samples = session.query(Sample).filter(Sample.analysis_id == selected_id).all()
analysis = session.get(Analysis, selected_id)
session.close()

if not samples:
    st.warning("该分析记录没有样本数据")
    st.stop()

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("总样本", analysis.total)
col2.metric("OK", analysis.ok_count)
col3.metric("Wrong", analysis.wrong_count)
col4.metric("Uncertain", analysis.uncertain_count)

# Sample table
df = pd.DataFrame([{
    "SID": s.sid, "状态": s.status, "原因": s.reason or "",
    "Identity": round(s.identity, 4) if s.identity else 0,
    "CDS Coverage": round(s.cds_coverage, 3) if s.cds_coverage else 0,
    "AA 变异数": s.aa_changes_n or 0, "规则": s.rule_id,
    "序列长度": s.seq_length, "平均质量": round(s.avg_quality, 1) if s.avg_quality else 0,
    "identity": s.identity, "cds_coverage": s.cds_coverage, "status": s.status,
} for s in samples])

st.dataframe(
    df[["SID", "状态", "原因", "Identity", "CDS Coverage", "AA 变异数", "规则", "序列长度", "平均质量"]],
    use_container_width=True,
    hide_index=True,
)

# Charts
chart1, chart2 = st.columns(2)
with chart1:
    st.plotly_chart(identity_distribution(df), use_container_width=True)
with chart2:
    st.plotly_chart(coverage_distribution(df), use_container_width=True)

# Sample detail expander
st.markdown("---")
st.subheader("样本详情")
for s in samples:
    with st.expander(f"{s.sid} — {s.status} {s.reason or ''}"):
        c1, c2, c3 = st.columns(3)
        c1.write(f"**Identity:** {s.identity:.4f}" if s.identity else "N/A")
        c2.write(f"**CDS Coverage:** {s.cds_coverage:.3f}" if s.cds_coverage else "N/A")
        c3.write(f"**方向:** {s.orientation}")
        c4, c5, c6 = st.columns(3)
        c4.write(f"**序列长度:** {s.seq_length} bp")
        c5.write(f"**平均质量:** {s.avg_quality:.1f}" if s.avg_quality else "N/A")
        c6.write(f"**规则:** #{s.rule_id}")

        if s.aa_changes:
            aa = json.loads(s.aa_changes)
            if aa:
                st.write(f"**AA 变异:** {' '.join(aa)}")

        st.write(f"**Substitutions:** {s.sub_count}  **Insertions:** {s.ins_count}  **Deletions:** {s.del_count}")

        render_alignment(s.ref_gapped, s.qry_gapped)
