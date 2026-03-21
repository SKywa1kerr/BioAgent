# frontend/pages/2_results.py
import streamlit as st
import pandas as pd
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.db.database import init_db, get_session_factory
from backend.db.models import Analysis, Sample
from frontend.components.charts import identity_distribution, coverage_distribution, quality_scatter, status_pie
from frontend.components.alignment_viewer import render_alignment
from frontend.components.styles import (
    inject_global_css, render_header, render_metric_cards,
    section_title, status_badge,
)

st.set_page_config(page_title="分析结果", page_icon="📊", layout="wide")

inject_global_css()
render_header("📊 分析结果", "查看 QC 判读结果、序列比对与质量分布")

init_db()
Session = get_session_factory()
session = Session()

# Get latest analysis or user-selected
analyses = session.query(Analysis).order_by(Analysis.created_at.desc()).limit(10).all()
if not analyses:
    st.info("暂无分析记录。请先在「新建分析」页面运行分析。")
    session.close()
    st.stop()

options = {a.id: f"{a.name}  ({a.total} 样本, {a.ok_count} OK / {a.wrong_count} Wrong / {a.uncertain_count} Uncertain)" for a in analyses}
selected_id = st.selectbox("选择分析记录", options.keys(), format_func=lambda x: options[x])

samples_orm = session.query(Sample).filter(Sample.analysis_id == selected_id).all()
analysis_orm = session.get(Analysis, selected_id)

# Extract data from ORM objects before closing session to avoid DetachedInstanceError
analysis_data = {
    "total": analysis_orm.total,
    "ok_count": analysis_orm.ok_count,
    "wrong_count": analysis_orm.wrong_count,
    "uncertain_count": analysis_orm.uncertain_count,
}
samples = []
for s in samples_orm:
    samples.append({
        "sid": s.sid, "status": s.status, "reason": s.reason,
        "identity": s.identity, "cds_coverage": s.cds_coverage,
        "aa_changes_n": s.aa_changes_n, "sub_count": s.sub_count,
        "ins_count": s.ins_count, "del_count": s.del_count,
        "rule_id": s.rule_id, "seq_length": s.seq_length,
        "avg_quality": s.avg_quality, "orientation": s.orientation,
        "aa_changes": s.aa_changes, "frameshift": s.frameshift,
        "ref_gapped": s.ref_gapped, "qry_gapped": s.qry_gapped,
    })
session.close()

if not samples:
    st.warning("该分析记录没有样本数据")
    st.stop()

# Summary metrics
render_metric_cards(analysis_data["total"], analysis_data["ok_count"], analysis_data["wrong_count"], analysis_data["uncertain_count"])

st.markdown("")

# -- Filter bar --
filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])
with filter_col1:
    status_filter = st.multiselect(
        "状态筛选",
        ["ok", "wrong", "uncertain"],
        default=["ok", "wrong", "uncertain"],
    )
with filter_col2:
    sort_by = st.selectbox("排序", ["SID", "Identity", "CDS Coverage", "AA 变异数"])

# Sample table
df = pd.DataFrame([{
    "SID": s["sid"],
    "状态": s["status"],
    "原因": s["reason"] or "",
    "Identity": round(s["identity"], 4) if s["identity"] else 0,
    "CDS Coverage": round(s["cds_coverage"], 3) if s["cds_coverage"] else 0,
    "AA 变异数": s["aa_changes_n"] or 0,
    "Sub": s["sub_count"] or 0, "Ins": s["ins_count"] or 0, "Del": s["del_count"] or 0,
    "规则": s["rule_id"],
    "序列长度": s["seq_length"], "平均质量": round(s["avg_quality"], 1) if s["avg_quality"] else 0,
    "identity": s["identity"], "cds_coverage": s["cds_coverage"], "status": s["status"],
} for s in samples])

# Apply filter
filtered_df = df[df["status"].isin(status_filter)]
if sort_by in filtered_df.columns:
    filtered_df = filtered_df.sort_values(sort_by, ascending=(sort_by == "SID"))

section_title("样本列表")

# Color-coded status column
def style_status(val):
    colors = {
        "ok": "background-color: #dcfce7; color: #15803d; font-weight: 600;",
        "wrong": "background-color: #fee2e2; color: #b91c1c; font-weight: 600;",
        "uncertain": "background-color: #fef3c7; color: #92400e; font-weight: 600;",
    }
    return colors.get(val, "")

display_cols = ["SID", "状态", "原因", "Identity", "CDS Coverage", "AA 变异数", "Sub", "Ins", "Del", "规则", "序列长度", "平均质量"]

_styler = filtered_df[display_cols].style
# pandas >= 2.1 uses map(), older uses applymap()
_apply_fn = getattr(_styler, "map", None) or getattr(_styler, "applymap")
styled_df = _apply_fn(
    style_status, subset=["状态"]
).format({
    "Identity": "{:.4f}",
    "CDS Coverage": "{:.3f}",
    "平均质量": "{:.1f}",
})

st.dataframe(
    styled_df,
    use_container_width=True,
    hide_index=True,
    height=min(400, 35 * len(filtered_df) + 40),
)

st.markdown("")

# Charts
section_title("数据可视化")

chart_tab1, chart_tab2 = st.tabs(["📈 分布图", "🥧 概览"])

with chart_tab1:
    chart1, chart2 = st.columns(2)
    with chart1:
        st.plotly_chart(identity_distribution(df), use_container_width=True)
    with chart2:
        st.plotly_chart(coverage_distribution(df), use_container_width=True)

    st.plotly_chart(quality_scatter(df), use_container_width=True)

with chart_tab2:
    pie_col, info_col = st.columns([1, 1])
    with pie_col:
        st.plotly_chart(status_pie(df), use_container_width=True)
    with info_col:
        st.markdown("")
        st.markdown("")
        total = len(df)
        if total > 0:
            ok_pct = analysis_data["ok_count"] / total * 100
            wrong_pct = analysis_data["wrong_count"] / total * 100
            uncertain_pct = analysis_data["uncertain_count"] / total * 100
            st.markdown(f"""
            **批次质量摘要**

            | 指标 | 值 |
            |------|------|
            | 通过率 | **{ok_pct:.1f}%** |
            | 异常率 | **{wrong_pct:.1f}%** |
            | 待定率 | **{uncertain_pct:.1f}%** |
            | 平均 Identity | **{df['identity'].mean():.4f}** |
            | 平均 CDS Coverage | **{df['cds_coverage'].mean():.3f}** |
            """)

# Sample detail expander
st.markdown("")
section_title("样本详情")

for s in samples:
    if s["status"] not in status_filter:
        continue
    badge = status_badge(s["status"])
    with st.expander(f"{s['sid']} — {s['status'].upper()}  {s['reason'] or ''}"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Identity", f"{s['identity']:.4f}" if s["identity"] else "N/A")
        c2.metric("CDS Coverage", f"{s['cds_coverage']:.3f}" if s["cds_coverage"] else "N/A")
        c3.metric("平均质量", f"{s['avg_quality']:.1f}" if s["avg_quality"] else "N/A")
        c4.metric("序列长度", f"{s['seq_length']} bp")

        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            st.markdown(f"**方向:** `{s['orientation']}`  &nbsp;&nbsp; **规则:** `R{s['rule_id']}`")
            st.markdown(f"**Substitutions:** {s['sub_count']}  &nbsp; **Insertions:** {s['ins_count']}  &nbsp; **Deletions:** {s['del_count']}")
        with detail_col2:
            if s["aa_changes"]:
                try:
                    aa = json.loads(s["aa_changes"])
                except (json.JSONDecodeError, TypeError):
                    aa = []
                if aa:
                    st.markdown(f"**AA 变异 ({len(aa)}):**")
                    st.code(" ".join(aa), language=None)

        render_alignment(s["ref_gapped"], s["qry_gapped"])
