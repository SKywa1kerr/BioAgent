# frontend/pages/4_settings.py
import streamlit as st
import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.core.rules import load_thresholds, DEFAULT_CONFIG

st.set_page_config(page_title="参数设置", page_icon="⚙️", layout="wide")
st.title("判读参数设置")

t = load_thresholds()

with st.form("thresholds_form"):
    st.subheader("测序失败")
    col1, col2 = st.columns(2)
    t["seq_failure_identity"] = col1.number_input("Identity 阈值", value=t["seq_failure_identity"], step=0.01, format="%.2f")
    t["seq_failure_min_length"] = col2.number_input("最短序列长度", value=t["seq_failure_min_length"], step=10)

    st.subheader("比对质量")
    col1, col2 = st.columns(2)
    t["identity_high"] = col1.number_input("高质量 Identity", value=t["identity_high"], step=0.01, format="%.2f")
    t["identity_medium_low"] = col2.number_input("重叠峰 Identity", value=t["identity_medium_low"], step=0.01, format="%.2f")

    st.subheader("CDS 覆盖度")
    col1, col2 = st.columns(2)
    t["cds_coverage_low"] = col1.number_input("低覆盖阈值", value=t["cds_coverage_low"], step=0.01, format="%.2f")
    t["cds_coverage_deletion"] = col2.number_input("片段缺失上界", value=t["cds_coverage_deletion"], step=0.01, format="%.2f")

    st.subheader("AA 突变数量")
    col1, col2 = st.columns(2)
    t["aa_overlap_severe"] = col1.number_input("重叠峰(严重)", value=t["aa_overlap_severe"], step=1)
    t["aa_mutation_max"] = col2.number_input("真实突变上限", value=t["aa_mutation_max"], step=1)

    st.subheader("生工重叠峰")
    col1, col2 = st.columns(2)
    t["synthetic_identity_min"] = col1.number_input("Identity 下界", value=t["synthetic_identity_min"], step=0.01, format="%.2f")
    t["synthetic_aa_min"] = col2.number_input("最低 AA 变异数", value=int(t["synthetic_aa_min"]), step=1)

    col_save, col_reset = st.columns(2)
    submitted = col_save.form_submit_button("保存配置", type="primary")
    if submitted:
        with open(DEFAULT_CONFIG, "w", encoding="utf-8") as f:
            yaml.dump({"thresholds": t}, f, allow_unicode=True, default_flow_style=False)
        st.success("配置已保存")
        st.rerun()
