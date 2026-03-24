# frontend/pages/4_settings.py
import streamlit as st
import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.core.rules import load_thresholds, DEFAULT_CONFIG
from frontend.components.styles import inject_global_css, render_header, section_title

st.set_page_config(page_title="参数设置", page_icon="⚙️", layout="wide")

inject_global_css()
render_header("⚙️ 参数设置", "调整 QC 判读阈值参数")

t = load_thresholds()

# Use columns to center the form
_, form_col, _ = st.columns([1, 3, 1])

with form_col:
    with st.form("thresholds_form"):

        # Section 1: Sequencing failure
        st.markdown('<div class="settings-section-title">🔴 测序失败阈值</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        t["seq_failure_identity"] = col1.number_input(
            "Identity 阈值", value=t["seq_failure_identity"], step=0.01, format="%.2f",
            help="低于此值判定为测序失败",
        )
        t["seq_failure_min_length"] = col2.number_input(
            "最短序列长度", value=t["seq_failure_min_length"], step=10,
            help="低于此长度判定为测序失败",
        )

        st.markdown("")

        # Section 2: Alignment quality
        st.markdown('<div class="settings-section-title">📐 比对质量阈值</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        t["identity_high"] = col1.number_input(
            "高质量 Identity", value=t["identity_high"], step=0.01, format="%.2f",
            help="高于此值认为比对质量良好",
        )
        t["identity_medium_low"] = col2.number_input(
            "重叠峰 Identity", value=t["identity_medium_low"], step=0.01, format="%.2f",
            help="低于此值可能存在重叠峰问题",
        )

        st.markdown("")

        # Section 3: CDS coverage
        st.markdown('<div class="settings-section-title">📏 CDS 覆盖度</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        t["cds_coverage_low"] = col1.number_input(
            "低覆盖阈值", value=t["cds_coverage_low"], step=0.01, format="%.2f",
            help="低于此值标记为低覆盖",
        )
        t["cds_coverage_deletion"] = col2.number_input(
            "片段缺失上界", value=t["cds_coverage_deletion"], step=0.01, format="%.2f",
            help="在此范围内可能存在片段缺失",
        )

        st.markdown("")

        # Section 4: AA mutations
        st.markdown('<div class="settings-section-title">🧪 AA 突变数量</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        t["aa_overlap_severe"] = col1.number_input(
            "重叠峰(严重)", value=t["aa_overlap_severe"], step=1,
            help="超过此数量判定为严重重叠峰",
        )
        t["aa_mutation_max"] = col2.number_input(
            "真实突变上限", value=t["aa_mutation_max"], step=1,
            help="不超过此数量的突变视为真实突变",
        )

        st.markdown("")

        # Section 5: Synthetic overlap
        st.markdown('<div class="settings-section-title">🏭 生工重叠峰</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        t["synthetic_identity_min"] = col1.number_input(
            "Identity 下界", value=t["synthetic_identity_min"], step=0.01, format="%.2f",
            help="生工重叠峰 identity 下界",
        )
        t["synthetic_aa_min"] = col2.number_input(
            "最低 AA 变异数", value=int(t["synthetic_aa_min"]), step=1,
            help="最低 AA 变异数",
        )

        st.markdown("")
        st.markdown("")

        col_save, col_reset = st.columns(2)
        submitted = col_save.form_submit_button("💾 保存配置", type="primary", width="full")
        if submitted:
            with open(DEFAULT_CONFIG, "w", encoding="utf-8") as f:
                yaml.dump({"thresholds": t}, f, allow_unicode=True, default_flow_style=False)
            st.success("配置已保存！")
            st.rerun()
