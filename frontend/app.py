# frontend/app.py
import streamlit as st
import sys
from pathlib import Path

# Add backend to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="BioAgent MAX", page_icon="🧬", layout="wide")

from frontend.components.styles import inject_global_css, render_header

inject_global_css()

render_header("🧬 BioAgent MAX", "Sanger 测序 QC 智能分析平台")

st.markdown("")

# Feature cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="fc-icon">🔬</div>
        <div class="fc-title">新建分析</div>
        <div class="fc-desc">上传文件或扫描目录<br>一键启动批量 QC 分析</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="fc-icon">📊</div>
        <div class="fc-title">分析结果</div>
        <div class="fc-desc">查看详细的 QC 报告<br>交互式图表与比对视图</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="fc-icon">📋</div>
        <div class="fc-title">历史记录</div>
        <div class="fc-desc">浏览所有历史分析<br>追溯数据质量趋势</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="feature-card">
        <div class="fc-icon">🤖</div>
        <div class="fc-title">AI 问答</div>
        <div class="fc-desc">AI 辅助解读结果<br>智能分析异常样本</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")
st.markdown("")

# Quick start section
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown('<div class="section-title">快速开始</div>', unsafe_allow_html=True)
    st.markdown("""
    1. 准备 **GenBank 参考序列**（`.gb` 文件）和 **AB1 测序文件**
    2. 进入 **新建分析** 页面，输入目录路径，点击「开始分析」
    3. 在 **分析结果** 页面查看判读结果、序列比对和质量图表
    4. 如有疑问，使用 **AI 问答** 深入解读异常样本
    """)

with col_right:
    st.markdown('<div class="section-title">判读规则概要</div>', unsafe_allow_html=True)
    st.markdown("""
    | 规则 | 判定 | 说明 |
    |------|------|------|
    | R1 | Wrong | 多读段冲突 |
    | R2 | Wrong | 测序失败 |
    | R5 | Wrong | 移码错误 |
    | R6 | Wrong | 真实 AA 突变 |
    | R8 | OK | 生工重叠峰 |
    | R10 | OK | 正常样本 |
    """)

# Footer
st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; color:#9ca3af; font-size:0.82rem;'>"
    "BioAgent MAX v1.0 &mdash; Sanger Sequencing QC Platform"
    "</div>",
    unsafe_allow_html=True,
)
