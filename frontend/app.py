# frontend/app.py
import streamlit as st
import sys
from pathlib import Path

# Add backend to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="BioAgent MAX", page_icon="🧬", layout="wide")

st.title("BioAgent MAX")
st.markdown("**Sanger 测序 QC 智能分析平台**")
st.markdown("---")
st.markdown("👈 使用左侧菜单导航")
st.markdown("""
- **新建分析**: 上传文件或扫描目录，触发分析
- **分析结果**: 查看最近一次分析的详细结果
- **历史记录**: 浏览所有历史分析
- **参数设置**: 调整判读阈值
""")
