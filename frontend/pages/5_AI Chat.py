# frontend/pages/5_ai_chat.py
import streamlit as st
import httpx
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.db.database import init_db, get_session_factory
from backend.db.models import Analysis, Sample
from frontend.components.styles import inject_global_css, render_header


def _load_env():
    """Load .env file from project root."""
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


_load_env()

st.set_page_config(page_title="AI 问答", page_icon="🤖", layout="wide")

inject_global_css()
render_header("🤖 AI 问答", "基于分析数据的智能解读助手")

init_db()

# --- Read config from env ---
ENV_API_KEY = os.environ.get("LLM_API_KEY", "")
ENV_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.chatanywhere.tech/v1")
ENV_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")


def call_openai_api(api_key: str, base_url: str, model: str, system: str, messages: list, max_tokens: int = 2048) -> str:
    """Call OpenAI-compatible API."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    api_messages = []
    if system:
        api_messages.append({"role": "system", "content": system})
    api_messages.extend(messages)

    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": api_messages,
    }
    resp = httpx.post(url, headers=headers, json=body, timeout=120)
    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# --- Init session state from env ---
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ENV_API_KEY
if "base_url" not in st.session_state:
    st.session_state["base_url"] = ENV_BASE_URL
if "model" not in st.session_state:
    st.session_state["model"] = ENV_MODEL

# --- API status (sidebar) ---
with st.sidebar:
    st.markdown("### API 配置")
    st.markdown("")

    if st.session_state["api_key"]:
        st.success("API Key 已配置")
    else:
        st.warning("未配置 API Key")

    st.markdown(f"**Base URL:** `{st.session_state['base_url']}`")
    st.markdown(f"**Model:** `{st.session_state['model']}`")

    st.markdown("")
    if st.button("🔗 测试连接", use_container_width=True):
        if not st.session_state["api_key"]:
            st.error("请先在 .env 中配置 LLM_API_KEY")
        else:
            with st.spinner("测试中..."):
                try:
                    reply = call_openai_api(
                        st.session_state["api_key"],
                        st.session_state["base_url"],
                        st.session_state["model"],
                        "", [{"role": "user", "content": "Hi"}], max_tokens=16,
                    )
                    st.success("连接成功！")
                except Exception as e:
                    st.error(f"连接失败: {e}")

    st.markdown("---")
    st.caption(
        "在项目根目录 `.env` 文件中配置：\n\n"
        "`LLM_API_KEY=sk-xxx`\n\n"
        "`LLM_BASE_URL=https://...`\n\n"
        "`LLM_MODEL=gpt-4o-mini`"
    )

    st.markdown("---")
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# --- Select analysis ---
Session = get_session_factory()
session = Session()
analyses = session.query(Analysis).order_by(Analysis.created_at.desc()).limit(20).all()

if not analyses:
    st.info("暂无分析记录。请先运行分析。")
    session.close()
    st.stop()

options = {a.id: f"{a.name} ({a.total} 样本, {a.ok_count} OK / {a.wrong_count} Wrong)" for a in analyses}
selected_id = st.selectbox("选择分析记录", options.keys(), format_func=lambda x: options[x])

samples_orm = session.query(Sample).filter(Sample.analysis_id == selected_id).all()
analysis_orm = session.get(Analysis, selected_id)

# Extract data from ORM objects before closing session to avoid DetachedInstanceError
analysis_data = {
    "name": analysis_orm.name, "status": analysis_orm.status,
    "total": analysis_orm.total, "ok_count": analysis_orm.ok_count,
    "wrong_count": analysis_orm.wrong_count, "uncertain_count": analysis_orm.uncertain_count,
}
samples_data = []
for s in samples_orm:
    samples_data.append({
        "sid": s.sid, "status": s.status, "reason": s.reason,
        "identity": s.identity, "cds_coverage": s.cds_coverage,
        "frameshift": s.frameshift, "aa_changes": s.aa_changes,
        "aa_changes_n": s.aa_changes_n, "sub_count": s.sub_count,
        "ins_count": s.ins_count, "del_count": s.del_count,
        "seq_length": s.seq_length, "avg_quality": s.avg_quality,
    })
session.close()

# Context header
st.markdown(
    f"""<div class="chat-header">
        <div>
            <div class="ch-title">当前分析: {analysis_data['name']}</div>
            <div class="ch-sub">{analysis_data['total']} 样本 &nbsp;|&nbsp; {analysis_data['ok_count']} OK &nbsp;|&nbsp; {analysis_data['wrong_count']} Wrong &nbsp;|&nbsp; {analysis_data['uncertain_count']} Uncertain</div>
        </div>
    </div>""",
    unsafe_allow_html=True,
)


def build_context(analysis_data, samples_data):
    """Build analysis context for AI."""
    lines = [
        f"当前分析: {analysis_data['name']}",
        f"状态: {analysis_data['status']} | 总计: {analysis_data['total']} | OK: {analysis_data['ok_count']} | Wrong: {analysis_data['wrong_count']} | Uncertain: {analysis_data['uncertain_count']}",
        "",
        "样本详情:",
    ]
    for s in samples_data:
        aa = json.loads(s["aa_changes"]) if s["aa_changes"] else []
        qual = f", avg_quality={s['avg_quality']:.1f}" if s["avg_quality"] else ""
        lines.append(
            f"  {s['sid']}: status={s['status']}, reason={s['reason'] or 'N/A'}, "
            f"identity={s['identity']:.4f}, cds_coverage={s['cds_coverage']:.3f}, "
            f"frameshift={s['frameshift']}, aa_changes={aa}, aa_n={s['aa_changes_n']}, "
            f"sub={s['sub_count']}, ins={s['ins_count']}, del={s['del_count']}, "
            f"seq_len={s['seq_length']}{qual}"
        )
    return "\n".join(lines)


SYSTEM_PROMPT = """你是 BioAgent MAX 的 AI 助手，专门帮助用户分析 Sanger 测序 QC 结果。

你拥有以下能力：
- 解读样本的判读结果（OK/Wrong/Uncertain）及其原因
- 解释规则引擎的判读逻辑（10 条规则，基于 identity、CDS 覆盖度、移码、AA 突变等指标）
- 对异常样本提供可能的生物学解释和建议（如建议重测、检查质粒构建等）
- 对整批数据的质量趋势进行总结

规则引擎概要：
- Rule 1: 多读段冲突 → wrong
- Rule 2: 测序失败（identity < 0.30 或序列太短）→ wrong
- Rule 3: 重叠峰+比对失败（identity < 0.85, aa_n > 40）→ wrong
- Rule 4: 重叠峰（identity < 0.85, 25 <= aa_n <= 40）→ wrong
- Rule 5: 移码错误 → wrong
- Rule 6: 真实 AA 突变（identity >= 0.95, 1-5 个突变）→ wrong
- Rule 7: 片段缺失（覆盖度 0.55-0.80, aa_n >= 5）→ wrong
- Rule 8: 生工重叠峰（identity 0.85-0.95, aa_n > 15）→ ok
- Rule 9: 未测通（覆盖度 < 0.55, 无突变）→ ok
- Rule 10: 正常（identity >= 0.95, 无突变无移码）→ ok

请用中文回答，简洁专业。"""

context = build_context(analysis_data, samples_data)

# --- Chat ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("针对当前分析结果提问..."):
    if not st.session_state.get("api_key"):
        st.error("请先在 .env 中配置 LLM_API_KEY")
        st.stop()

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build messages for API
    api_messages = [
        {"role": "user", "content": f"以下是当前分析数据:\n\n{context}\n\n请基于这些数据回答我后续的问题。"},
        {"role": "assistant", "content": "好的，我已了解当前分析数据。请提问，我会基于这些数据为你解答。"},
    ]
    for msg in st.session_state["messages"]:
        api_messages.append({"role": msg["role"], "content": msg["content"]})

    with st.chat_message("assistant"):
        try:
            with st.spinner("思考中..."):
                reply = call_openai_api(
                    st.session_state["api_key"],
                    st.session_state["base_url"],
                    st.session_state["model"],
                    SYSTEM_PROMPT,
                    api_messages,
                )
            st.markdown(reply)
            st.session_state["messages"].append({"role": "assistant", "content": reply})
        except Exception as e:
            st.error(f"API 调用失败: {e}\n\n请点击左侧「测试连接」确认 API 可用，或检查 .env 配置。")
