# frontend/components/styles.py
"""Shared CSS styles and UI helpers for BioAgent MAX."""
import streamlit as st

# -- Brand colors --
COLOR_PRIMARY = "#1a73e8"
COLOR_OK = "#16a34a"
COLOR_WRONG = "#dc2626"
COLOR_UNCERTAIN = "#d97706"


GLOBAL_CSS = """
<style>
/* ---- Header bar ---- */
.bioagent-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
    color: white;
    padding: 1.2rem 1.8rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}
.bioagent-header h1 {
    margin: 0;
    font-size: 1.5rem;
    font-weight: 700;
    color: white !important;
}
.bioagent-header .subtitle {
    font-size: 0.9rem;
    opacity: 0.85;
    margin-top: 2px;
    color: white !important;
}

/* ---- Status badges ---- */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
}
.badge-ok {
    background: #16a34a;
    color: white;
}
.badge-wrong {
    background: #dc2626;
    color: white;
}
.badge-uncertain {
    background: #d97706;
    color: white;
}

/* ---- Metric cards ---- */
.metric-card {
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: transform 0.15s;
}
.metric-card:hover {
    transform: translateY(-2px);
}
.metric-card .metric-value {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.1;
}
.metric-card .metric-label {
    font-size: 0.85rem;
    opacity: 0.8;
    margin-top: 4px;
    font-weight: 500;
}
.metric-card.total {
    background: rgba(30, 58, 95, 0.15);
    border: 2px solid rgba(30, 58, 95, 0.3);
}
.metric-card.total .metric-value { color: #60a5fa; }

.metric-card.ok {
    background: rgba(22, 163, 74, 0.15);
    border: 2px solid rgba(22, 163, 74, 0.3);
}
.metric-card.ok .metric-value { color: #4ade80; }

.metric-card.wrong {
    background: rgba(220, 38, 38, 0.15);
    border: 2px solid rgba(220, 38, 38, 0.3);
}
.metric-card.wrong .metric-value { color: #f87171; }

.metric-card.uncertain {
    background: rgba(217, 119, 6, 0.15);
    border: 2px solid rgba(217, 119, 6, 0.3);
}
.metric-card.uncertain .metric-value { color: #fbbf24; }

/* ---- Section titles ---- */
.section-title {
    font-size: 1.15rem;
    font-weight: 600;
    border-left: 4px solid #2563eb;
    padding-left: 12px;
    margin: 1.5rem 0 1rem 0;
}

/* ---- History card ---- */
.history-card {
    border: 1px solid rgba(128, 128, 128, 0.3);
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: border-color 0.15s;
}
.history-card:hover {
    border-color: #2563eb;
}
.history-card .hc-name {
    font-weight: 600;
    font-size: 1rem;
}
.history-card .hc-date {
    font-size: 0.82rem;
    opacity: 0.6;
}
.history-stats {
    display: flex;
    gap: 8px;
    align-items: center;
}
.history-stats .hs-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 10px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
    color: white;
}
.hs-ok { background: #16a34a; }
.hs-wrong { background: #dc2626; }
.hs-uncertain { background: #d97706; }

/* ---- Alignment viewer ---- */
.alignment-block {
    background: #0f172a;
    color: #e2e8f0;
    border-radius: 8px;
    padding: 1rem;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    font-size: 0.8rem;
    line-height: 1.5;
    overflow-x: auto;
    white-space: pre;
}
.aln-match { color: #86efac; }
.aln-mismatch { color: #f87171; font-weight: bold; }
.aln-gap { color: #fbbf24; }
.aln-label { color: #93c5fd; font-weight: bold; }
.aln-pos { color: #64748b; font-size: 0.75rem; }

/* ---- Feature cards on home page ---- */
.feature-card {
    border: 1px solid rgba(128, 128, 128, 0.3);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    height: 100%;
    transition: transform 0.15s, border-color 0.15s;
}
.feature-card:hover {
    transform: translateY(-3px);
    border-color: #2563eb;
}
.feature-card .fc-icon {
    font-size: 2.2rem;
    margin-bottom: 0.6rem;
}
.feature-card .fc-title {
    font-weight: 600;
    font-size: 1.05rem;
    margin-bottom: 0.4rem;
}
.feature-card .fc-desc {
    font-size: 0.85rem;
    opacity: 0.7;
}

/* ---- Settings form ---- */
.settings-section-title {
    font-weight: 600;
    font-size: 1rem;
    margin-bottom: 0.5rem;
    border-bottom: 2px solid rgba(128, 128, 128, 0.3);
    padding-bottom: 0.4rem;
}

/* ---- Chat styling ---- */
.chat-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
    color: white;
    padding: 0.8rem 1.2rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}
.chat-header .ch-title {
    font-weight: 600;
    font-size: 1rem;
    color: white !important;
}
.chat-header .ch-sub {
    font-size: 0.82rem;
    opacity: 0.8;
    color: white !important;
}

/* ---- Sidebar nav ---- */
[data-testid="stSidebarNav"] a span {
    font-size: 1.1rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em;
}

/* ---- Expander styling ---- */
[data-testid="stExpander"] {
    border: 1px solid rgba(128, 128, 128, 0.3) !important;
    border-radius: 10px !important;
    margin-bottom: 0.5rem !important;
}
</style>
"""


def inject_global_css():
    """Inject global CSS into the page."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def render_header(title: str, subtitle: str = ""):
    """Render the branded header bar."""
    sub_html = f'<div class="subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""<div class="bioagent-header">
            <div>
                <h1>{title}</h1>
                {sub_html}
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_metric_cards(total: int, ok: int, wrong: int, uncertain: int):
    """Render colored metric cards in 4 columns."""
    cols = st.columns(4)
    cards = [
        ("total", "Total", total),
        ("ok", "OK", ok),
        ("wrong", "Wrong", wrong),
        ("uncertain", "Uncertain", uncertain),
    ]
    for col, (cls, label, value) in zip(cols, cards):
        with col:
            st.markdown(
                f"""<div class="metric-card {cls}">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>""",
                unsafe_allow_html=True,
            )


def status_badge(status: str) -> str:
    """Return HTML for a status badge."""
    cls_map = {"ok": "badge-ok", "wrong": "badge-wrong", "uncertain": "badge-uncertain"}
    cls = cls_map.get(status, "badge-uncertain")
    label = status.upper()
    return f'<span class="badge {cls}">{label}</span>'


def section_title(text: str):
    """Render a styled section title."""
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)
