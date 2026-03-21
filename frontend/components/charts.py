# frontend/components/charts.py
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Consistent color palette
STATUS_COLORS = {"ok": "#16a34a", "wrong": "#dc2626", "uncertain": "#d97706"}

_LAYOUT_COMMON = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="system-ui, -apple-system, sans-serif", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#e5e7eb",
        borderwidth=1,
    ),
)


def _apply_common(fig):
    """Apply common layout styling."""
    fig.update_layout(**_LAYOUT_COMMON)
    fig.update_xaxes(gridcolor="#f3f4f6", gridwidth=1, zeroline=False)
    fig.update_yaxes(gridcolor="#f3f4f6", gridwidth=1, zeroline=False)
    return fig


def identity_distribution(df: pd.DataFrame):
    """Histogram of identity values, colored by status."""
    fig = px.histogram(
        df, x="identity", nbins=20,
        title="Identity 分布",
        color="status",
        color_discrete_map=STATUS_COLORS,
        labels={"identity": "Identity", "count": "样本数", "status": "状态"},
    )
    fig.update_layout(
        xaxis_title="Identity",
        yaxis_title="样本数",
        bargap=0.05,
    )
    fig.update_traces(opacity=0.85)
    return _apply_common(fig)


def coverage_distribution(df: pd.DataFrame):
    """Histogram of CDS coverage values, colored by status."""
    fig = px.histogram(
        df, x="cds_coverage", nbins=20,
        title="CDS 覆盖度分布",
        color="status",
        color_discrete_map=STATUS_COLORS,
        labels={"cds_coverage": "CDS Coverage", "count": "样本数", "status": "状态"},
    )
    fig.update_layout(
        xaxis_title="CDS Coverage",
        yaxis_title="样本数",
        bargap=0.05,
    )
    fig.update_traces(opacity=0.85)
    return _apply_common(fig)


def quality_scatter(df: pd.DataFrame):
    """Scatter plot: Identity vs CDS Coverage, sized by AA changes."""
    fig = px.scatter(
        df, x="identity", y="cds_coverage",
        color="status",
        color_discrete_map=STATUS_COLORS,
        size="AA 变异数",
        size_max=18,
        hover_data=["SID", "AA 变异数", "规则"],
        title="Identity vs CDS Coverage",
        labels={"identity": "Identity", "cds_coverage": "CDS Coverage", "status": "状态"},
    )
    fig.update_layout(
        xaxis_title="Identity",
        yaxis_title="CDS Coverage",
    )
    fig.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color="white")))
    return _apply_common(fig)


def status_pie(df: pd.DataFrame):
    """Pie chart of status distribution."""
    counts = df["status"].value_counts().reset_index()
    counts.columns = ["status", "count"]

    color_map = {s: STATUS_COLORS.get(s, "#9ca3af") for s in counts["status"]}

    fig = go.Figure(data=[go.Pie(
        labels=counts["status"].str.upper(),
        values=counts["count"],
        marker=dict(colors=[color_map[s] for s in counts["status"]]),
        hole=0.45,
        textinfo="label+value+percent",
        textfont=dict(size=13),
        hovertemplate="<b>%{label}</b><br>数量: %{value}<br>占比: %{percent}<extra></extra>",
    )])
    fig.update_layout(
        title="判读结果分布",
        showlegend=False,
        **{k: v for k, v in _LAYOUT_COMMON.items() if k != "legend"},
    )
    return fig
