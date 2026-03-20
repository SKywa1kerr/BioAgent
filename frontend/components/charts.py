# frontend/components/charts.py
import plotly.express as px
import pandas as pd


def identity_distribution(df: pd.DataFrame):
    """Histogram of identity values."""
    fig = px.histogram(df, x="identity", nbins=20, title="Identity 分布",
                       color="status", color_discrete_map={"ok": "#52c41a", "wrong": "#ff4d4f", "uncertain": "#faad14"})
    fig.update_layout(xaxis_title="Identity", yaxis_title="样本数")
    return fig


def coverage_distribution(df: pd.DataFrame):
    """Histogram of CDS coverage values."""
    fig = px.histogram(df, x="cds_coverage", nbins=20, title="CDS 覆盖度分布",
                       color="status", color_discrete_map={"ok": "#52c41a", "wrong": "#ff4d4f", "uncertain": "#faad14"})
    fig.update_layout(xaxis_title="CDS Coverage", yaxis_title="样本数")
    return fig
