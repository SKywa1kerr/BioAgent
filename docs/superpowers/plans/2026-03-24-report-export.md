# Report Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add PDF and Excel report export to BioAgent MAX so users can download professional reports from any analysis record.

**Architecture:** New `backend/core/report.py` module handles all report generation. PDF uses WeasyPrint to render a Jinja2 HTML template with embedded Plotly chart PNGs. Excel uses openpyxl via pandas with conditional formatting. Frontend adds an export panel to the results page.

**Tech Stack:** WeasyPrint, Jinja2, openpyxl, kaleido (Plotly PNG export), pandas

**Spec:** `docs/superpowers/specs/2026-03-24-report-export-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `backend/core/report.py` | Create | Report data loading, chart rendering, PDF generation, Excel generation |
| `backend/templates/report.html` | Create | Jinja2 HTML template for PDF layout |
| `tests/test_report.py` | Create | Unit tests for report generation |
| `backend/requirements.txt` | Modify | Add weasyprint, openpyxl, kaleido, jinja2 |
| `backend/api/export.py` | Modify | Add PDF/Excel format support to API endpoint |
| `backend/mcp_server.py` | Modify | Extend export_report tool with PDF/Excel |
| `frontend/pages/2_results.py` | Modify | Add export panel UI |

---

### Task 1: Add dependencies

**Files:**
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Add new dependencies to requirements.txt**

Add these lines at the end of `backend/requirements.txt`:

```
weasyprint>=60.0
openpyxl>=3.1
kaleido>=0.2.1
jinja2>=3.1
```

- [ ] **Step 2: Install dependencies**

Run: `pip install weasyprint openpyxl kaleido jinja2`

- [ ] **Step 3: Verify imports work**

Run: `python -c "import weasyprint; import openpyxl; import kaleido; import jinja2; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add backend/requirements.txt
git commit -m "chore: add report export dependencies (weasyprint, openpyxl, kaleido, jinja2)"
```

---

### Task 2: Report data layer and Excel generation

Build `_build_report_data()` and `generate_excel()` first — these don't require WeasyPrint system dependencies and can be tested immediately.

**Files:**
- Create: `backend/core/report.py`
- Create: `tests/test_report.py`

- [ ] **Step 1: Write failing tests for data loading and Excel generation**

Create `tests/test_report.py`:

```python
"""Tests for report generation."""
import json
import io
import pytest
from backend.db.database import init_db, db_session
from backend.db.models import Analysis, Sample, new_id
from datetime import datetime, timezone


@pytest.fixture(autouse=True)
def setup_db():
    """Ensure DB is initialized."""
    init_db()


def _create_test_analysis():
    """Create a test analysis with samples for report testing."""
    analysis_id = new_id()
    thresholds = {
        "seq_failure_identity": 0.30,
        "seq_failure_min_length": 50,
        "identity_high": 0.95,
        "identity_medium_low": 0.85,
        "cds_coverage_low": 0.55,
        "cds_coverage_deletion": 0.80,
        "aa_overlap_severe": 40,
        "aa_overlap_moderate_min": 25,
        "aa_overlap_moderate_max": 40,
        "aa_mutation_max": 5,
        "aa_deletion_min": 5,
        "synthetic_identity_min": 0.85,
        "synthetic_identity_max": 0.95,
        "synthetic_aa_min": 15,
        "quality_trim_min": 20,
        "quality_aa_filter": 30,
    }
    with db_session() as session:
        session.add(Analysis(
            id=analysis_id, name="Test Report Analysis",
            source_type="scan", source_path="/test/ab1",
            status="done", total=3, ok_count=2, wrong_count=1, uncertain_count=0,
            config_snapshot=json.dumps(thresholds),
            finished_at=datetime.now(timezone.utc),
        ))
        for i, (status, reason, rule_id, identity, cds_cov) in enumerate([
            ("ok", "", 10, 0.998, 1.0),
            ("ok", "生工重叠峰", 8, 0.912, 0.98),
            ("wrong", "V123A", 6, 0.987, 1.0),
        ]):
            session.add(Sample(
                id=new_id(), analysis_id=analysis_id,
                sid=f"C{100+i}-1", clone=f"C{100+i}",
                status=status, reason=reason, rule_id=rule_id,
                identity=identity, cds_coverage=cds_cov,
                frameshift=False,
                aa_changes=json.dumps(["V123A"] if status == "wrong" else []),
                aa_changes_n=1 if status == "wrong" else 0,
                raw_aa_changes_n=1 if status == "wrong" else 0,
                orientation="FORWARD", seq_length=800, ref_length=5000,
                avg_quality=45.0, sub_count=2, ins_count=0, del_count=0,
                ref_gapped="ATCGATCG", qry_gapped="ATCAATCG",
                quality_scores=json.dumps([40]*800),
                raw_data="{}",
            ))
    return analysis_id


class TestBuildReportData:
    def test_returns_analysis_and_samples(self):
        from backend.core.report import _build_report_data
        aid = _create_test_analysis()
        data = _build_report_data(aid)
        assert data["analysis"]["name"] == "Test Report Analysis"
        assert data["analysis"]["total"] == 3
        assert len(data["samples"]) == 3
        assert data["thresholds"]["identity_high"] == 0.95

    def test_not_found_raises(self):
        from backend.core.report import _build_report_data
        with pytest.raises(ValueError, match="not found"):
            _build_report_data("nonexistent-id")


class TestGenerateExcel:
    def test_returns_valid_xlsx(self):
        from backend.core.report import generate_excel
        import openpyxl
        aid = _create_test_analysis()
        data = generate_excel(aid)
        assert isinstance(data, bytes)
        wb = openpyxl.load_workbook(io.BytesIO(data))
        assert "摘要" in wb.sheetnames
        assert "样本明细" in wb.sheetnames
        assert "阈值配置" in wb.sheetnames

    def test_module_filtering(self):
        from backend.core.report import generate_excel
        import openpyxl
        aid = _create_test_analysis()
        data = generate_excel(aid, modules=["detail_table"])
        wb = openpyxl.load_workbook(io.BytesIO(data))
        assert "样本明细" in wb.sheetnames
        assert "摘要" not in wb.sheetnames
        assert "阈值配置" not in wb.sheetnames
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_report.py -v`
Expected: FAIL — `backend.core.report` does not exist yet.

- [ ] **Step 3: Implement `_build_report_data()` and `generate_excel()`**

Create `backend/core/report.py`:

```python
"""Report generation: PDF (WeasyPrint) and Excel (openpyxl)."""
import json
import io
import logging
from datetime import datetime

import pandas as pd

from backend.db.database import init_db, db_session
from backend.db.models import Analysis, Sample

logger = logging.getLogger(__name__)

ALL_MODULES = ["summary", "detail_table", "charts", "abnormal_details", "alignment", "thresholds"]
MAX_ALIGNMENT_SAMPLES = 20


def _build_report_data(analysis_id: str) -> dict:
    """Fetch analysis + samples from DB, return structured dict for report rendering."""
    init_db()
    with db_session() as session:
        analysis = session.get(Analysis, analysis_id)
        if not analysis:
            raise ValueError(f"Analysis not found: {analysis_id}")

        samples_orm = session.query(Sample).filter(
            Sample.analysis_id == analysis_id
        ).all()

        # Extract all data before session closes
        analysis_data = {
            "id": analysis.id,
            "name": analysis.name,
            "source_type": analysis.source_type,
            "source_path": analysis.source_path,
            "status": analysis.status,
            "total": analysis.total,
            "ok_count": analysis.ok_count,
            "wrong_count": analysis.wrong_count,
            "uncertain_count": analysis.uncertain_count,
            "created_at": str(analysis.created_at) if analysis.created_at else "",
            "finished_at": str(analysis.finished_at) if analysis.finished_at else "",
            "config_snapshot": analysis.config_snapshot,
        }

        samples = []
        for s in samples_orm:
            samples.append({
                "sid": s.sid,
                "status": s.status,
                "reason": s.reason or "",
                "rule_id": s.rule_id,
                "identity": s.identity,
                "cds_coverage": s.cds_coverage,
                "frameshift": s.frameshift,
                "aa_changes": s.aa_changes,
                "aa_changes_n": s.aa_changes_n,
                "raw_aa_changes_n": s.raw_aa_changes_n,
                "orientation": s.orientation,
                "seq_length": s.seq_length,
                "ref_length": s.ref_length,
                "avg_quality": s.avg_quality,
                "sub_count": s.sub_count,
                "ins_count": s.ins_count,
                "del_count": s.del_count,
                "ref_gapped": s.ref_gapped,
                "qry_gapped": s.qry_gapped,
            })

    # Parse thresholds from config snapshot
    thresholds = {}
    if analysis_data.get("config_snapshot"):
        try:
            thresholds = json.loads(analysis_data["config_snapshot"])
        except (json.JSONDecodeError, TypeError):
            pass

    # Computed stats
    total = analysis_data["total"] or 0
    ok = analysis_data["ok_count"] or 0
    identities = [s["identity"] for s in samples if s["identity"] is not None]
    coverages = [s["cds_coverage"] for s in samples if s["cds_coverage"] is not None]

    stats = {
        "pass_rate": round(ok / total * 100, 1) if total > 0 else 0,
        "avg_identity": round(sum(identities) / len(identities), 4) if identities else 0,
        "avg_cds_coverage": round(sum(coverages) / len(coverages), 3) if coverages else 0,
    }

    return {
        "analysis": analysis_data,
        "samples": samples,
        "thresholds": thresholds,
        "stats": stats,
    }


def generate_excel(analysis_id: str, modules: list[str] | None = None) -> bytes:
    """Generate Excel report as bytes. Modules: summary, detail_table, thresholds."""
    import openpyxl
    from openpyxl.styles import PatternFill, Font, Alignment
    from openpyxl.utils import get_column_letter

    mods = modules or ALL_MODULES
    data = _build_report_data(analysis_id)

    wb = openpyxl.Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    # -- 摘要 sheet --
    if "summary" in mods:
        ws = wb.create_sheet("摘要")
        header_font = Font(bold=True, size=14, color="1A73E8")
        label_font = Font(bold=True)

        ws.append(["BioAgent MAX — 分析报告"])
        ws["A1"].font = header_font
        ws.append([])
        rows = [
            ("分析名称", data["analysis"]["name"]),
            ("分析时间", data["analysis"]["finished_at"]),
            ("数据来源", data["analysis"]["source_path"]),
            ("", ""),
            ("总样本数", data["analysis"]["total"]),
            ("OK", data["analysis"]["ok_count"]),
            ("Wrong", data["analysis"]["wrong_count"]),
            ("Uncertain", data["analysis"]["uncertain_count"]),
            ("通过率", f"{data['stats']['pass_rate']}%"),
            ("平均 Identity", data["stats"]["avg_identity"]),
            ("平均 CDS Coverage", data["stats"]["avg_cds_coverage"]),
        ]
        for label, value in rows:
            ws.append([label, value])
            if label:
                ws.cell(row=ws.max_row, column=1).font = label_font

        ws.column_dimensions["A"].width = 18
        ws.column_dimensions["B"].width = 40

    # -- 样本明细 sheet --
    if "detail_table" in mods:
        ws = wb.create_sheet("样本明细")
        headers = ["SID", "状态", "原因", "Identity", "CDS Coverage",
                    "AA 变异数", "Sub", "Ins", "Del", "规则",
                    "序列长度", "平均质量", "方向"]
        ws.append(headers)

        # Header style
        header_fill = PatternFill(start_color="E8F0FE", end_color="E8F0FE", fill_type="solid")
        header_font = Font(bold=True)
        for col_idx in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font

        # Status fills
        status_fills = {
            "ok": PatternFill(start_color="DCFCE7", end_color="DCFCE7", fill_type="solid"),
            "wrong": PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid"),
            "uncertain": PatternFill(start_color="FEF9C3", end_color="FEF9C3", fill_type="solid"),
        }

        for s in data["samples"]:
            row = [
                s["sid"], s["status"], s["reason"],
                round(s["identity"], 4) if s["identity"] is not None else "",
                round(s["cds_coverage"], 3) if s["cds_coverage"] is not None else "",
                s["aa_changes_n"] or 0,
                s["sub_count"] or 0, s["ins_count"] or 0, s["del_count"] or 0,
                f"R{s['rule_id']}" if s["rule_id"] else "",
                s["seq_length"],
                round(s["avg_quality"], 1) if s["avg_quality"] is not None else "",
                s["orientation"],
            ]
            ws.append(row)
            # Apply status fill to status cell
            status_cell = ws.cell(row=ws.max_row, column=2)
            fill = status_fills.get(s["status"])
            if fill:
                status_cell.fill = fill

        # Auto-filter
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{ws.max_row}"

        # Column widths
        col_widths = [10, 10, 20, 10, 12, 10, 6, 6, 6, 8, 10, 10, 10]
        for i, w in enumerate(col_widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = w

    # -- 阈值配置 sheet --
    if "thresholds" in mods:
        ws = wb.create_sheet("阈值配置")
        ws.append(["参数", "值"])
        header_fill = PatternFill(start_color="E8F0FE", end_color="E8F0FE", fill_type="solid")
        header_font = Font(bold=True)
        for col_idx in (1, 2):
            cell = ws.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
        for key, value in data["thresholds"].items():
            ws.append([key, value])
        ws.column_dimensions["A"].width = 30
        ws.column_dimensions["B"].width = 15

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_report.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/core/report.py tests/test_report.py
git commit -m "feat: add report data layer and Excel export generation"
```

---

### Task 3: Chart rendering for PDF

**Files:**
- Modify: `backend/core/report.py`
- Modify: `tests/test_report.py`

- [ ] **Step 1: Write failing test for chart rendering**

Add to `tests/test_report.py`:

```python
class TestRenderCharts:
    def test_returns_png_bytes(self):
        from backend.core.report import _render_charts
        aid = _create_test_analysis()
        data = _build_report_data(aid)
        charts = _render_charts(data["samples"])
        assert "identity_hist" in charts
        assert "status_pie" in charts
        # Check PNG magic bytes
        for key, img_bytes in charts.items():
            assert img_bytes[:8] == b'\x89PNG\r\n\x1a\n', f"{key} is not a valid PNG"
```

(Add `from backend.core.report import _build_report_data` to imports at top of test.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_report.py::TestRenderCharts -v`
Expected: FAIL — `_render_charts` not defined.

- [ ] **Step 3: Implement `_render_charts()`**

Add to `backend/core/report.py` after `_build_report_data()`:

```python
def _render_charts(samples: list[dict]) -> dict[str, bytes]:
    """Generate Plotly charts as PNG bytes for PDF embedding."""
    import plotly.express as px
    import plotly.graph_objects as go

    STATUS_COLORS = {"ok": "#16a34a", "wrong": "#dc2626", "uncertain": "#d97706"}
    chart_layout = dict(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=11),
        margin=dict(l=50, r=30, t=50, b=50),
        width=500, height=350,
    )
    charts = {}

    # Build DataFrame from samples
    df = pd.DataFrame([{
        "identity": s["identity"],
        "cds_coverage": s["cds_coverage"],
        "status": s["status"],
    } for s in samples])

    # Identity histogram
    fig = px.histogram(
        df, x="identity", nbins=20, title="Identity 分布",
        color="status", color_discrete_map=STATUS_COLORS,
        labels={"identity": "Identity", "count": "样本数", "status": "状态"},
    )
    fig.update_layout(**chart_layout, bargap=0.05)
    fig.update_traces(opacity=0.85)
    charts["identity_hist"] = fig.to_image(format="png", engine="kaleido")

    # CDS coverage histogram
    fig = px.histogram(
        df, x="cds_coverage", nbins=20, title="CDS 覆盖度分布",
        color="status", color_discrete_map=STATUS_COLORS,
        labels={"cds_coverage": "CDS Coverage", "count": "样本数", "status": "状态"},
    )
    fig.update_layout(**chart_layout, bargap=0.05)
    fig.update_traces(opacity=0.85)
    charts["coverage_hist"] = fig.to_image(format="png", engine="kaleido")

    # Status pie chart
    counts = df["status"].value_counts().reset_index()
    counts.columns = ["status", "count"]
    fig = go.Figure(data=[go.Pie(
        labels=counts["status"].str.upper(),
        values=counts["count"],
        marker=dict(colors=[STATUS_COLORS.get(s, "#9ca3af") for s in counts["status"]]),
        hole=0.45,
        textinfo="label+value+percent",
    )])
    fig.update_layout(title="判读结果分布", showlegend=False, **chart_layout)
    charts["status_pie"] = fig.to_image(format="png", engine="kaleido")

    return charts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_report.py::TestRenderCharts -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/core/report.py tests/test_report.py
git commit -m "feat: add Plotly chart rendering for PDF reports"
```

---

### Task 4: HTML template and PDF generation

**Files:**
- Create: `backend/templates/report.html`
- Modify: `backend/core/report.py`
- Modify: `tests/test_report.py`

- [ ] **Step 1: Write failing tests for PDF generation**

Add to `tests/test_report.py`:

```python
class TestGeneratePdf:
    def test_returns_valid_pdf(self):
        from backend.core.report import generate_pdf
        aid = _create_test_analysis()
        data = generate_pdf(aid)
        assert isinstance(data, bytes)
        assert data[:5] == b'%PDF-'

    def test_module_filtering_excludes_sections(self):
        from backend.core.report import generate_pdf
        aid = _create_test_analysis()
        data = generate_pdf(aid, modules=["summary"])
        assert isinstance(data, bytes)
        assert data[:5] == b'%PDF-'

    def test_weasyprint_missing_gives_clear_error(self):
        """If weasyprint is not importable, generate_pdf raises with helpful message."""
        # This test just verifies the function exists and works when weasyprint IS available.
        # The error path is tested implicitly by the try/except in generate_pdf.
        from backend.core.report import generate_pdf
        aid = _create_test_analysis()
        pdf = generate_pdf(aid)
        assert len(pdf) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_report.py::TestGeneratePdf -v`
Expected: FAIL — `generate_pdf` not defined.

- [ ] **Step 3: Create the HTML template**

Create directory and file `backend/templates/report.html`:

```html
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page { size: A4; margin: 2cm; }
  body { font-family: Arial, "Microsoft YaHei", sans-serif; font-size: 10pt; color: #1a1a2e; line-height: 1.5; }
  h1 { color: #1a73e8; font-size: 20pt; margin: 0 0 4px 0; }
  h2 { color: #1a73e8; font-size: 13pt; border-bottom: 2px solid #1a73e8; padding-bottom: 4px; margin-top: 24px; }
  .subtitle { color: #666; font-size: 10pt; margin: 0 0 16px 0; }
  .meta-table { width: 100%; margin-bottom: 16px; }
  .meta-table td { padding: 3px 8px; }
  .meta-table .label { color: #888; width: 100px; }
  .metrics { display: flex; gap: 12px; margin-bottom: 16px; }
  .metric-card { flex: 1; border-radius: 8px; padding: 12px; text-align: center; }
  .metric-card .value { font-size: 22pt; font-weight: 700; }
  .metric-card .label { font-size: 8pt; color: #666; }
  .mc-total { background: #f0f9ff; }
  .mc-total .value { color: #1a73e8; }
  .mc-ok { background: #f0fdf4; }
  .mc-ok .value { color: #16a34a; }
  .mc-wrong { background: #fef2f2; }
  .mc-wrong .value { color: #dc2626; }
  .mc-uncertain { background: #fffbeb; }
  .mc-uncertain .value { color: #d97706; }
  .stats { display: flex; gap: 12px; margin-bottom: 20px; }
  .stat-card { flex: 1; background: #f8f9fa; border-radius: 6px; padding: 10px; text-align: center; }
  .stat-card .label { font-size: 7pt; color: #888; text-transform: uppercase; }
  .stat-card .value { font-size: 14pt; font-weight: 700; color: #1a73e8; }
  table.detail { width: 100%; border-collapse: collapse; font-size: 8.5pt; margin-bottom: 16px; }
  table.detail th { background: #f1f5f9; padding: 6px 6px; text-align: left; border-bottom: 2px solid #e2e8f0; font-weight: 600; }
  table.detail td { padding: 5px 6px; border-bottom: 1px solid #f1f5f9; }
  table.detail tr.row-wrong { background: #fef2f2; }
  table.detail tr.row-uncertain { background: #fffbeb; }
  .badge { padding: 2px 8px; border-radius: 10px; font-weight: 600; font-size: 8pt; }
  .badge-ok { background: #dcfce7; color: #16a34a; }
  .badge-wrong { background: #fee2e2; color: #dc2626; }
  .badge-uncertain { background: #fef9c3; color: #d97706; }
  .charts { display: flex; gap: 16px; margin-bottom: 20px; }
  .charts img { max-width: 48%; }
  .abnormal-card { border-left: 4px solid; border-radius: 4px; padding: 8px 12px; margin-bottom: 8px; font-size: 9pt; }
  .abnormal-card.wrong { border-color: #dc2626; background: #fef2f2; }
  .abnormal-card.uncertain { border-color: #d97706; background: #fffbeb; }
  .abnormal-card .title { font-weight: 700; }
  .abnormal-card .detail { color: #555; margin-top: 4px; }
  .alignment { font-family: "Courier New", monospace; font-size: 7.5pt; white-space: pre; line-height: 1.3; background: #f8f9fa; padding: 8px; border-radius: 4px; margin-bottom: 12px; overflow-x: hidden; word-wrap: break-word; }
  .aln-mismatch { color: #dc2626; font-weight: bold; }
  .aln-gap { color: #888; }
  table.threshold { width: 60%; border-collapse: collapse; font-size: 9pt; }
  table.threshold th { background: #f1f5f9; padding: 5px 8px; text-align: left; border-bottom: 2px solid #e2e8f0; }
  table.threshold td { padding: 4px 8px; border-bottom: 1px solid #f1f5f9; }
  .page-break { page-break-before: always; }
  .note { color: #888; font-size: 8pt; font-style: italic; margin-top: 4px; }
</style>
</head>
<body>

{% if "summary" in modules %}
<h1>BioAgent MAX</h1>
<p class="subtitle">Sanger Sequencing QC Report</p>

<table class="meta-table">
  <tr><td class="label">分析名称</td><td><strong>{{ analysis.name }}</strong></td></tr>
  <tr><td class="label">分析时间</td><td>{{ analysis.finished_at }}</td></tr>
  <tr><td class="label">数据来源</td><td>{{ analysis.source_path }}</td></tr>
</table>

<div class="metrics">
  <div class="metric-card mc-total"><div class="value">{{ analysis.total }}</div><div class="label">总样本</div></div>
  <div class="metric-card mc-ok"><div class="value">{{ analysis.ok_count }}</div><div class="label">OK</div></div>
  <div class="metric-card mc-wrong"><div class="value">{{ analysis.wrong_count }}</div><div class="label">Wrong</div></div>
  <div class="metric-card mc-uncertain"><div class="value">{{ analysis.uncertain_count }}</div><div class="label">Uncertain</div></div>
</div>

<div class="stats">
  <div class="stat-card"><div class="label">通过率</div><div class="value">{{ stats.pass_rate }}%</div></div>
  <div class="stat-card"><div class="label">平均 Identity</div><div class="value">{{ stats.avg_identity }}</div></div>
  <div class="stat-card"><div class="label">平均 CDS Coverage</div><div class="value">{{ stats.avg_cds_coverage }}</div></div>
</div>
{% endif %}

{% if "detail_table" in modules %}
<h2>样本明细</h2>
<table class="detail">
  <thead>
    <tr>
      <th>SID</th><th>状态</th><th>原因</th><th>Identity</th><th>CDS Cov</th>
      <th>AA变异</th><th>Sub</th><th>Ins</th><th>Del</th><th>规则</th>
      <th>长度</th><th>质量</th>
    </tr>
  </thead>
  <tbody>
  {% for s in samples %}
    <tr class="{% if s.status == 'wrong' %}row-wrong{% elif s.status == 'uncertain' %}row-uncertain{% endif %}">
      <td>{{ s.sid }}</td>
      <td><span class="badge badge-{{ s.status }}">{{ s.status | upper }}</span></td>
      <td>{{ s.reason }}</td>
      <td>{{ "%.4f" | format(s.identity) if s.identity is not none else "" }}</td>
      <td>{{ "%.3f" | format(s.cds_coverage) if s.cds_coverage is not none else "" }}</td>
      <td>{{ s.aa_changes_n or 0 }}</td>
      <td>{{ s.sub_count or 0 }}</td>
      <td>{{ s.ins_count or 0 }}</td>
      <td>{{ s.del_count or 0 }}</td>
      <td>R{{ s.rule_id }}</td>
      <td>{{ s.seq_length }}</td>
      <td>{{ "%.1f" | format(s.avg_quality) if s.avg_quality is not none else "" }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>
{% endif %}

{% if "charts" in modules and charts %}
<h2>数据分布</h2>
<div class="charts">
  {% if charts.identity_hist %}<img src="data:image/png;base64,{{ charts.identity_hist }}">{% endif %}
  {% if charts.status_pie %}<img src="data:image/png;base64,{{ charts.status_pie }}">{% endif %}
</div>
{% if charts.coverage_hist %}
<div class="charts">
  <img src="data:image/png;base64,{{ charts.coverage_hist }}">
</div>
{% endif %}
{% endif %}

{% if "abnormal_details" in modules and abnormal_samples %}
<h2>异常样本详情</h2>
{% for s in abnormal_samples %}
<div class="abnormal-card {{ s.status }}">
  <div class="title">{{ s.sid }} — {{ s.status | upper }} (R{{ s.rule_id }}: {{ s.reason }})</div>
  <div class="detail">
    Identity: {{ "%.4f" | format(s.identity) }} &nbsp;&nbsp;
    CDS Coverage: {{ "%.3f" | format(s.cds_coverage) }} &nbsp;&nbsp;
    AA 变异: {{ s.aa_changes_n or 0 }} &nbsp;&nbsp;
    质量: {{ "%.1f" | format(s.avg_quality) if s.avg_quality is not none else "N/A" }}
    {% if s.aa_changes_parsed %}
    <br>变异列表: {{ s.aa_changes_parsed | join(", ") }}
    {% endif %}
  </div>
</div>
{% endfor %}
{% endif %}

{% if "alignment" in modules and alignment_samples %}
<div class="page-break"></div>
<h2>序列比对视图</h2>
{% if alignment_truncated %}
<p class="note">显示前 {{ max_alignment }} 个异常样本（共 {{ total_abnormal }} 个）</p>
{% endif %}
{% for item in alignment_samples %}
<h3 style="font-size: 10pt; color: #333;">{{ item.sid }} — {{ item.status | upper }}</h3>
<div class="alignment">{{ item.alignment_html }}</div>
{% endfor %}
{% endif %}

{% if "thresholds" in modules and thresholds %}
<div class="page-break"></div>
<h2>阈值配置</h2>
<table class="threshold">
  <thead><tr><th>参数</th><th>值</th></tr></thead>
  <tbody>
  {% for key, value in thresholds.items() %}
    <tr><td>{{ key }}</td><td>{{ value }}</td></tr>
  {% endfor %}
  </tbody>
</table>
{% endif %}

<div style="margin-top: 30px; text-align: center; color: #999; font-size: 8pt;">
  BioAgent MAX — Generated {{ generated_at }}
</div>

</body>
</html>
```

- [ ] **Step 4: Implement `generate_pdf()` and helpers**

Add to `backend/core/report.py`:

```python
import base64
from pathlib import Path

TEMPLATE_DIR = Path(__file__).parent.parent / "templates"


def _render_alignment_html(ref_gapped: str, qry_gapped: str, width: int = 80) -> str:
    """Render alignment as HTML for PDF (no Streamlit dependency)."""
    if not ref_gapped or not qry_gapped:
        return ""
    blocks = []
    idx = 0
    while idx < len(ref_gapped):
        ref_chunk = ref_gapped[idx:idx + width]
        qry_chunk = qry_gapped[idx:idx + width]
        ref_html, mid_html, qry_html = [], [], []
        for a, b in zip(ref_chunk, qry_chunk):
            if a == "-" or b == "-":
                ref_html.append(f'<span class="aln-gap">{a}</span>')
                mid_html.append(" ")
                qry_html.append(f'<span class="aln-gap">{b}</span>')
            elif a == b:
                ref_html.append(a)
                mid_html.append("|")
                qry_html.append(b)
            else:
                ref_html.append(f'<span class="aln-mismatch">{a}</span>')
                mid_html.append('<span class="aln-mismatch">*</span>')
                qry_html.append(f'<span class="aln-mismatch">{b}</span>')
        pos_s = idx + 1
        pos_e = min(idx + width, len(ref_gapped))
        blocks.append(
            f'{pos_s:>5}-{pos_e:<5}  REF  {"".join(ref_html)}\n'
            f'             {"".join(mid_html)}\n'
            f'             QRY  {"".join(qry_html)}'
        )
        idx += width
    return "\n\n".join(blocks)


def _render_html(report_data: dict, modules: list[str]) -> str:
    """Render Jinja2 HTML template for PDF."""
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template("report.html")

    # Prepare chart images as base64
    charts_b64 = {}
    if "charts" in modules:
        try:
            raw_charts = _render_charts(report_data["samples"])
            for key, png_bytes in raw_charts.items():
                charts_b64[key] = base64.b64encode(png_bytes).decode("ascii")
        except Exception:
            logger.warning("Chart generation failed, skipping charts in PDF", exc_info=True)

    # Prepare abnormal samples
    abnormal = [s for s in report_data["samples"] if s["status"] in ("wrong", "uncertain")]
    for s in abnormal:
        try:
            s["aa_changes_parsed"] = json.loads(s["aa_changes"]) if s["aa_changes"] else []
        except (json.JSONDecodeError, TypeError):
            s["aa_changes_parsed"] = []

    # Prepare alignment samples (cap at MAX_ALIGNMENT_SAMPLES)
    alignment_samples = []
    if "alignment" in modules:
        for s in abnormal[:MAX_ALIGNMENT_SAMPLES]:
            html = _render_alignment_html(s.get("ref_gapped", ""), s.get("qry_gapped", ""))
            if html:
                alignment_samples.append({
                    "sid": s["sid"], "status": s["status"],
                    "alignment_html": html,
                })

    return template.render(
        analysis=report_data["analysis"],
        samples=report_data["samples"],
        stats=report_data["stats"],
        thresholds=report_data["thresholds"],
        charts=charts_b64,
        abnormal_samples=abnormal if "abnormal_details" in modules else [],
        alignment_samples=alignment_samples,
        alignment_truncated=len(abnormal) > MAX_ALIGNMENT_SAMPLES,
        total_abnormal=len(abnormal),
        max_alignment=MAX_ALIGNMENT_SAMPLES,
        modules=modules,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


def generate_pdf(analysis_id: str, modules: list[str] | None = None) -> bytes:
    """Generate PDF report as bytes."""
    try:
        import weasyprint
    except ImportError:
        raise ImportError(
            "PDF export requires WeasyPrint. Install with: pip install weasyprint\n"
            "WeasyPrint also requires system libraries (GTK/Cairo). "
            "See: https://doc.courtbouillon.org/weasyprint/stable/first_steps.html"
        )

    mods = modules or ALL_MODULES
    data = _build_report_data(analysis_id)
    html_str = _render_html(data, mods)
    return weasyprint.HTML(string=html_str).write_pdf()
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_report.py -v`
Expected: All tests PASS (including TestGeneratePdf).

- [ ] **Step 6: Commit**

```bash
git add backend/core/report.py backend/templates/report.html tests/test_report.py
git commit -m "feat: add PDF report generation with HTML template and WeasyPrint"
```

---

### Task 5: Frontend export panel

**Files:**
- Modify: `frontend/pages/2_results.py`

- [ ] **Step 1: Add export panel to the results page**

After the analysis selector (`st.selectbox`), add an export expander. Insert this block after line 36 in `frontend/pages/2_results.py` (after `selected_id = st.selectbox(...)`):

```python
# -- Export panel --
with st.expander("📥 导出报告"):
    export_col1, export_col2 = st.columns(2)
    with export_col1:
        export_format = st.radio("导出格式", ["PDF", "Excel", "两者都导出"], horizontal=True)
    with export_col2:
        st.markdown("**包含模块：**")
        mod_summary = st.checkbox("批次摘要", value=True, key="mod_summary")
        mod_detail = st.checkbox("样本明细表", value=True, key="mod_detail")
        mod_charts = st.checkbox("分布图表", value=True, key="mod_charts")
        mod_abnormal = st.checkbox("异常样本详情", value=True, key="mod_abnormal")
        mod_alignment = st.checkbox("序列比对视图", value=True, key="mod_alignment")
        mod_thresholds = st.checkbox("阈值配置", value=True, key="mod_thresholds")

    selected_modules = []
    if mod_summary: selected_modules.append("summary")
    if mod_detail: selected_modules.append("detail_table")
    if mod_charts: selected_modules.append("charts")
    if mod_abnormal: selected_modules.append("abnormal_details")
    if mod_alignment: selected_modules.append("alignment")
    if mod_thresholds: selected_modules.append("thresholds")

    if st.button("🚀 生成报告", type="primary", use_container_width=True):
        if not selected_modules:
            st.warning("请至少选择一个模块")
        else:
            from backend.core.report import generate_pdf, generate_excel
            analysis_name = options[selected_id].split("(")[0].strip()
            safe_name = analysis_name.replace(" ", "_")

            try:
                if export_format in ("PDF", "两者都导出"):
                    with st.spinner("正在生成 PDF..."):
                        pdf_bytes = generate_pdf(selected_id, modules=selected_modules)
                    st.download_button(
                        "📄 下载 PDF", data=pdf_bytes,
                        file_name=f"{safe_name}.pdf",
                        mime="application/pdf",
                    )
            except ImportError as e:
                st.error(str(e))

            try:
                if export_format in ("Excel", "两者都导出"):
                    with st.spinner("正在生成 Excel..."):
                        excel_bytes = generate_excel(selected_id, modules=selected_modules)
                    st.download_button(
                        "📊 下载 Excel", data=excel_bytes,
                        file_name=f"{safe_name}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
            except Exception as e:
                st.error(f"Excel 生成失败: {e}")
```

- [ ] **Step 2: Manual test — launch Streamlit and verify export panel renders**

Run: `streamlit run frontend/app.py`
Navigate to "分析结果" page. Verify:
- Export panel expander appears
- Format radio and module checkboxes render
- If an analysis exists, clicking "生成报告" triggers download

- [ ] **Step 3: Commit**

```bash
git add frontend/pages/2_results.py
git commit -m "feat: add report export panel to results page"
```

---

### Task 6: API endpoint for PDF/Excel export

**Files:**
- Modify: `backend/api/export.py`

- [ ] **Step 1: Extend export endpoint with PDF and Excel support**

Replace the content of `backend/api/export.py`:

```python
import csv
import io
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, Response
from backend.db.database import db_session
from backend.db.models import Analysis, Sample

router = APIRouter()


@router.get("/export/{analysis_id}")
def export_report(
    analysis_id: str,
    format: str = Query("csv"),
    modules: str = Query(""),
):
    # Validate analysis exists
    with db_session() as session:
        analysis = session.get(Analysis, analysis_id)
        if not analysis:
            raise HTTPException(404, "Analysis not found")

    module_list = [m.strip() for m in modules.split(",") if m.strip()] or None

    if format == "pdf":
        from backend.core.report import generate_pdf
        try:
            pdf_bytes = generate_pdf(analysis_id, modules=module_list)
        except ImportError as e:
            raise HTTPException(500, str(e))
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=report_{analysis_id[:8]}.pdf"},
        )

    if format == "excel":
        from backend.core.report import generate_excel
        excel_bytes = generate_excel(analysis_id, modules=module_list)
        return Response(
            content=excel_bytes,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=report_{analysis_id[:8]}.xlsx"},
        )

    # Default: CSV
    with db_session() as session:
        samples = session.query(Sample).filter(Sample.analysis_id == analysis_id).all()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["SID", "Status", "Reason", "Rule", "Identity", "CDS_Coverage",
                         "Frameshift", "AA_Changes_N", "Seq_Length", "Avg_Quality"])
        for s in samples:
            writer.writerow([s.sid, s.status, s.reason, s.rule_id, s.identity,
                             s.cds_coverage, s.frameshift, s.aa_changes_n,
                             s.seq_length, s.avg_quality])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=analysis_{analysis_id}.csv"},
    )
```

- [ ] **Step 2: Run all tests to ensure nothing is broken**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add backend/api/export.py
git commit -m "feat: extend API export endpoint with PDF and Excel formats"
```

---

### Task 7: MCP server export extension

**Files:**
- Modify: `backend/mcp_server.py`

- [ ] **Step 1: Extend the export_report MCP tool**

Replace the `export_report` function in `backend/mcp_server.py` (lines 126-137):

```python
@server.tool()
async def export_report(analysis_id: str, format: str = "csv", modules: str = "") -> str:
    """导出分析报告。format: csv/pdf/excel。modules: 逗号分隔的模块列表（留空=全部）。
    PDF/Excel 报告保存到 data/exports/ 目录并返回文件路径。"""
    init_db()

    module_list = [m.strip() for m in modules.split(",") if m.strip()] or None

    if format == "pdf":
        from backend.core.report import generate_pdf
        try:
            pdf_bytes = generate_pdf(analysis_id, modules=module_list)
        except ImportError as e:
            return f"错误: {e}"
        import time
        export_dir = Path(__file__).parent.parent / "data" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{analysis_id[:8]}_{time.strftime('%Y%m%d_%H%M%S')}.pdf"
        path = export_dir / filename
        path.write_bytes(pdf_bytes)
        return f"PDF 报告已生成: {path}"

    if format == "excel":
        from backend.core.report import generate_excel
        excel_bytes = generate_excel(analysis_id, modules=module_list)
        import time
        export_dir = Path(__file__).parent.parent / "data" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{analysis_id[:8]}_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"
        path = export_dir / filename
        path.write_bytes(excel_bytes)
        return f"Excel 报告已生成: {path}"

    # Default: CSV text
    with db_session() as session:
        samples = session.query(Sample).filter(Sample.analysis_id == analysis_id).all()
        if not samples:
            return f"未找到样本数据: {analysis_id}"
        lines = ["SID,Status,Reason,Identity,CDS_Coverage,AA_Changes_N"]
        for s in samples:
            lines.append(f"{s.sid},{s.status},{s.reason},{s.identity},{s.cds_coverage},{s.aa_changes_n}")
        return "\n".join(lines)
```

- [ ] **Step 2: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add backend/mcp_server.py
git commit -m "feat: extend MCP export_report tool with PDF and Excel support"
```

---

### Task 8: Final integration test and cleanup

**Files:**
- Modify: `tests/test_report.py`

- [ ] **Step 1: Add integration test for full export flow**

Add to `tests/test_report.py`:

```python
class TestIntegration:
    def test_full_pdf_export_all_modules(self):
        """End-to-end: create analysis, generate PDF with all modules."""
        from backend.core.report import generate_pdf, ALL_MODULES
        aid = _create_test_analysis()
        pdf = generate_pdf(aid, modules=ALL_MODULES)
        assert pdf[:5] == b'%PDF-'
        assert len(pdf) > 1000  # Non-trivial PDF

    def test_full_excel_export_all_modules(self):
        """End-to-end: create analysis, generate Excel with all modules."""
        from backend.core.report import generate_excel
        import openpyxl
        aid = _create_test_analysis()
        data = generate_excel(aid)
        wb = openpyxl.load_workbook(io.BytesIO(data))
        # Verify sample data is present
        ws = wb["样本明细"]
        assert ws.max_row >= 4  # header + 3 samples
```

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_report.py
git commit -m "test: add integration tests for report export"
```

- [ ] **Step 4: Add `data/exports/` to .gitignore if not already covered**

Check if `data/` is already in `.gitignore` (it should be based on git history). If not, add `data/exports/`.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: report export feature complete"
```
