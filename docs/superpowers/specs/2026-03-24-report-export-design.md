# Report Export Feature — Design Spec

**Date:** 2026-03-24
**Status:** Draft
**Branch:** feature/web-platform

## Goal

Add professional PDF and Excel report export to BioAgent MAX, allowing users to generate downloadable reports from any analysis record. Reports should be comprehensive by default with optional module selection.

## Context

Currently the platform only supports CSV export via the MCP server (`export_report` tool), which outputs a flat text table with no formatting. Users need to share analysis results with lab colleagues (PI, collaborators) who understand sequencing but want a polished, readable summary rather than raw data.

## Target Audience

Internal lab teams — PI/supervisors, colleagues. They understand Sanger sequencing terminology and need technical detail, but presented in a clear, scannable format.

## Requirements

### Report Formats

- **PDF** — Fixed-layout document for sharing and archival. Contains text, tables, charts, and alignment views.
- **Excel** — Multi-sheet workbook for data exploration and secondary analysis. Contains structured data only (no charts or alignment views).

### Report Modules (6 total, all enabled by default)

Users can toggle each module on/off before export:

| # | Module | PDF | Excel | Description |
|---|--------|-----|-------|-------------|
| 1 | Batch Summary | Page 1 | "摘要" sheet | Analysis name, time, source, sample counts, pass rate, avg identity, avg CDS coverage |
| 2 | Sample Detail Table | Page 2 | "样本明细" sheet | Full sample table: SID, Status, Reason, Identity, CDS Coverage, AA Changes, Sub/Ins/Del, Rule ID, Seq Length, Avg Quality |
| 3 | Distribution Charts | Page 3 top | — | Identity histogram, CDS coverage histogram, status pie chart (Plotly → PNG via kaleido) |
| 4 | Abnormal Sample Details | Page 3 bottom + extra pages | — | Wrong/Uncertain samples with key metrics, reason, AA change list. Grouped by severity. |
| 5 | Alignment View | Additional pages | — | Gapped ref vs query alignment for Wrong/Uncertain samples only. Monospace text rendering. |
| 6 | Threshold Config | Final page | "阈值配置" sheet | Snapshot of the 15 threshold parameters used for this analysis |

### PDF Layout

**Page 1 — Cover + Batch Summary:**
- BioAgent MAX header with branding
- Analysis metadata (name, time range, source path)
- Four metric cards: Total, OK (green), Wrong (red), Uncertain (amber)
- Three stat cards: Pass rate, Avg Identity, Avg CDS Coverage

**Page 2 — Sample Detail Table:**
- Section header "样本明细"
- Full table with all samples
- Status column uses colored badges (OK=green, Wrong=red, Uncertain=amber)
- Wrong/Uncertain rows have tinted background for quick scanning
- Table paginates automatically across pages if needed

**Page 3 — Charts + Abnormal Details:**
- Top half: Identity distribution histogram + Status pie chart (side by side, rendered as PNG images from Plotly)
- Bottom half: Abnormal sample detail cards with colored left border (red for Wrong, amber for Uncertain), showing SID, status, reason, key metrics

**Additional pages — Alignment View (per abnormal sample):**
- Section header with SID and status
- Monospace rendered ref_gapped vs qry_gapped with position rulers
- Mismatches highlighted

**Final page — Threshold Config:**
- Table of all 15 threshold parameters with current values

### Excel Layout

| Sheet Name | Content |
|------------|---------|
| 摘要 | Analysis name, time, source, total/ok/wrong/uncertain counts, pass rate, avg identity, avg CDS coverage |
| 样本明细 | One row per sample, all columns from the detail table. Auto-filter enabled. Column widths auto-fitted. Status column uses conditional formatting (green/red/amber fill). |
| 阈值配置 | Two columns: Parameter Name, Value. Lists all 15 thresholds. |

## Architecture

### New Files

**`backend/core/report.py`** — Report generation module:
- `generate_pdf(analysis_id: str, modules: list[str] | None = None) -> bytes`
- `generate_excel(analysis_id: str, modules: list[str] | None = None) -> bytes`
- `_build_report_data(analysis_id: str) -> dict` — Fetch analysis + samples from DB, structure for templates
- `_render_charts(samples_data: list[dict]) -> dict[str, bytes]` — Generate Plotly charts as PNG bytes
- `_render_html(report_data: dict, modules: list[str]) -> str` — Build HTML string for PDF rendering
- `_render_alignment_html(ref_gapped: str, qry_gapped: str) -> str` — Alignment view HTML fragment

Module identifiers: `"summary"`, `"detail_table"`, `"charts"`, `"abnormal_details"`, `"alignment"`, `"thresholds"`

**`backend/templates/report.html`** — Jinja2 HTML template for PDF. Sections wrapped in `{% if "module_name" in modules %}` blocks. Inline CSS for WeasyPrint compatibility (no external stylesheets).

### Modified Files

**`frontend/pages/2_results.py`:**
- Add "导出报告" button next to analysis selector
- Expandable panel with: format radio (PDF / Excel / Both), 6 module checkboxes (all checked by default), "生成报告" button
- On click: call `generate_pdf()` / `generate_excel()` directly (Streamlit direct import), use `st.download_button` for file download

**`backend/api/export.py`:**
- Extend `GET /api/export/{id}` to accept `format` query param: `csv` (existing), `pdf`, `excel`
- Accept `modules` query param: comma-separated module list
- Return file with appropriate Content-Type and Content-Disposition headers

**`backend/mcp_server.py`:**
- Extend `export_report` tool: add `format` param (`"csv"`, `"pdf"`, `"excel"`), add `modules` param (comma-separated string)
- For PDF/Excel: generate file, save to `data/exports/{analysis_id}_{timestamp}.{ext}`, return file path

**`backend/requirements.txt`:**
- Add: `weasyprint>=60.0`, `openpyxl>=3.1`, `kaleido>=0.2.1`, `jinja2>=3.1`

### Data Flow

```
User clicks "导出报告"
  → frontend calls generate_pdf() / generate_excel()
    → _build_report_data() reads DB (analysis + samples + config_snapshot)
    → For PDF:
        → _render_charts() generates Plotly PNGs via kaleido
        → _render_html() fills Jinja2 template with data + base64-encoded chart images
        → weasyprint.HTML(string=html).write_pdf() → bytes
    → For Excel:
        → pandas DataFrames written to openpyxl Workbook with formatting
        → workbook saved to BytesIO → bytes
  → st.download_button serves the file
```

### Chart Generation

Use existing chart functions from `frontend/components/charts.py` as reference, but create standalone Plotly figures in `report.py` (no Streamlit dependency). Export to PNG using `fig.to_image(format="png", engine="kaleido")`.

Charts needed:
- Identity distribution histogram (from `identity_distribution()` logic)
- CDS coverage distribution histogram (from `coverage_distribution()` logic)
- Status pie chart (from `status_pie()` logic)

## Constraints

- WeasyPrint requires system-level dependencies (GTK/Cairo) on some platforms. Document installation in usage_guide.md.
- PDF generation is synchronous — acceptable for typical batch sizes (<100 samples).
- Alignment views can be long — limit to Wrong/Uncertain samples only, and cap at configurable max (default 20 samples) to keep PDF size reasonable.
- Excel files do not include charts or alignment views to keep the format focused on data manipulation.

## Testing

- Unit test: `generate_pdf()` returns valid PDF bytes (check magic bytes `%PDF`)
- Unit test: `generate_excel()` returns valid xlsx (check with openpyxl.load_workbook)
- Unit test: module filtering — pass subset of modules, verify excluded sections are absent
- Integration test: full export from a test analysis record via API endpoint
