"""Tests for backend.core.report — report data layer and Excel generation."""
import json
import os
import pytest
from io import BytesIO

# Force in-memory SQLite for tests
os.environ["DATABASE_URL"] = "sqlite://"

from backend.db.database import init_db, db_session, get_engine, Base
from backend.db.models import Analysis, Sample, new_id


def _reset_db():
    """Drop and recreate all tables for a clean slate."""
    from backend.db import database
    # Reset module-level singletons so init_db creates fresh tables
    Base.metadata.drop_all(get_engine())
    Base.metadata.create_all(get_engine())


def _create_test_analysis() -> str:
    """Create an Analysis with 3 samples (2 ok, 1 wrong) in the DB. Returns analysis_id."""
    _reset_db()
    analysis_id = new_id()
    thresholds = {
        "identity_high": 0.99,
        "identity_low": 0.95,
        "cds_coverage_min": 0.9,
        "max_aa_changes": 3,
    }
    with db_session() as session:
        session.add(Analysis(
            id=analysis_id,
            name="Test Report Analysis",
            source_type="upload",
            source_path="/tmp/test",
            status="done",
            total=3,
            ok_count=2,
            wrong_count=1,
            uncertain_count=0,
            config_snapshot=json.dumps(thresholds),
            finished_at=None,
        ))
        # Sample 1: ok
        session.add(Sample(
            id=new_id(), analysis_id=analysis_id,
            sid="SAMPLE-001", clone="CLN-A", status="ok", reason="",
            rule_id=10, identity=0.998, cds_coverage=1.0,
            frameshift=False, aa_changes="[]", aa_changes_n=0,
            raw_aa_changes_n=0, orientation="FORWARD",
            seq_length=800, ref_length=900, avg_quality=38.5,
            sub_count=1, ins_count=0, del_count=0,
            ref_gapped="ATCG", qry_gapped="ATCG",
            quality_scores="[30,35,40]", raw_data="{}",
        ))
        # Sample 2: ok
        session.add(Sample(
            id=new_id(), analysis_id=analysis_id,
            sid="SAMPLE-002", clone="CLN-B", status="ok", reason="",
            rule_id=10, identity=0.995, cds_coverage=0.98,
            frameshift=False, aa_changes="[]", aa_changes_n=0,
            raw_aa_changes_n=0, orientation="FORWARD",
            seq_length=850, ref_length=900, avg_quality=36.0,
            sub_count=2, ins_count=0, del_count=0,
            ref_gapped="ATCG", qry_gapped="ATCG",
            quality_scores="[28,33,38]", raw_data="{}",
        ))
        # Sample 3: wrong
        session.add(Sample(
            id=new_id(), analysis_id=analysis_id,
            sid="SAMPLE-003", clone="CLN-C", status="wrong", reason="Identity too low",
            rule_id=1, identity=0.85, cds_coverage=0.70,
            frameshift=True, aa_changes='["A100T","G200D"]', aa_changes_n=2,
            raw_aa_changes_n=2, orientation="REVERSE",
            seq_length=600, ref_length=900, avg_quality=25.0,
            sub_count=10, ins_count=3, del_count=2,
            ref_gapped="ATCG--NN", qry_gapped="ATCG--NN",
            quality_scores="[20,22,25]", raw_data="{}",
        ))
    return analysis_id


# ── _build_report_data tests ──────────────────────────────────────────

class TestBuildReportData:

    def test_returns_analysis_and_samples(self):
        analysis_id = _create_test_analysis()
        from backend.core.report import _build_report_data

        data = _build_report_data(analysis_id)

        assert "analysis" in data
        assert "samples" in data
        assert "thresholds" in data
        assert "stats" in data

        assert data["analysis"]["name"] == "Test Report Analysis"
        assert data["analysis"]["total"] == 3
        assert data["analysis"]["ok_count"] == 2
        assert data["analysis"]["wrong_count"] == 1

        assert len(data["samples"]) == 3

        # Stats should include computed fields
        assert "pass_rate" in data["stats"]
        assert "avg_identity" in data["stats"]
        assert "avg_coverage" in data["stats"]

    def test_raises_for_nonexistent_id(self):
        _reset_db()
        from backend.core.report import _build_report_data

        with pytest.raises(ValueError, match="not found"):
            _build_report_data("nonexistent-id-12345")


# ── generate_excel tests ─────────────────────────────────────────────

class TestGenerateExcel:

    def test_returns_valid_xlsx_with_3_sheets(self):
        analysis_id = _create_test_analysis()
        from backend.core.report import generate_excel
        import openpyxl

        xlsx_bytes = generate_excel(analysis_id)

        assert isinstance(xlsx_bytes, bytes)
        assert len(xlsx_bytes) > 0

        wb = openpyxl.load_workbook(BytesIO(xlsx_bytes))
        sheet_names = wb.sheetnames
        assert "摘要" in sheet_names
        assert "样本明细" in sheet_names
        assert "阈值配置" in sheet_names
        assert len(sheet_names) == 3

    def test_module_filtering_detail_only(self):
        analysis_id = _create_test_analysis()
        from backend.core.report import generate_excel
        import openpyxl

        xlsx_bytes = generate_excel(analysis_id, modules=["detail_table"])

        wb = openpyxl.load_workbook(BytesIO(xlsx_bytes))
        assert wb.sheetnames == ["样本明细"]

    def test_module_filtering_summary_and_thresholds(self):
        analysis_id = _create_test_analysis()
        from backend.core.report import generate_excel
        import openpyxl

        xlsx_bytes = generate_excel(analysis_id, modules=["summary", "thresholds"])

        wb = openpyxl.load_workbook(BytesIO(xlsx_bytes))
        assert "摘要" in wb.sheetnames
        assert "阈值配置" in wb.sheetnames
        assert "样本明细" not in wb.sheetnames

    def test_detail_sheet_has_all_samples(self):
        analysis_id = _create_test_analysis()
        from backend.core.report import generate_excel
        import openpyxl

        xlsx_bytes = generate_excel(analysis_id, modules=["detail_table"])

        wb = openpyxl.load_workbook(BytesIO(xlsx_bytes))
        ws = wb["样本明细"]
        # Header row + 3 data rows
        assert ws.max_row == 4
