from backend.db.database import init_db, db_session
from backend.db.models import save_analysis_with_samples, Analysis, Sample

def test_save_analysis_with_samples():
    init_db()
    samples = [{
        "sid": "TEST-1", "clone": "TEST", "identity": 0.99,
        "cds_coverage": 1.0, "frameshift": False,
        "aa_changes": [], "aa_changes_n": 0, "raw_aa_changes_n": 0,
        "orientation": "FORWARD", "seq_length": 1000, "ref_length": 5000,
        "avg_qry_quality": 35.0, "sub": 0, "ins": 0, "del": 0,
        "ref_gapped": "ATCG", "qry_gapped": "ATCG",
        "quality_scores": [30, 35, 40, 45],
    }]
    judgments = [{"sid": "TEST-1", "status": "ok", "reason": "", "rule": 10}]
    thresholds = {"identity_high": 0.95}

    analysis_id = save_analysis_with_samples(
        samples=samples, judgments=judgments,
        thresholds=thresholds, name="Test Analysis",
        source_type="test", source_path="/tmp/test",
    )
    assert analysis_id is not None

    with db_session() as session:
        a = session.get(Analysis, analysis_id)
        assert a.total == 1
        assert a.ok_count == 1
        s = session.query(Sample).filter(Sample.analysis_id == analysis_id).first()
        assert s.sid == "TEST-1"
        assert s.status == "ok"
