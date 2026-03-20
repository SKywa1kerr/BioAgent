import json
from fastapi import APIRouter, HTTPException, Query
from backend.db.database import get_session_factory
from backend.db.models import Analysis, Sample

router = APIRouter()

@router.get("/results")
def list_analyses(limit: int = Query(20, le=100), offset: int = Query(0)):
    Session = get_session_factory()
    session = Session()
    analyses = session.query(Analysis).order_by(Analysis.created_at.desc()).offset(offset).limit(limit).all()
    total = session.query(Analysis).count()
    session.close()
    return {"total": total, "items": [{"id": a.id, "name": a.name, "status": a.status, "total": a.total, "ok_count": a.ok_count, "wrong_count": a.wrong_count, "uncertain_count": a.uncertain_count, "created_at": str(a.created_at)} for a in analyses]}

@router.get("/results/{analysis_id}")
def get_analysis_detail(analysis_id: str):
    Session = get_session_factory()
    session = Session()
    analysis = session.get(Analysis, analysis_id)
    if not analysis:
        session.close()
        raise HTTPException(404, "Analysis not found")
    samples = session.query(Sample).filter(Sample.analysis_id == analysis_id).all()
    session.close()
    return {"analysis": {"id": analysis.id, "name": analysis.name, "status": analysis.status, "total": analysis.total, "ok_count": analysis.ok_count, "wrong_count": analysis.wrong_count, "config_snapshot": json.loads(analysis.config_snapshot) if analysis.config_snapshot else None}, "samples": [{"sid": s.sid, "status": s.status, "reason": s.reason, "rule_id": s.rule_id, "identity": s.identity, "cds_coverage": s.cds_coverage, "frameshift": s.frameshift, "aa_changes_n": s.aa_changes_n, "orientation": s.orientation, "seq_length": s.seq_length, "avg_quality": s.avg_quality} for s in samples]}

@router.get("/results/{analysis_id}/samples/{sid}")
def get_sample_detail(analysis_id: str, sid: str):
    Session = get_session_factory()
    session = Session()
    sample = session.query(Sample).filter(Sample.analysis_id == analysis_id, Sample.sid == sid).first()
    session.close()
    if not sample:
        raise HTTPException(404, "Sample not found")
    return {"sid": sample.sid, "status": sample.status, "reason": sample.reason, "rule_id": sample.rule_id, "identity": sample.identity, "cds_coverage": sample.cds_coverage, "frameshift": sample.frameshift, "aa_changes": json.loads(sample.aa_changes) if sample.aa_changes else [], "aa_changes_n": sample.aa_changes_n, "raw_aa_changes_n": sample.raw_aa_changes_n, "orientation": sample.orientation, "seq_length": sample.seq_length, "ref_length": sample.ref_length, "avg_quality": sample.avg_quality, "sub_count": sample.sub_count, "ins_count": sample.ins_count, "del_count": sample.del_count, "ref_gapped": sample.ref_gapped, "qry_gapped": sample.qry_gapped, "quality_scores": json.loads(sample.quality_scores) if sample.quality_scores else []}
