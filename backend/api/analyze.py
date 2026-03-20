import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from backend.core.alignment import analyze_dataset
from backend.core.rules import judge_batch, load_thresholds
from backend.db.database import get_session_factory
from backend.db.models import Analysis, Sample, new_id

logger = logging.getLogger(__name__)
router = APIRouter()

class AnalyzeRequest(BaseModel):
    gb_dir: str
    ab1_dir: str
    name: str | None = None

def _run_analysis(analysis_id: str, gb_dir: str, ab1_dir: str):
    Session = get_session_factory()
    session = Session()
    try:
        analysis = session.get(Analysis, analysis_id)
        analysis.status = "running"
        session.commit()
        samples = analyze_dataset(Path(gb_dir), Path(ab1_dir))
        thresholds = load_thresholds()
        judgments = judge_batch(samples, thresholds)
        ok = wrong = uncertain = 0
        for sd, jd in zip(samples, judgments):
            st = jd["status"]
            if st == "ok": ok += 1
            elif st == "wrong": wrong += 1
            else: uncertain += 1
            session.add(Sample(
                id=new_id(), analysis_id=analysis_id,
                sid=sd["sid"], clone=sd.get("clone",""),
                status=st, reason=jd.get("reason",""), rule_id=jd.get("rule"),
                identity=sd["identity"], cds_coverage=sd["cds_coverage"],
                frameshift=sd["frameshift"],
                aa_changes=json.dumps(sd.get("aa_changes",[])),
                aa_changes_n=sd.get("aa_changes_n",0),
                raw_aa_changes_n=sd.get("raw_aa_changes_n",0),
                orientation=sd.get("orientation",""),
                seq_length=sd.get("seq_length",0), ref_length=sd.get("ref_length",0),
                avg_quality=sd.get("avg_qry_quality"),
                sub_count=sd.get("sub",0), ins_count=sd.get("ins",0), del_count=sd.get("del",0),
                ref_gapped=sd.get("ref_gapped",""), qry_gapped=sd.get("qry_gapped",""),
                quality_scores=json.dumps(sd.get("quality_scores",[]) or []),
                raw_data=json.dumps(sd, default=str),
            ))
        analysis.status = "done"
        analysis.total = len(samples)
        analysis.ok_count = ok
        analysis.wrong_count = wrong
        analysis.uncertain_count = uncertain
        analysis.config_snapshot = json.dumps(thresholds)
        analysis.finished_at = datetime.now(timezone.utc)
        session.commit()
    except Exception as e:
        logger.error(f"Analysis {analysis_id} failed: {e}")
        analysis = session.get(Analysis, analysis_id)
        if analysis:
            analysis.status = "error"
            session.commit()
    finally:
        session.close()

@router.post("/analyze")
def trigger_analysis(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    if not Path(req.gb_dir).exists():
        raise HTTPException(404, f"GB directory not found: {req.gb_dir}")
    if not Path(req.ab1_dir).exists():
        raise HTTPException(404, f"AB1 directory not found: {req.ab1_dir}")
    Session = get_session_factory()
    session = Session()
    analysis = Analysis(id=new_id(), name=req.name or f"Analysis {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}", source_type="scan", source_path=req.ab1_dir)
    session.add(analysis)
    session.commit()
    aid = analysis.id
    session.close()
    background_tasks.add_task(_run_analysis, aid, req.gb_dir, req.ab1_dir)
    return {"analysis_id": aid, "status": "pending"}

@router.get("/analyze/{analysis_id}")
def get_analysis_status(analysis_id: str):
    Session = get_session_factory()
    session = Session()
    analysis = session.get(Analysis, analysis_id)
    session.close()
    if not analysis:
        raise HTTPException(404, "Analysis not found")
    return {"id": analysis.id, "name": analysis.name, "status": analysis.status, "total": analysis.total, "ok_count": analysis.ok_count, "wrong_count": analysis.wrong_count, "uncertain_count": analysis.uncertain_count, "created_at": str(analysis.created_at), "finished_at": str(analysis.finished_at) if analysis.finished_at else None}
