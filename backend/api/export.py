import csv
import io
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from backend.db.database import db_session
from backend.db.models import Analysis, Sample

router = APIRouter()

@router.get("/export/{analysis_id}")
def export_report(analysis_id: str, format: str = Query("csv")):
    with db_session() as session:
        analysis = session.get(Analysis, analysis_id)
        if not analysis:
            raise HTTPException(404, "Analysis not found")
        samples = session.query(Sample).filter(Sample.analysis_id == analysis_id).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["SID","Status","Reason","Rule","Identity","CDS_Coverage","Frameshift","AA_Changes_N","Seq_Length","Avg_Quality"])
    for s in samples:
        writer.writerow([s.sid,s.status,s.reason,s.rule_id,s.identity,s.cds_coverage,s.frameshift,s.aa_changes_n,s.seq_length,s.avg_quality])
    output.seek(0)
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=analysis_{analysis_id}.csv"})
