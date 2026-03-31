"""SQLAlchemy models for analysis history."""
from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, Text, Integer, Float, Boolean, ForeignKey, DateTime
from .db import Base


def new_id() -> str:
    return str(uuid.uuid4())


class Analysis(Base):
    __tablename__ = "analyses"
    id = Column(Text, primary_key=True, default=new_id)
    name = Column(Text)
    source_path = Column(Text)
    status = Column(Text, default="done")
    total = Column(Integer, default=0)
    ok_count = Column(Integer, default=0)
    wrong_count = Column(Integer, default=0)
    uncertain_count = Column(Integer, default=0)
    config_snapshot = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Sample(Base):
    __tablename__ = "samples"
    id = Column(Text, primary_key=True, default=new_id)
    analysis_id = Column(Text, ForeignKey("analyses.id"))
    sid = Column(Text)
    clone = Column(Text)
    status = Column(Text)
    reason = Column(Text)
    rule_id = Column(Integer)
    identity = Column(Float)
    cds_coverage = Column(Float)
    frameshift = Column(Boolean)
    aa_changes = Column(Text)
    aa_changes_n = Column(Integer)
    seq_length = Column(Integer)
    ref_length = Column(Integer)
    avg_quality = Column(Float)
    sub_count = Column(Integer)
    ins_count = Column(Integer)
    del_count = Column(Integer)


def save_analysis(samples_data: list[dict], judgments: list[dict],
                  thresholds: dict, source_path: str = "") -> str:
    """Save analysis + samples to DB. Returns analysis_id."""
    from .db import init_db, db_session
    init_db()
    ok = sum(1 for j in judgments if j["status"] == "ok")
    wrong = sum(1 for j in judgments if j["status"] == "wrong")
    uncertain = sum(1 for j in judgments if j["status"] == "uncertain")
    analysis_id = new_id()

    with db_session() as session:
        session.add(Analysis(
            id=analysis_id,
            name=f"分析 {datetime.now().strftime('%m-%d %H:%M')}",
            source_path=source_path,
            total=len(samples_data),
            ok_count=ok, wrong_count=wrong, uncertain_count=uncertain,
            config_snapshot=json.dumps(thresholds),
        ))
        for sd, jd in zip(samples_data, judgments):
            session.add(Sample(
                id=new_id(), analysis_id=analysis_id,
                sid=sd.get("sample_id", ""),
                clone=sd.get("clone", ""),
                status=jd["status"],
                reason=jd.get("reason", ""),
                rule_id=jd.get("rule", -1),
                identity=sd.get("identity", 0),
                cds_coverage=sd.get("coverage", 0),
                frameshift=sd.get("frameshift", False),
                aa_changes=json.dumps(sd.get("aa_changes", [])),
                aa_changes_n=sd.get("aa_changes_n", 0),
                seq_length=sd.get("seq_length", 0),
                ref_length=sd.get("ref_length", 0),
                avg_quality=sd.get("avg_qry_quality"),
                sub_count=sd.get("sub", 0),
                ins_count=sd.get("ins", 0),
                del_count=sd.get("dele", 0),
            ))
    return analysis_id


def list_analyses() -> list[dict]:
    """Return all past analyses, newest first."""
    from .db import init_db, db_session
    init_db()
    with db_session() as session:
        rows = session.query(Analysis).order_by(Analysis.created_at.desc()).all()
        return [
            {"id": r.id, "name": r.name, "source_path": r.source_path,
             "total": r.total, "ok_count": r.ok_count, "wrong_count": r.wrong_count,
             "uncertain_count": r.uncertain_count,
             "created_at": r.created_at.isoformat() if r.created_at else ""}
            for r in rows
        ]


def get_analysis_samples(analysis_id: str) -> list[dict]:
    """Return all samples for a given analysis."""
    from .db import init_db, db_session
    init_db()
    with db_session() as session:
        rows = session.query(Sample).filter(Sample.analysis_id == analysis_id).all()
        return [
            {"sid": r.sid, "clone": r.clone, "status": r.status,
             "reason": r.reason, "rule_id": r.rule_id,
             "identity": r.identity, "cds_coverage": r.cds_coverage,
             "frameshift": r.frameshift, "aa_changes_n": r.aa_changes_n,
             "seq_length": r.seq_length, "ref_length": r.ref_length}
            for r in rows
        ]
