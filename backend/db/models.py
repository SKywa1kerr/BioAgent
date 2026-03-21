import json
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, Text, Integer, Float, Boolean, ForeignKey, DateTime
from backend.db.database import Base

def new_id() -> str:
    return str(uuid.uuid4())

class Analysis(Base):
    __tablename__ = "analyses"
    id = Column(Text, primary_key=True, default=new_id)
    name = Column(Text)
    source_type = Column(Text)
    source_path = Column(Text)
    status = Column(Text, default="pending")
    total = Column(Integer, default=0)
    ok_count = Column(Integer, default=0)
    wrong_count = Column(Integer, default=0)
    uncertain_count = Column(Integer, default=0)
    config_snapshot = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    finished_at = Column(DateTime, nullable=True)

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
    raw_aa_changes_n = Column(Integer)
    orientation = Column(Text)
    seq_length = Column(Integer)
    ref_length = Column(Integer)
    avg_quality = Column(Float)
    sub_count = Column(Integer)
    ins_count = Column(Integer)
    del_count = Column(Integer)
    ref_gapped = Column(Text)
    qry_gapped = Column(Text)
    quality_scores = Column(Text)
    raw_data = Column(Text)


def save_analysis_with_samples(
    samples: list[dict],
    judgments: list[dict],
    thresholds: dict,
    name: str = "",
    source_type: str = "scan",
    source_path: str = "",
) -> str:
    """Save analysis + samples to database. Returns analysis_id."""
    from backend.db.database import init_db, db_session

    init_db()
    ok = sum(1 for j in judgments if j["status"] == "ok")
    wrong = sum(1 for j in judgments if j["status"] == "wrong")
    uncertain = sum(1 for j in judgments if j["status"] == "uncertain")

    analysis_id = new_id()
    with db_session() as session:
        session.add(Analysis(
            id=analysis_id,
            name=name or f"分析 {datetime.now().strftime('%m-%d %H:%M')}",
            source_type=source_type, source_path=source_path,
            status="done", total=len(samples),
            ok_count=ok, wrong_count=wrong, uncertain_count=uncertain,
            config_snapshot=json.dumps(thresholds),
            finished_at=datetime.now(timezone.utc),
        ))
        for sd, jd in zip(samples, judgments):
            session.add(Sample(
                id=new_id(), analysis_id=analysis_id,
                sid=sd["sid"], clone=sd.get("clone", ""),
                status=jd["status"], reason=jd.get("reason", ""),
                rule_id=jd.get("rule"),
                identity=sd["identity"], cds_coverage=sd["cds_coverage"],
                frameshift=sd["frameshift"],
                aa_changes=json.dumps(sd.get("aa_changes", [])),
                aa_changes_n=sd.get("aa_changes_n", 0),
                raw_aa_changes_n=sd.get("raw_aa_changes_n", 0),
                orientation=sd.get("orientation", ""),
                seq_length=sd.get("seq_length", 0),
                ref_length=sd.get("ref_length", 0),
                avg_quality=sd.get("avg_qry_quality"),
                sub_count=sd.get("sub", 0), ins_count=sd.get("ins", 0),
                del_count=sd.get("del", 0),
                ref_gapped=sd.get("ref_gapped", ""),
                qry_gapped=sd.get("qry_gapped", ""),
                quality_scores=json.dumps(sd.get("quality_scores", []) or []),
                raw_data=json.dumps(sd, default=str),
            ))
    return analysis_id
