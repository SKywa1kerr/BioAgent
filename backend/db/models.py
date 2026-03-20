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
