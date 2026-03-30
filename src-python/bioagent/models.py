"""Data models for analysis results."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class Mutation:
    position: int
    ref_base: str
    query_base: str
    ref_codon: Optional[str] = None
    query_codon: Optional[str] = None
    ref_aa: Optional[str] = None
    query_aa: Optional[str] = None
    type: str = "substitution"
    effect: Optional[str] = None


@dataclass
class AlignmentResult:
    sample_id: str
    ref_sequence: str
    query_sequence: str
    aligned_query: str
    matches: List[bool]
    mutations: List[Mutation]
    cds_start: int
    cds_end: int
    frameshift: bool
    identity: float
    coverage: float
    aligned_ref_g: str = ""
    aligned_query_g: str = ""
    # Added fields from core/alignment.py
    clone: str = ""
    ab1: str = ""
    gb: str = ""
    orientation: str = "FORWARD"
    aa_changes: List[str] = field(default_factory=list)
    aa_changes_n: int = 0
    raw_aa_changes_n: int = 0
    has_indel: bool = False
    sub: int = 0
    ins: int = 0
    dele: int = 0
    seq_length: int = 0
    ref_length: int = 0
    avg_qry_quality: Optional[float] = None
    dual_read: bool = False
    total_cds_coverage: Optional[float] = None
    read_conflict: Optional[bool] = None
    other_reads: List[str] = field(default_factory=list)
    other_read_issues: List[str] = field(default_factory=list)
    other_read_notes: List[str] = field(default_factory=list)
    _cds_positions: Dict[int, str] = field(default_factory=dict)
    # Chromatogram data
    traces_a: List[int] = field(default_factory=list)
    traces_t: List[int] = field(default_factory=list)
    traces_g: List[int] = field(default_factory=list)
    traces_c: List[int] = field(default_factory=list)
    quality: List[int] = field(default_factory=list)
    base_locations: List[int] = field(default_factory=list)
    mixed_peaks: List[int] = field(default_factory=list)


@dataclass
class ChromatogramData:
    traces_a: List[int]
    traces_t: List[int]
    traces_g: List[int]
    traces_c: List[int]
    quality: List[int]
    base_calls: str
    base_locations: List[int] = field(default_factory=list)
    mixed_peaks: List[int] = field(default_factory=list)
