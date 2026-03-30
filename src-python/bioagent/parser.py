"""Parse AB1 and GenBank files."""

from pathlib import Path
from typing import Tuple, Optional
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from .models import ChromatogramData


def parse_ab1(filepath: str) -> Tuple[str, ChromatogramData]:
    """Parse AB1 file and return sequence with chromatogram data."""
    record = SeqIO.read(filepath, "abi")
    seq = str(record.seq)

    # Get quality scores
    quality = record.letter_annotations.get("phred_quality", [])

    # Get trace data
    traces = {
        "A": record.annotations["abif_raw"].get("DATA1", []),
        "T": record.annotations["abif_raw"].get("DATA2", []),
        "G": record.annotations["abif_raw"].get("DATA3", []),
        "C": record.annotations["abif_raw"].get("DATA4", []),
    }

    chrom_data = ChromatogramData(
        traces_a=list(traces["A"]),
        traces_t=list(traces["T"]),
        traces_g=list(traces["G"]),
        traces_c=list(traces["C"]),
        quality=list(quality),
        base_calls=seq,
    )

    return seq, chrom_data


def parse_genbank(filepath: str) -> Tuple[SeqRecord, str, int, int]:
    """Parse GenBank file and return record with CDS info."""
    record = SeqIO.read(filepath, "genbank")
    seq = str(record.seq).upper()

    # Find CDS feature
    cds_start, cds_end = None, None
    for feature in record.features:
        if feature.type == "CDS":
            cds_start = int(feature.location.start) + 1  # 1-based
            cds_end = int(feature.location.end)
            break

    return record, seq, cds_start, cds_end


def parse_fasta(filepath: str) -> Tuple[SeqRecord, str]:
    """Parse FASTA file and return record and sequence."""
    record = SeqIO.read(filepath, "fasta")
    seq = str(record.seq).upper()
    return record, seq


def trim_sequence(seq: str, quality: list, min_quality: int = 20) -> Tuple[str, list]:
    """Trim low-quality ends from sequence using middle-out logic from sanger.py."""
    if not quality:
        return seq, quality

    num = len(quality)
    mid = num // 2
    start = 0
    end = num

    # Find first low-quality base to the left of middle
    for i in range(mid, 0, -1):
        if quality[i] < min_quality:
            start = i
            break

    # Find first low-quality base to the right of middle
    for i in range(mid, num):
        if quality[i] < min_quality:
            end = i
            break

    return seq[start:end], quality[start:end]
