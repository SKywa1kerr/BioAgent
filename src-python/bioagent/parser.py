"""Parse AB1 and GenBank files."""

from pathlib import Path
from typing import Tuple, Optional, List, Dict
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from .models import ChromatogramData


def is_concave(traces: Dict[str, List[int]], pos: int, base: str) -> bool:
    """Determine whether the trace at given position is concave."""
    if pos < 2 or pos >= len(traces[base]) - 2:
        return False
    x1 = pos - 2
    x2 = pos
    x3 = pos + 2
    func = traces[base.upper()]
    left = (func[x2] - func[x1]) / (x2 - x1)
    right = (func[x3] - func[x1]) / (x3 - x1)
    return left >= right


def peak_borders(base_locations: List[int], traces_len: int, i: int) -> Tuple[int, int]:
    """Determine positions in the trace that correspond to a single chromatogram peak."""
    pos = base_locations[i]
    if i == 0:
        prev = -pos
    else:
        prev = base_locations[i - 1]
    if i == len(base_locations) - 1:
        end = traces_len
        start = (pos - prev) // 2
    else:
        next_pos = base_locations[i + 1]
        width = (next_pos - prev) // 4
        start = max(0, pos - width + 2)
        end = min(pos + width - 2, traces_len)
    return start, end


def area_under_peak(traces: Dict[str, List[int]], start: int, end: int, base: str) -> int:
    """Return area under the curve in a given trace between indices start and end."""
    trace = traces[base.upper()]
    start = max(0, start)
    end = min(end, len(trace) - 1)
    if start > end:
        return 0
    return sum(trace[start : end + 1])


def signal_to_noise(seq: str, base_locations: List[int], traces: Dict[str, List[int]], i: int) -> float:
    """Calculate the signal to noise ratio of a position."""
    base = seq[i]
    if base.upper() == "N":
        return 1
    start, end = peak_borders(base_locations, len(traces["A"]), i)
    areas = {b: area_under_peak(traces, start, end, b) for b in traces.keys()}
    primary = areas[base.upper()]
    secondary = sum([areas[letter.upper()] for letter in areas.keys() if letter != base])
    if secondary == 0:
        return 35
    return primary / secondary


def find_mixed_peaks(seq: str, base_locations: List[int], traces: Dict[str, List[int]], fraction: float = 0.15) -> List[int]:
    """For each position of the sequence, determine if the peak in the chromatogram is 'mixed'."""
    if not base_locations:
        return []
    stn = [signal_to_noise(seq, base_locations, traces, i) for i in range(len(base_locations))]
    avg_stn = sum(stn) / len(stn)
    mixed_peaks = []
    threshold = max(25, avg_stn * 1.35)

    for i, pos in enumerate(base_locations):
        stn_local = stn[max(0, i - 10) : i] + stn[i + 1 : i + 10]
        if not stn_local:
            continue
        signal_to_noise_val = sum(stn_local) / len(stn_local)

        if signal_to_noise_val < threshold:
            continue

        base = seq[i]
        traces_len = len(next(iter(traces.values())))
        if pos >= traces_len:
            continue
        start, end = peak_borders(base_locations, traces_len, i)
        areas = {b: area_under_peak(traces, start, end, b) for b in traces.keys()}
        peaks = {b: values[pos] if pos < len(values) else 0 for b, values in traces.items()}
        if base != "N":
            main_peak = peaks.get(base.upper(), 0)
        else:
            main_peak = max(peaks.values()) if peaks else 0
            base = max(peaks, key=peaks.get) if peaks else "N"

        for letter, area in areas.items():
            if (
                base != letter
                and area > (areas[base.upper()] * fraction)
                and peaks[letter] > (main_peak * fraction)
                and is_concave(traces, pos, letter)
            ):
                mixed_peaks.append(i)
    return mixed_peaks


def parse_ab1(filepath: str) -> Tuple[str, ChromatogramData]:
    """Parse AB1 file and return sequence with chromatogram data."""
    record = SeqIO.read(filepath, "abi")
    seq = str(record.seq)

    # Get quality scores
    quality = record.letter_annotations.get("phred_quality", [])

    # Get trace data - TraceTrack uses DATA9-12 for GATC
    # BioPython's abi parser usually maps DATA1-4 to the bases
    # We'll try to find the correct DATA tags for GATC
    raw = record.annotations.get("abif_raw", {})
    
    # Mapping from TraceTrack: G:9, A:10, T:11, C:12
    # But BioPython often maps them to DATA1-4. We'll check both.
    def get_data(tag_idx):
        return list(raw.get(f"DATA{tag_idx}", []))

    # Use FWO_1 tag for robust channel mapping (recommended by Biopython)
    # FWO_1 contains the order of bases for DATA1-4 (e.g., "GATC")
    fwo = raw.get("FWO_1", b"GATC")
    if isinstance(fwo, bytes):
        fwo = fwo.decode("utf-8")
    
    traces = {}
    for i, base in enumerate(fwo):
        traces[base.upper()] = get_data(i + 1)
    
    # Fallback for DATA9-12 if DATA1-4 are empty or FWO_1 is missing
    if not traces.get("A") and raw.get("DATA9"):
        traces = {
            "G": get_data(9),
            "A": get_data(10),
            "T": get_data(11),
            "C": get_data(12),
        }

    base_locations = list(record.annotations["abif_raw"].get("PLOC1", []))

    mixed_peaks = find_mixed_peaks(seq, base_locations, traces)

    chrom_data = ChromatogramData(
        traces_a=traces["A"],
        traces_t=traces["T"],
        traces_g=traces["G"],
        traces_c=traces["C"],
        quality=list(quality),
        base_calls=seq,
        base_locations=base_locations,
        mixed_peaks=mixed_peaks,
    )

    return seq, chrom_data


def find_orf(seq: str) -> Tuple[Optional[int], Optional[int], str]:
    """Find the longest ORF starting with ATG and ending with TAA, TAG, or TGA.
    Returns (start, end, orientation).
    """
    seq = seq.upper()
    stops = ["TAA", "TAG", "TGA"]
    
    def _find_in_strand(s):
        best = (None, None)
        curr_max = 0
        for frame in range(3):
            for i in range(frame, len(s) - 2, 3):
                if s[i : i + 3] == "ATG":
                    for j in range(i + 3, len(s) - 2, 3):
                        if s[j : j + 3] in stops:
                            length = j + 3 - i
                            if length > curr_max:
                                curr_max = length
                                best = (i + 1, j + 3)
                            break
        return best

    # Forward strand
    f_start, f_end = _find_in_strand(seq)
    
    # Reverse strand
    from Bio.Seq import Seq
    rev_seq = str(Seq(seq).reverse_complement())
    r_start, r_end = _find_in_strand(rev_seq)
    
    f_len = (f_end - f_start) if f_start else 0
    r_len = (r_end - r_start) if r_start else 0
    
    if f_len >= r_len and f_start:
        return f_start, f_end, "FORWARD"
    elif r_start:
        # Convert reverse coordinates back to forward
        # r_start is 1-based from the end
        actual_start = len(seq) - r_end + 1
        actual_end = len(seq) - r_start + 1
        return actual_start, actual_end, "REVERSE"
    
    return None, None, "FORWARD"


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
    
    # Fallback to ORF detection if no CDS found
    if cds_start is None:
        cds_start, cds_end, _ = find_orf(seq)

    return record, seq, cds_start, cds_end


def parse_fasta(filepath: str) -> Tuple[SeqRecord, str]:
    """Parse FASTA file and return record and sequence."""
    record = SeqIO.read(filepath, "fasta")
    seq = str(record.seq).upper()
    return record, seq


def trim_sequence(seq: str, quality: list, min_quality: int = 20) -> Tuple[str, list, int]:
    """Trim low-quality ends from sequence using middle-out logic from sanger.py."""
    if not quality:
        return seq, quality, 0

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

    return seq[start:end], quality[start:end], start
