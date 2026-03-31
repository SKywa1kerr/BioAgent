"""CLI entry point for bioagent sidecar."""

import argparse
import json
import sys
import os
import re
import shutil
import datetime
import glob
from pathlib import Path
from dataclasses import asdict
from typing import Optional, List, Dict, Tuple

from .parser import parse_ab1, parse_genbank, parse_fasta, trim_sequence
from .alignment import analyze_sample
from .evidence import format_evidence_for_llm
from .llm_client import call_llm, parse_llm_result


def auto_setup_sanger_dir() -> Path:
    """Automatically setup directory from Downloads as in sanger.py."""
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    base_dir = Path.cwd() / current_date
    base_dir.mkdir(exist_ok=True)

    downloads_path = Path.home() / "Downloads"
    zip_patterns = ["范瑜*.zip", "S*.zip"]
    zip_files = []
    for pattern in zip_patterns:
        zip_files.extend(glob.glob(str(downloads_path / pattern)))
    
    zip_files = list(set(zip_files))
    if zip_files:
        for idx, zip_file in enumerate(zip_files):
            zip_path = Path(zip_file)
            dest_zip = base_dir / f"{current_date}-{idx}.zip"
            shutil.move(str(zip_path), str(dest_zip))
            # Unzip
            os.system(f"unzip -q '{dest_zip}' -d '{base_dir}'")
        print(f"Auto-setup complete in {base_dir}", file=sys.stderr)
    else:
        print("Warning: No matching zip files found in Downloads", file=sys.stderr)
    
    return base_dir


def get_plasmid_template(plasmid: str, insert_seq: str) -> str:
    """Apply plasmid template logic from sanger.py."""
    p = plasmid.lower()
    insert_seq = insert_seq.upper()
    if p == "pet22b":
        prefix = "TAATACGACTCACTATAGGGGAATTGTGAGCGGATAACAATTCCCCTCTAGAAATAATTTTGTTTAACTTTAAGAAGGAGATATACAT"
        suffix = "GATCCGGCTGCTAACAAAGCCCGAAAGGAAGCTGAGTTGGCTGCTGCCACCGCTGAGCAATAACTAGCATAACCCCTTGGGGCCTCTAAACGGGTCTTGAGGGGTTTTTTGCTGAAAGGAGGAACTATATCCGGATTGGCGAATGGGACGCGCCCTGTAGCGGCGCATTAAGCGCGGCGGGTGTGGTGGTTACGCGCAGCGTGACCGCTACACTTGCCAGCGCCCTAGCGCCCGCTCCTTTCGCTTTCTTCCCTTCCTTTCTCGCCACGTTCGCCGGCTTTCCCCGTCAAGCTCTAAATCGGGGGCTCCCTTTAGGGTTCCGATTTAGTGCTTTACGGCACCTCGACCCCAAAAAACTTGATTAGGGTGATGGTTCACGTAGTGGGCCATCGCCCTGATAGACGGTTTTTCGCCCTTTGACGTTGGAGTCCACGTTCTTTAATAGTGGACTCTTGTTCCAAACTGGAACAACACTCAACCCTATCTCGGTCTATTCTTTTGATTTATAAGGGATTTTGCCGATTTCGGCCTATTGGTTAAAAAATGAGCTGATTTAACAAAAATTTAACGCGAATTTTAACAAAATATTAACGTTTACAATTT"
        return prefix + insert_seq + suffix
    elif p == "pet15b":
        prefix = "TAATACGACTCACTATAGGGGAATTGTGAGCGGATAACAATTCCCCTCTAGAAATAATTTTGTTTAACTTTAAGAAGGAGATATACCATGGGCAGCAGCCATCATCATCATCATCACAGCAGCGGCCTGGTGCCGCGCGGCAGCCAT"
        return prefix.upper() + insert_seq
    return insert_seq


def find_reference_file(clone_id: str, genes_dir: Optional[str]) -> Tuple[Optional[Path], Optional[str]]:
    """Find reference file (GenBank or FASTA) for a clone ID."""
    if not genes_dir:
        return None, None
    
    genes_path = Path(genes_dir)
    # Extract pro_id (e.g., C123 from C123-1)
    pro_id = clone_id.split("-")[0].upper()
    
    # Try GenBank first
    for ext in [".gb", ".gbk"]:
        candidate = genes_path / f"{pro_id}{ext}"
        if candidate.exists():
            return candidate, "genbank"
    
    # Try FASTA
    for ext in [".fasta", ".fa", ".dna.fasta"]:
        # sanger.py uses B{i}_dna.fasta or C{i}_dna.fasta
        candidates = [
            genes_path / f"{pro_id}{ext}",
            genes_path / f"{pro_id}_dna{ext}"
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate, "fasta"
                
    return None, None


def analyze_folder(
    ab1_dir: str, 
    gb_dir: Optional[str] = None, 
    genes_dir: Optional[str] = None,
    plasmid: str = "pet22b",
    use_llm: bool = False, 
    model: str = "google/gemma-3-27b-it:free"
) -> dict:
    """Analyze all samples in a folder with automated discovery."""
    ab1_path = Path(ab1_dir)
    
    # Collect all AB1 files recursively as in third_gen_seq
    ab1_files = list(ab1_path.rglob("*.ab1"))
    all_results = []

    for ab1_file in ab1_files:
        # Skip non-ab1 files that might be picked up by rglob
        if ab1_file.suffix.lower() != ".ab1":
            continue

        # Extract clone ID using multiple strategies from sanger.py
        clone_id = None
        
        # Strategy 1: Regex from sanger.py
        match = re.search(r"([A-C][A-Z0-9]{3}-[A-Z0-9])", ab1_file.name, re.IGNORECASE)
        if match:
            clone_id = match.group().upper()
        
        # Strategy 2: General ID pattern
        if not clone_id:
            match = re.search(r"([A-Z][0-9]+-[0-9A-Z]+)", ab1_file.name, re.IGNORECASE)
            if match:
                clone_id = match.group().upper()
        
        # Strategy 3: Underscore split (sanger.py: abi_file.split('_')[2])
        if not clone_id:
            parts = ab1_file.stem.split("_")
            if len(parts) >= 3:
                clone_id = parts[2].replace("(", "").replace(")", "").upper()
        
        # Strategy 4: Simple hyphen split
        if not clone_id:
            clone_id = ab1_file.stem.split("-")[0].upper()

        # Find reference
        ref_file, ref_type = None, None
        if gb_dir:
            ref_file, ref_type = find_reference_file(clone_id, gb_dir)
        
        if not ref_file and genes_dir:
            ref_file, ref_type = find_reference_file(clone_id, genes_dir)

        if not ref_file:
            print(f"Warning: No reference file found for {clone_id}", file=sys.stderr)
            # Create a dummy result to indicate missing reference in UI
            all_results.append({
                "sample_id": ab1_file.stem,
                "clone": clone_id,
                "ab1": ab1_file.name,
                "status": "error",
                "error": "Missing reference file"
            })
            continue

        try:
            # Parse files
            query_seq, chrom_data = parse_ab1(str(ab1_file))
            
            if ref_type == "genbank":
                _, ref_seq, cds_start, cds_end = parse_genbank(str(ref_file))
            else: # FASTA
                _, raw_insert_seq = parse_fasta(str(ref_file))
                ref_seq = get_plasmid_template(plasmid, raw_insert_seq)
                cds_start = ref_seq.find(raw_insert_seq.upper()) + 1
                cds_end = cds_start + len(raw_insert_seq) - 1

            # Trim sequence
            trimmed_seq, trimmed_qual, trim_start = trim_sequence(query_seq, chrom_data.quality)

            # Analyze
            result = analyze_sample(
                sample_id=ab1_file.stem,
                ref_seq=ref_seq,
                query_seq=trimmed_seq,
                cds_start=cds_start or 1,
                cds_end=cds_end or len(ref_seq),
                query_qual=trimmed_qual
            )
            
            result.clone = clone_id
            result.ab1 = ab1_file.name
            result.gb = ref_file.name
            
            # Add chromatogram data (trimmed to match query_seq)
            trim_end = trim_start + len(trimmed_seq)
            
            # Calculate trace range to trim (with 20 points padding)
            if chrom_data.base_locations:
                locs = chrom_data.base_locations[trim_start:trim_end]
                trace_start = max(0, min(locs) - 20) if locs else 0
                trace_end = min(len(chrom_data.traces_a), max(locs) + 20) if locs else len(chrom_data.traces_a)
            else:
                trace_start = 0
                trace_end = len(chrom_data.traces_a)

            result.traces_a = chrom_data.traces_a[trace_start:trace_end]
            result.traces_t = chrom_data.traces_t[trace_start:trace_end]
            result.traces_g = chrom_data.traces_g[trace_start:trace_end]
            result.traces_c = chrom_data.traces_c[trace_start:trace_end]
            result.quality = chrom_data.quality[trim_start:trim_end]
            
            # Offset base locations and mixed peaks to match trimmed traces
            result.base_locations = [loc - trace_start for loc in chrom_data.base_locations[trim_start:trim_end]]
            result.mixed_peaks = [p - trim_start for p in chrom_data.mixed_peaks if trim_start <= p < trim_end]

            all_results.append(result)
            # print(f"Analyzed: {clone_id} ({ab1_file.name})", file=sys.stderr)

        except Exception as e:
            print(f"Error analyzing {ab1_file}: {e}", file=sys.stderr)
            all_results.append({
                "sample_id": ab1_file.stem,
                "clone": clone_id,
                "ab1": ab1_file.name,
                "status": "error",
                "error": str(e)
            })

    # Group by Clone ID and merge
    by_clone = {}
    for r in all_results:
        # Skip error results for merging but keep them for final output
        if isinstance(r, dict) and r.get("status") == "error":
            continue
        
        cid = r.clone
        if cid not in by_clone:
            by_clone[cid] = []
        by_clone[cid].append(r)

    final_samples = []
    processed_clones = set()

    for cid, entries in by_clone.items():
        processed_clones.add(cid)
        if len(entries) == 1:
            best = entries[0]
            final_samples.append(best)
        else:
            # Merge logic for multiple reads of same clone
            entries_sorted = sorted(entries, key=lambda x: x.identity, reverse=True)
            best = entries_sorted[0]
            best.dual_read = True
            best.other_reads = [f"{e.ab1}(id={e.identity:.4f},cov={e.coverage:.3f})" for e in entries_sorted[1:]]
            
            # Merge CDS coverage
            merged_positions = {}
            read_conflict = False
            for e in entries_sorted:
                for pos, base in e._cds_positions.items():
                    if pos in merged_positions:
                        if merged_positions[pos] != base:
                            read_conflict = True
                    else:
                        merged_positions[pos] = base
            
            cds_len = best.cds_end - best.cds_start + 1
            best.total_cds_coverage = round(len(merged_positions) / cds_len, 4) if cds_len > 0 else 0.0
            
            best_authoritative = (best.identity >= 0.99 and best.coverage >= 0.8)
            if not best_authoritative:
                best.read_conflict = read_conflict
            
            final_samples.append(best)

    # Add back the error results
    for r in all_results:
        if isinstance(r, dict) and r.get("status") == "error":
            final_samples.append(r)

    # Convert to dict for JSON output
    samples_dict = []
    for s in final_samples:
        if isinstance(s, dict):
            # Error result
            d = s
            d["id"] = s["sample_id"]
        else:
            d = asdict(s)
            from .rules import judge_sample
            judgment = judge_sample(asdict(s))
            d["status"] = judgment["status"]
            d["reason"] = judgment.get("reason", "")
            d["rule_id"] = judgment.get("rule", -1)
            d["id"] = s.sample_id
            d["mutations"] = [
                {
                    "position": m.position,
                    "refBase": m.ref_base,
                    "queryBase": m.query_base,
                    "effect": m.effect,
                }
                for m in s.mutations
            ]
        samples_dict.append(d)

    # LLM Stage (only for non-error samples)
    valid_samples = [s for s in final_samples if not isinstance(s, dict)]
    llm_results = {}
    if use_llm and valid_samples:
        print("Calling LLM...", file=sys.stderr)
        evidence_text = format_evidence_for_llm([asdict(s) for s in valid_samples])
        try:
            raw_response = call_llm(evidence_text, model=model)
            result_lines = parse_llm_result(raw_response)
            for line in result_lines:
                parts = line.split(maxsplit=1)
                if parts:
                    llm_results[parts[0]] = line
        except Exception as e:
            print(f"LLM Error: {e}", file=sys.stderr)

    for s in samples_dict:
        if s["id"] in llm_results:
            s["llm_verdict"] = llm_results[s["id"]]

    return {"samples": samples_dict}


def main():
    parser = argparse.ArgumentParser(description="BioAgent analysis sidecar")
    parser.add_argument("--ab1-dir", help="Directory containing AB1 files")
    parser.add_argument("--gb-dir", help="Directory containing GenBank files")
    parser.add_argument("--genes-dir", help="Directory containing reference FASTA files")
    parser.add_argument("--auto-import", action="store_true", help="Automatically setup from Downloads")
    parser.add_argument("--plasmid", default="pet22b", help="Plasmid template (pet22b, pet15b)")
    parser.add_argument("--output", "-o", help="Output JSON file (default: stdout)")
    parser.add_argument("--llm", action="store_true", help="Use LLM for judgment")
    parser.add_argument("--model", default="google/gemma-3-27b-it:free", help="LLM model")
    parser.add_argument("--history", action="store_true", help="List past analyses as JSON")
    parser.add_argument("--history-detail", help="Get samples for an analysis ID")

    args = parser.parse_args()

    if args.history:
        from .db_models import list_analyses
        print(json.dumps(list_analyses(), ensure_ascii=False))
        return

    if args.history_detail:
        from .db_models import get_analysis_samples
        print(json.dumps(get_analysis_samples(args.history_detail), ensure_ascii=False))
        return

    ab1_dir = args.ab1_dir
    if args.auto_import:
        ab1_dir = str(auto_setup_sanger_dir())
    
    if not ab1_dir:
        print("Error: --ab1-dir or --auto-import is required", file=sys.stderr)
        sys.exit(1)

    results = analyze_folder(
        ab1_dir=ab1_dir, 
        gb_dir=args.gb_dir, 
        genes_dir=args.genes_dir,
        plasmid=args.plasmid,
        use_llm=args.llm, 
        model=args.model
    )

    # Save to database
    from .rules import load_thresholds
    from .db_models import save_analysis
    valid = [s for s in results["samples"] if s.get("status") != "error"]
    judgments = [{"sid": s["id"], "status": s["status"], "reason": s.get("reason", ""), "rule": s.get("rule_id", -1)} for s in valid]
    if valid:
        save_analysis(valid, judgments, load_thresholds(), source_path=ab1_dir or "")

    # Use compact JSON (no indent) to reduce payload size for large trace arrays
    output = json.dumps(results, separators=(',', ':'))

    if args.output:
        Path(args.output).write_text(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
