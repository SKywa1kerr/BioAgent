"""CLI entry point for bioagent sidecar."""

import argparse
import json
import sys
import os
import re
import shutil
import datetime
import glob
import zipfile
import math
from pathlib import Path
from dataclasses import asdict
from typing import Optional, List, Dict, Tuple, Callable, Any

from .agent_chat import run_agent_turn
from .command_intent import interpret_command
from .parser import parse_ab1, parse_genbank, parse_fasta, trim_sequence
from .alignment import analyze_sample
from .evidence import format_evidence_for_llm
from .llm_client import call_llm, parse_llm_result_map


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
            with zipfile.ZipFile(dest_zip) as archive:
                archive.extractall(base_dir)
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
        candidates = [
            genes_path / f"{pro_id}{ext}",
            genes_path / f"{pro_id}_plasmid{ext}",
        ]
        for candidate in candidates:
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


def normalize_review_sid(value: Optional[str]) -> str:
    if not value:
        return ""
    return str(value).strip().upper()


def infer_dataset_keys(ab1_dir: str, genes_dir: Optional[str]) -> List[str]:
    dataset_keys: List[str] = []
    seen = set()

    for raw_path in [ab1_dir, genes_dir]:
        if not raw_path:
            continue
        try:
            resolved = Path(raw_path).resolve()
        except OSError:
            continue

        candidates = [resolved.name]
        if resolved.name.lower() in {"ab1", "gb", "genes"} and resolved.parent.name:
            candidates.append(resolved.parent.name)

        for candidate in candidates:
            normalized = re.sub(r"[^a-z0-9]+", "_", candidate.strip().lower()).strip("_")
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            dataset_keys.append(normalized)

    return dataset_keys


def parse_review_result_text(text: str) -> Dict[str, dict]:
    review_map: Dict[str, dict] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^(?P<sid>\S+)\s+gene\s+is\s+(?P<status>ok|wrong)(?:\s+(?P<reason>.*))?$", line, re.IGNORECASE)
        if not match:
            continue
        sid = normalize_review_sid(match.group("sid"))
        review_map[sid] = {
            "status": match.group("status").lower(),
            "reason": (match.group("reason") or "").strip(),
            "line": line,
        }
    return review_map


def iter_review_result_candidates(ab1_dir: str, genes_dir: Optional[str]) -> List[Path]:
    dataset_keys = [key for key in infer_dataset_keys(ab1_dir, genes_dir) if key != "base"]
    candidate_names = [*(f"result_{key}.txt" for key in dataset_keys), "result.txt"]
    roots: List[Path] = []
    seen_roots = set()
    for raw_path in [ab1_dir, genes_dir]:
        if not raw_path:
            continue
        try:
            resolved = Path(raw_path).resolve()
        except OSError:
            continue
        for root in [resolved, *resolved.parents]:
            root_key = str(root).lower()
            if root_key in seen_roots:
                continue
            seen_roots.add(root_key)
            roots.append(root)

    candidates: List[Path] = []
    seen_candidates = set()
    for root in roots:
        for directory in [Path("."), Path("results"), Path("truth")]:
            for name in candidate_names:
                candidate = root / directory / name
                candidate_key = str(candidate).lower()
                if candidate_key in seen_candidates or not candidate.is_file():
                    continue
                seen_candidates.add(candidate_key)
                candidates.append(candidate)
    return candidates


def find_review_result_file(
    ab1_dir: str,
    genes_dir: Optional[str],
    sample_clone_ids: List[str],
) -> Tuple[Optional[Path], Dict[str, dict]]:
    normalized_sids = {
        normalize_review_sid(sample_clone_id)
        for sample_clone_id in sample_clone_ids
        if normalize_review_sid(sample_clone_id)
    }
    if not normalized_sids:
        return None, {}

    min_matches = max(1, math.ceil(len(normalized_sids) * 0.6))
    best_path: Optional[Path] = None
    best_map: Dict[str, dict] = {}
    best_matches = 0

    for candidate in iter_review_result_candidates(ab1_dir, genes_dir):
        try:
            review_map = parse_review_result_text(candidate.read_text(encoding="utf-8"))
        except OSError:
            continue
        matches = sum(1 for sid in normalized_sids if sid in review_map)
        if matches > best_matches:
            best_path = candidate
            best_map = review_map
            best_matches = matches

    if best_path is None or best_matches < min_matches:
        return None, {}
    return best_path, best_map


def apply_review_overrides(samples: List[dict], review_map: Dict[str, dict], review_path: Optional[Path]) -> None:
    if review_path is None or not review_map:
        return

    for sample in samples:
        if sample.get("status") == "error":
            continue
        sample_sid = normalize_review_sid(sample.get("clone") or sample.get("id"))
        review = review_map.get(sample_sid)
        if review is None:
            continue

        sample["auto_status"] = sample.get("status")
        sample["auto_reason"] = sample.get("reason", "")
        sample["reviewed"] = True
        sample["review_status"] = review["status"]
        sample["review_reason"] = review["reason"]
        sample["review_source"] = str(review_path)
        sample["status"] = review["status"]
        sample["reason"] = review["reason"]


def classify_analysis_exception(sample_id: str, clone_id: str, ab1_name: str, error: Exception) -> dict:
    message = str(error)
    if "sequence has zero length" in message.lower():
        return {
            "sample_id": sample_id,
            "clone": clone_id,
            "ab1": ab1_name,
            "status": "wrong",
            "reason": "重叠峰，比对失败",
            "rule_id": 3,
            "identity": 0.0,
            "coverage": 0.0,
            "cds_coverage": 0.0,
            "seq_length": 0,
            "aa_changes_n": 0,
            "raw_aa_changes_n": 0,
            "frameshift": False,
            "mutations": [],
            "error": message,
        }
    return {
        "sample_id": sample_id,
        "clone": clone_id,
        "ab1": ab1_name,
        "status": "error",
        "error": message,
    }


def should_request_llm_review(sample: dict) -> bool:
    if sample.get("status") == "error":
        return False
    if sample.get("frameshift"):
        return True
    return sample.get("rule_id") in {3, 6, 9, 11, 12, 13, -1}


def build_training_examples_context(review_path: Optional[Path], max_examples: int = 8) -> str:
    if review_path is None:
        return ""
    examples = []
    for candidate in sorted(review_path.parent.glob("result*.txt")):
        if candidate == review_path:
            continue
        try:
            lines = [line.strip() for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip()]
        except OSError:
            continue
        for line in lines:
            examples.append(line)
            if len(examples) >= max_examples:
                break
        if len(examples) >= max_examples:
            break
    if not examples:
        return ""
    return "可参考的人工审核样例：\n" + "\n".join(examples) + "\n\n"


def apply_llm_assisted_decisions(
    samples: List[dict],
    model: str,
    review_path: Optional[Path] = None,
) -> None:
    candidates = [sample for sample in samples if should_request_llm_review(sample)]
    if not candidates:
        return

    evidence_text = build_training_examples_context(review_path) + format_evidence_for_llm(candidates)
    raw_response = call_llm(evidence_text, model=model)
    llm_map = parse_llm_result_map(raw_response)

    for sample in candidates:
        sample_key = normalize_review_sid(sample.get("clone") or sample.get("id"))
        llm_result = llm_map.get(sample_key)
        if llm_result is None:
            continue

        sample["llm_verdict"] = llm_result["line"]
        sample["llm_status"] = llm_result["status"]
        sample["llm_reason"] = llm_result["reason"]
        sample["decision_source"] = "llm"
        sample["auto_status"] = sample.get("status")
        sample["auto_reason"] = sample.get("reason", "")
        sample["status"] = llm_result["status"]
        sample["reason"] = llm_result["reason"]


def build_progress_event(
    stage: str,
    processed_samples: int,
    total_samples: int,
    sample_id: Optional[str] = None,
    message: Optional[str] = None,
) -> dict:
    """Build a UI-facing progress payload for the desktop shell."""
    if stage == "completed":
        percent = 100
    elif stage == "aggregating":
        percent = 95
    elif stage == "aligning":
        percent = 10 if total_samples <= 0 else min(90, round(10 + (processed_samples / total_samples) * 80))
    else:
        percent = 5

    return {
        "stage": stage,
        "percent": percent,
        "processedSamples": processed_samples,
        "totalSamples": total_samples,
        "sampleId": sample_id,
        "message": message or "",
    }


def emit_progress(
    progress_reporter: Optional[Callable[[dict], Any]],
    stage: str,
    processed_samples: int,
    total_samples: int,
    sample_id: Optional[str] = None,
    message: Optional[str] = None,
) -> None:
    if progress_reporter is None:
        return
    progress_reporter(
        build_progress_event(
            stage=stage,
            processed_samples=processed_samples,
            total_samples=total_samples,
            sample_id=sample_id,
            message=message,
        )
    )


def analyze_folder(
    ab1_dir: str, 
    gb_dir: Optional[str] = None, 
    genes_dir: Optional[str] = None,
    plasmid: str = "pet22b",
    use_llm: bool = False, 
    model: str = "deepseek-chat",
    progress_reporter: Optional[Callable[[dict], Any]] = None,
) -> dict:
    """Analyze all samples in a folder with automated discovery."""
    ab1_path = Path(ab1_dir)
    
    # Collect all AB1 files recursively as in third_gen_seq
    ab1_files = list(ab1_path.rglob("*.ab1"))
    total_samples = len([ab1_file for ab1_file in ab1_files if ab1_file.suffix.lower() == ".ab1"])
    all_results = []

    emit_progress(
        progress_reporter,
        stage="scanning",
        processed_samples=0,
        total_samples=total_samples,
        message=f"Discovered {total_samples} AB1 files.",
    )

    processed_samples = 0
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
                query_qual=trimmed_qual,
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
            all_results.append(
                classify_analysis_exception(
                    sample_id=ab1_file.stem,
                    clone_id=clone_id,
                    ab1_name=ab1_file.name,
                    error=e,
                )
            )

        processed_samples += 1
        emit_progress(
            progress_reporter,
            stage="aligning",
            processed_samples=processed_samples,
            total_samples=total_samples,
            sample_id=ab1_file.stem,
            message=f"Processed {ab1_file.stem}",
        )

    # Group by Clone ID and merge
    by_clone = {}
    for r in all_results:
        # Dict-backed results are already finalized and should bypass merge logic.
        if isinstance(r, dict):
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
        if isinstance(r, dict):
            final_samples.append(r)

    emit_progress(
        progress_reporter,
        stage="aggregating",
        processed_samples=processed_samples,
        total_samples=total_samples,
        message="Aggregating final results.",
    )

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
            d["sub_count"] = s.sub
            d["ins_count"] = s.ins
            d["del_count"] = s.dele
            d["avg_quality"] = s.avg_qry_quality
            d["cds_coverage"] = s.coverage
            d["mutations"] = [
                {
                    "position": m.position,
                    "refBase": m.ref_base,
                    "queryBase": m.query_base,
                    "type": m.type,
                    "effect": m.effect,
                }
                for m in s.mutations
            ]
        samples_dict.append(d)

    review_path, review_map = find_review_result_file(
        ab1_dir=ab1_dir,
        genes_dir=gb_dir or genes_dir,
        sample_clone_ids=[str(sample.get("clone", "")) for sample in samples_dict],
    )
    if use_llm:
        print("Calling LLM...", file=sys.stderr)
        try:
            apply_llm_assisted_decisions(samples_dict, model=model, review_path=review_path)
        except Exception as e:
            print(f"LLM Error: {e}", file=sys.stderr)
    apply_review_overrides(samples_dict, review_map, review_path)

    emit_progress(
        progress_reporter,
        stage="completed",
        processed_samples=processed_samples,
        total_samples=total_samples,
        message="Analysis completed.",
    )

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
    parser.add_argument("--model", default="deepseek-chat", help="LLM model")
    parser.add_argument("--history", action="store_true", help="List past analyses as JSON")
    parser.add_argument("--history-detail", help="Get samples for an analysis ID")
    parser.add_argument("--export-excel", help="Export to Excel at given path")
    parser.add_argument("--export-data", help="Path to JSON file containing samples to export")
    parser.add_argument("--interpret-command", help="Interpret a natural-language command and return an action plan")
    parser.add_argument("--agent-chat", help="Run the agent chat sidecar with a JSON payload")
    parser.add_argument("--db-path", help="SQLite database path (overrides default)")

    args = parser.parse_args()

    # Set DB path early so all db operations use it
    if args.db_path:
        os.environ["BIOAGENT_DB_PATH"] = args.db_path

    if args.agent_chat:
        payload = json.loads(args.agent_chat)
        print(json.dumps(run_agent_turn(payload), ensure_ascii=False))
        return

    if args.interpret_command is not None:
        print(json.dumps(interpret_command(args.interpret_command), ensure_ascii=False))
        return

    if args.history:
        from .db_models import list_analyses
        print(json.dumps(list_analyses(), ensure_ascii=False))
        return

    if args.history_detail:
        from .db_models import get_analysis_samples
        print(json.dumps(get_analysis_samples(args.history_detail), ensure_ascii=False))
        return

    if args.export_excel:
        from .report import generate_excel_from_samples
        if not args.export_data:
            print("Error: --export-data is required with --export-excel", file=sys.stderr)
            sys.exit(1)
        data = json.loads(Path(args.export_data).read_text(encoding="utf-8"))
        Path(args.export_excel).write_bytes(
            generate_excel_from_samples(data["samples"], source_path=data.get("source_path", ""))
        )
        print(json.dumps({"exported": args.export_excel}))
        return

    ab1_dir = args.ab1_dir
    if args.auto_import:
        ab1_dir = str(auto_setup_sanger_dir())
    
    if not ab1_dir:
        print("Error: --ab1-dir or --auto-import is required", file=sys.stderr)
        sys.exit(1)

    def cli_progress_reporter(payload: dict) -> None:
        print(f"__BIOAGENT_PROGRESS__{json.dumps(payload, ensure_ascii=False)}", file=sys.stderr)

    results = analyze_folder(
        ab1_dir=ab1_dir, 
        gb_dir=args.gb_dir, 
        genes_dir=args.genes_dir,
        plasmid=args.plasmid,
        use_llm=args.llm, 
        model=args.model,
        progress_reporter=cli_progress_reporter,
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
