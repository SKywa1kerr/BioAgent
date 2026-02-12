#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run.py — BioAgent main entry point.

Two-stage pipeline:
  Stage 1: Bioinformatics (GB/AB1 alignment, mutation detection, AA translation)
  Stage 2: LLM judgment (Claude API for QC verdict)

Usage:
  python run.py --dataset base
  python run.py --dataset pro
  python run.py --dataset promax
  python run.py --dataset promax --model claude-sonnet-4-5-20250929
"""

import argparse
import sys
from pathlib import Path

from core.alignment import analyze_dataset
from core.evidence import format_evidence_for_llm, format_evidence_table
from core.llm_client import call_claude, parse_llm_result


def main():
    parser = argparse.ArgumentParser(
        description="BioAgent: Sanger sequencing QC & mutation analysis"
    )
    parser.add_argument(
        "--dataset", required=True, choices=["base", "pro", "promax"],
        help="Dataset to analyze"
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-5-20250929",
        help="Claude model to use (default: claude-sonnet-4-5-20250929)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: outputs/<dataset>)"
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM stage; output raw bioinformatics results only"
    )
    args = parser.parse_args()

    root = Path(__file__).parent.resolve()
    data_dir = root / "data"
    output_dir = Path(args.output_dir) if args.output_dir else root / "outputs" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    html_dir = output_dir / "html"

    # ── Stage 1: Bioinformatics analysis ─────────────────────────────────
    print(f"{'='*60}")
    print(f" BioAgent — Dataset: {args.dataset}")
    print(f"{'='*60}")
    print(f"\n[Stage 1] Running bioinformatics analysis...")

    samples = analyze_dataset(args.dataset, data_dir, out_html_dir=html_dir)

    if not samples:
        print("[ERR] No samples analyzed. Check data directory.")
        sys.exit(1)

    print(f"\n  {len(samples)} samples analyzed.\n")

    # Save evidence table
    table_text = format_evidence_table(samples)
    evidence_path = output_dir / "evidence.txt"
    evidence_path.write_text(table_text, encoding="utf-8")
    print(f"  Evidence table saved to: {evidence_path.name}")
    print(f"\n{table_text}\n")

    # ── Stage 2: LLM judgment ────────────────────────────────────────────
    if args.no_llm:
        print("[INFO] --no-llm flag set. Skipping LLM stage.")
        # Write basic bioinformatics-only result
        bio_lines = []
        for s in samples:
            if s["aa_changes"]:
                bio_lines.append(f"{s['sid']} gene is wrong {' '.join(s['aa_changes'])}")
            elif s["frameshift"]:
                bio_lines.append(f"{s['sid']} gene is wrong")
            elif s.get("other_read_issues"):
                bio_lines.append(f"{s['sid']} gene is wrong")
            else:
                bio_lines.append(f"{s['sid']} gene is ok")
        result_path = output_dir / "result.txt"
        result_path.write_text("\n".join(bio_lines) + "\n", encoding="utf-8")
        print(f"\n  Bio-only result saved to: {result_path}")
        return

    print(f"[Stage 2] Calling Claude API ({args.model})...")
    evidence_text = format_evidence_for_llm(samples)

    try:
        raw_response = call_claude(evidence_text, model=args.model)
    except RuntimeError as e:
        print(f"\n[ERR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERR] Claude API call failed: {e}")
        sys.exit(1)

    # Save raw LLM response
    raw_path = output_dir / "llm_raw.txt"
    raw_path.write_text(raw_response, encoding="utf-8")
    print(f"  Raw LLM response saved to: {raw_path.name}")

    # Parse and save final result
    result_lines = parse_llm_result(raw_response)
    result_text = "\n".join(result_lines)

    result_path = output_dir / "result.txt"
    result_path.write_text(result_text + "\n", encoding="utf-8")

    print(f"\n{'='*60}")
    print(f" Results ({len(result_lines)} samples)")
    print(f"{'='*60}")
    for line in result_lines:
        print(f"  {line}")

    print(f"\n  Result saved to: {result_path}")

    # ── Optional: Compare with truth ─────────────────────────────────────
    truth_map = {
        "base": root / "truth" / "result.txt",
        "pro": root / "truth" / "result_pro.txt",
        "promax": root / "truth" / "result_promax.txt",
    }
    truth_path = truth_map.get(args.dataset)
    if truth_path and truth_path.exists():
        print(f"\n[Truth comparison] {truth_path.name}")
        truth_lines = [
            l.strip() for l in truth_path.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        # Build SID -> line map for both
        def sid_from_line(line):
            return line.split()[0] if line.split() else ""

        truth_map_by_sid = {sid_from_line(l): l for l in truth_lines}
        pred_map_by_sid = {sid_from_line(l): l for l in result_lines}

        match = mismatch = missing = 0
        for sid, truth_line in truth_map_by_sid.items():
            if sid not in pred_map_by_sid:
                print(f"  MISSING: {sid}")
                missing += 1
                continue
            pred_line = pred_map_by_sid[sid]
            # Compare ok/wrong status
            truth_status = "wrong" if "wrong" in truth_line else "ok"
            pred_status = "wrong" if "wrong" in pred_line else "ok"
            if truth_status == pred_status:
                match += 1
            else:
                mismatch += 1
                print(f"  MISMATCH: {sid}")
                print(f"    truth: {truth_line}")
                print(f"    pred:  {pred_line}")

        total = match + mismatch + missing
        print(f"\n  Accuracy (ok/wrong): {match}/{total} ({match/total*100:.0f}%)" if total else "")


if __name__ == "__main__":
    main()
