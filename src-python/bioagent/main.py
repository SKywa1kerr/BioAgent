from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from core.alignment import analyze_dataset
from core.evidence import format_evidence_for_llm, format_evidence_table
from core.llm_client import call_llm, parse_llm_result
from bioagent.mcp_server import run_stdio_server


DEFAULT_MODEL = "google/gemma-3-27b-it:free"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BioAgent: Sanger sequencing QC & mutation analysis"
    )
    parser.add_argument(
        "--dataset", choices=["base", "pro", "promax"],
        help="Dataset to analyze"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"LLM model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: outputs/<dataset>)"
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM stage; output raw bioinformatics results only"
    )
    parser.add_argument(
        "--mcp-server", action="store_true",
        help="Run the MCP stdio server instead of the normal CLI flow"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mcp_server:
        run_stdio_server()
        return 0

    if not args.dataset:
        parser.error("--dataset is required unless --mcp-server is used")

    root = _PACKAGE_ROOT
    data_dir = root / "data"
    output_dir = Path(args.output_dir) if args.output_dir else root / "outputs" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    html_dir = output_dir / "html"

    print(f"{'=' * 60}")
    print(f" BioAgent — Dataset: {args.dataset}")
    print(f"{'=' * 60}")
    print("\n[Stage 1] Running bioinformatics analysis...")

    samples = analyze_dataset(args.dataset, data_dir, out_html_dir=html_dir)
    if not samples:
        print("[ERR] No samples analyzed. Check data directory.")
        return 1

    print(f"\n  {len(samples)} samples analyzed.\n")

    table_text = format_evidence_table(samples)
    evidence_path = output_dir / "evidence.txt"
    evidence_path.write_text(table_text, encoding="utf-8")
    print(f"  Evidence table saved to: {evidence_path.name}")
    print(f"\n{table_text}\n")

    if args.no_llm:
        print("[INFO] --no-llm flag set. Skipping LLM stage.")
        bio_lines = []
        for sample in samples:
            if sample["aa_changes"]:
                bio_lines.append(f"{sample['sid']} gene is wrong {' '.join(sample['aa_changes'])}")
            elif sample["frameshift"]:
                bio_lines.append(f"{sample['sid']} gene is wrong")
            elif sample.get("other_read_issues"):
                bio_lines.append(f"{sample['sid']} gene is wrong")
            else:
                bio_lines.append(f"{sample['sid']} gene is ok")
        result_path = output_dir / "result.txt"
        result_path.write_text("\n".join(bio_lines) + "\n", encoding="utf-8")
        print(f"\n  Bio-only result saved to: {result_path}")
        return 0

    print(f"[Stage 2] Calling LLM API ({args.model})...")
    evidence_text = format_evidence_for_llm(samples)

    try:
        raw_response = call_llm(evidence_text, model=args.model)
    except RuntimeError as exc:
        print(f"\n[ERR] {exc}")
        return 1
    except Exception as exc:
        print(f"\n[ERR] LLM API call failed: {exc}")
        return 1

    raw_path = output_dir / "llm_raw.txt"
    raw_path.write_text(raw_response, encoding="utf-8")
    print(f"  Raw LLM response saved to: {raw_path.name}")

    result_lines = parse_llm_result(raw_response)
    result_text = "\n".join(result_lines)
    result_path = output_dir / "result.txt"
    result_path.write_text(result_text + "\n", encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f" Results ({len(result_lines)} samples)")
    print(f"{'=' * 60}")
    for line in result_lines:
        print(f"  {line}")

    print(f"\n  Result saved to: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
