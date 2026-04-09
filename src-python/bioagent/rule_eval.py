"""Evaluate deterministic rules against human-reviewed truth files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .main import analyze_folder, parse_review_result_text


def evaluate_dataset(dataset_root: Path, truth_path: Path) -> dict:
    result = analyze_folder(
        ab1_dir=str(dataset_root / "ab1"),
        gb_dir=str(dataset_root / "gb"),
    )
    truth = parse_review_result_text(truth_path.read_text(encoding="utf-8"))

    total = 0
    status_match = 0
    exact_match = 0
    mismatches = []

    for sample in result["samples"]:
        clone = str(sample.get("clone") or "").upper()
        if clone not in truth:
            continue

        total += 1
        pred_status = sample.get("auto_status", sample.get("status"))
        pred_reason = (sample.get("auto_reason", sample.get("reason")) or "").strip()
        truth_status = truth[clone]["status"]
        truth_reason = truth[clone]["reason"]

        if pred_status == truth_status:
            status_match += 1
        if pred_status == truth_status and pred_reason == truth_reason:
            exact_match += 1
            continue

        mismatches.append(
            {
                "clone": clone,
                "pred_status": pred_status,
                "pred_reason": pred_reason,
                "truth_status": truth_status,
                "truth_reason": truth_reason,
                "identity": sample.get("identity"),
                "coverage": sample.get("coverage"),
                "aa_changes_n": sample.get("aa_changes_n"),
                "seq_length": sample.get("seq_length"),
                "rule_id": sample.get("rule_id"),
            }
        )

    return {
        "dataset": dataset_root.name,
        "truth_file": str(truth_path),
        "total": total,
        "status_match": status_match,
        "exact_match": exact_match,
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BioAgent rules against reviewed truth files.")
    parser.add_argument("--data-root", required=True, help="Directory that contains dataset folders like base/pro/promax")
    parser.add_argument("--truth-dir", required=True, help="Directory that contains result.txt/result_pro.txt/result_promax.txt")
    parser.add_argument("--datasets", nargs="+", default=["base", "pro", "promax"])
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    truth_dir = Path(args.truth_dir).resolve()
    truth_map = {
        "base": truth_dir / "result.txt",
        "pro": truth_dir / "result_pro.txt",
        "promax": truth_dir / "result_promax.txt",
    }

    reports = []
    for dataset in args.datasets:
        truth_path = truth_map[dataset]
        reports.append(evaluate_dataset(data_root / dataset, truth_path))

    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return

    for report in reports:
        print(f"=== {report['dataset']} ===")
        print(
            f"status_match={report['status_match']}/{report['total']} "
            f"exact_match={report['exact_match']}/{report['total']}"
        )
        for mismatch in report["mismatches"]:
            print(
                f"- {mismatch['clone']}: pred={mismatch['pred_status']} {mismatch['pred_reason'] or '-'} | "
                f"truth={mismatch['truth_status']} {mismatch['truth_reason'] or '-'} | "
                f"id={mismatch['identity']} cov={mismatch['coverage']} len={mismatch['seq_length']} rule={mismatch['rule_id']}"
            )
        print()


if __name__ == "__main__":
    main()
