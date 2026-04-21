"""Truth-set calibration tool. Dev only — not imported by runtime.

Usage:
    python -m scripts.calibrate --dataset base
    python -m scripts.calibrate --dataset pro --data-dir ./data
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


CHINESE_KEYWORDS: dict[str, str] = {
    "未测通": "untested",
    "测序失败": "untested",
    "片段缺失": "untested",
    "重叠峰": "uncertain",
    "生工重叠峰": "uncertain",
    "比对失败": "uncertain",
    "移码": "wrong",
}

AA_CHANGE_RE = re.compile(r"^[A-Z*]\d+[A-Z*]$")
LINE_RE = re.compile(
    r"^(?P<sid>C\d+-\w+)\s+gene\s+is\s+(?P<status>ok|wrong)(?P<rest>.*)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TruthRecord:
    sid: str
    status: str  # ok | wrong | uncertain | untested
    aa: list[str] = field(default_factory=list)
    note: str | None = None


def parse_truth_line(line: str) -> TruthRecord | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    # Chinese-keyword-only lines (no "gene is ok/wrong")
    if "gene is" not in stripped.lower():
        for kw, mapped in CHINESE_KEYWORDS.items():
            if kw in stripped:
                m = re.match(r"^(C\d+-\w+)", stripped)
                sid = m.group(1) if m else stripped.split()[0]
                return TruthRecord(sid=sid, status=mapped, note=stripped)

    m = LINE_RE.match(stripped)
    if not m:
        return None
    sid = m.group("sid")
    status = m.group("status").lower()
    rest = (m.group("rest") or "").strip()
    aa: list[str] = []
    note_tokens: list[str] = []
    for tok in rest.split():
        if AA_CHANGE_RE.match(tok):
            aa.append(tok)
        else:
            note_tokens.append(tok)
    note = " ".join(note_tokens) or None
    if note:
        for kw, mapped in CHINESE_KEYWORDS.items():
            if kw in note:
                status = mapped
                break
    return TruthRecord(sid=sid, status=status, aa=aa, note=note)


def load_truth_file(path: Path) -> list[TruthRecord]:
    records: list[TruthRecord] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        rec = parse_truth_line(raw)
        if rec is not None:
            records.append(rec)
    return records


def aa_set(aa_changes) -> set[str]:
    return {x.strip() for x in (aa_changes or []) if x and isinstance(x, str)}


def summarize_diff(
    truth: dict[str, "TruthRecord"],
    engine: dict[str, dict],
) -> dict:
    agree = 0
    total = 0
    mismatches: list[dict] = []
    for sid, rec in truth.items():
        total += 1
        sample = engine.get(sid)
        actual_bucket = (sample or {}).get("bucket") or "unknown"
        actual_aa = aa_set((sample or {}).get("aa_changes"))
        expected_aa = aa_set(rec.aa)
        bucket_match = rec.status == actual_bucket
        aa_match = (
            rec.status != "wrong"
            or expected_aa == actual_aa
            or not expected_aa
        )
        if bucket_match and aa_match:
            agree += 1
        else:
            mismatches.append(
                {
                    "sid": sid,
                    "expected": rec.status,
                    "actual": actual_bucket,
                    "expected_aa": sorted(expected_aa),
                    "actual_aa": sorted(actual_aa),
                    "note": rec.note,
                }
            )
    return {"agree": agree, "total": total, "mismatches": mismatches}


def run_engine(dataset: str, data_dir: Path) -> dict[str, dict]:
    from core.alignment import analyze_dataset  # lazy import
    results = analyze_dataset(dataset, data_dir)
    return {r["sid"]: r for r in results}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="calibrate")
    ap.add_argument(
        "--dataset", required=True, choices=["base", "pro", "promax"]
    )
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument(
        "--truth",
        type=Path,
        help="Override truth file path (default: truth/result[_pro|_promax].txt)",
    )
    args = ap.parse_args(argv)

    if args.truth:
        truth_path = args.truth
    else:
        suffix = "" if args.dataset == "base" else f"_{args.dataset}"
        truth_path = Path("truth") / f"result{suffix}.txt"

    if not truth_path.exists():
        print(f"Truth file missing: {truth_path}", file=sys.stderr)
        return 2

    truth_records = load_truth_file(truth_path)
    truth = {r.sid: r for r in truth_records}

    try:
        engine = run_engine(args.dataset, args.data_dir)
    except FileNotFoundError as exc:
        print(f"Engine input missing: {exc}", file=sys.stderr)
        return 3

    summary = summarize_diff(truth, engine)
    total = summary["total"] or 1
    print(
        f"Match rate: {summary['agree']}/{summary['total']} "
        f"= {summary['agree'] / total:.1%}"
    )
    for m in summary["mismatches"]:
        print(
            f"  {m['sid']}: expected={m['expected']} actual={m['actual']} "
            f"exp_aa={m['expected_aa']} act_aa={m['actual_aa']} note={m['note']}"
        )
    return 0 if summary["agree"] / total >= 0.90 else 1


if __name__ == "__main__":
    sys.exit(main())
