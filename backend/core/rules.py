"""Deterministic rules engine for Sanger sequencing QC judgment."""
from __future__ import annotations
from typing import TypedDict
from pathlib import Path
import yaml

DEFAULT_CONFIG = Path(__file__).parent.parent / "rules_config.yaml"

def load_thresholds(config_path: Path = DEFAULT_CONFIG) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)["thresholds"]

class SampleInput(TypedDict, total=False):
    sid: str
    identity: float
    cds_coverage: float
    frameshift: bool
    aa_changes: list[str]
    aa_changes_n: int
    seq_length: int
    other_read_issues: list[str] | None

def judge_sample(sample: dict, thresholds: dict | None = None) -> dict:
    t = thresholds or load_thresholds()
    sid = sample["sid"]
    identity = sample["identity"]
    cds_cov = sample["cds_coverage"]
    frameshift = sample["frameshift"]
    aa_changes = sample.get("aa_changes", [])
    aa_n = sample.get("aa_changes_n", len(aa_changes))
    seq_len = sample.get("seq_length", 0)

    if sample.get("other_read_issues"):
        return {"sid": sid, "status": "wrong", "reason": "多读段冲突", "rule": 1, "details": sample["other_read_issues"]}
    if identity < t["seq_failure_identity"] or seq_len < t["seq_failure_min_length"]:
        return {"sid": sid, "status": "wrong", "reason": "测序失败", "rule": 2}
    if identity < t["identity_medium_low"] and aa_n > t["aa_overlap_severe"]:
        return {"sid": sid, "status": "wrong", "reason": "重叠峰，比对失败", "rule": 3}
    if identity < t["identity_medium_low"] and t["aa_overlap_moderate_min"] <= aa_n <= t["aa_overlap_moderate_max"]:
        return {"sid": sid, "status": "wrong", "reason": "重叠峰", "rule": 4}
    if frameshift:
        return {"sid": sid, "status": "wrong", "reason": "移码错误", "rule": 5}
    if aa_changes and identity >= t["identity_high"] and 1 <= aa_n <= t["aa_mutation_max"]:
        return {"sid": sid, "status": "wrong", "reason": " ".join(aa_changes), "rule": 6}
    if t["cds_coverage_low"] <= cds_cov <= t["cds_coverage_deletion"] and aa_n >= t["aa_deletion_min"]:
        return {"sid": sid, "status": "wrong", "reason": "片段缺失", "rule": 7}
    if t["synthetic_identity_min"] <= identity <= t["synthetic_identity_max"] and aa_n > t["synthetic_aa_min"]:
        return {"sid": sid, "status": "ok", "reason": "生工重叠峰", "rule": 8}
    if cds_cov < t["cds_coverage_low"] and aa_n == 0 and not frameshift:
        return {"sid": sid, "status": "ok", "reason": "未测通", "rule": 9}
    if identity >= t["identity_high"] and aa_n == 0 and not frameshift:
        return {"sid": sid, "status": "ok", "reason": "", "rule": 10}
    return {"sid": sid, "status": "uncertain", "reason": "需人工复核", "rule": -1}

def judge_batch(samples: list[dict], thresholds: dict | None = None) -> list[dict]:
    t = thresholds or load_thresholds()
    return [judge_sample(s, t) for s in samples]
