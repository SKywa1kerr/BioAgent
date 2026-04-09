"""Deterministic rules engine for Sanger sequencing QC judgment."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import yaml

DEFAULT_CONFIG = Path(__file__).parent / "rules_config.yaml"

_threshold_cache: Optional[dict] = None
_threshold_mtime: float = 0


def load_thresholds(config_path: Path = DEFAULT_CONFIG) -> dict:
    global _threshold_cache, _threshold_mtime
    try:
        mtime = config_path.stat().st_mtime
    except OSError:
        mtime = 0
    if _threshold_cache is not None and mtime == _threshold_mtime:
        return dict(_threshold_cache)
    with open(config_path, encoding="utf-8") as f:
        _threshold_cache = yaml.safe_load(f)["thresholds"]
        _threshold_mtime = mtime
    return dict(_threshold_cache)


def judge_sample(sample: dict, thresholds: Optional[dict] = None) -> dict:
    """Apply deterministic rules to a sample dict. Returns {sid, status, reason, rule}."""
    t = thresholds or load_thresholds()
    sid = sample.get("sample_id", sample.get("sid", ""))
    identity = sample.get("identity", 0)
    cds_cov = sample.get("coverage", sample.get("cds_coverage", 0))
    frameshift = sample.get("frameshift", False)
    aa_changes = sample.get("aa_changes", [])
    aa_n = sample.get("aa_changes_n", len(aa_changes))
    seq_len = sample.get("seq_length", 0)
    high_identity = identity >= t["identity_high"]
    no_cds_findings = (not frameshift) and aa_n == 0

    if sample.get("other_read_issues"):
        return {"sid": sid, "status": "wrong", "reason": "多读段冲突", "rule": 1}
    if identity < t["seq_failure_identity"] or (seq_len < t["seq_failure_min_length"] and not high_identity):
        return {"sid": sid, "status": "wrong", "reason": "测序失败", "rule": 2}
    if high_identity and no_cds_findings and seq_len < t["short_synthetic_seq_max_length"] and cds_cov <= t["short_synthetic_coverage_max"]:
        return {"sid": sid, "status": "ok", "reason": "生工重叠峰", "rule": 11}
    if (
        high_identity
        and no_cds_findings
        and t["short_synthetic_seq_max_length"] <= seq_len < t["short_overlap_seq_max_length"]
        and cds_cov <= t["short_overlap_coverage_max"]
    ):
        return {"sid": sid, "status": "wrong", "reason": "重叠峰", "rule": 12}
    if (
        high_identity
        and no_cds_findings
        and t["short_overlap_seq_max_length"] <= seq_len < t["short_fragment_seq_max_length"]
        and cds_cov <= t["short_fragment_coverage_max"]
    ):
        return {"sid": sid, "status": "wrong", "reason": "片段缺失", "rule": 13}
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
        ins = sample.get("ins", 0)
        dels = sample.get("dele", sample.get("del", 0))
        note = f"非编码区 ins={ins} del={dels}" if (ins or dels) else ""
        return {"sid": sid, "status": "ok", "reason": note, "rule": 10}
    return {"sid": sid, "status": "uncertain", "reason": "需人工复核", "rule": -1}


def judge_batch(samples: list, thresholds: Optional[dict] = None) -> list:
    t = thresholds or load_thresholds()
    return [judge_sample(s, t) for s in samples]
