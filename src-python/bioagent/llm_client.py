#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bioagent/llm_client.py
LLM client for Sanger sequencing QC judgment.
"""

import os
import re
import time
from typing import Dict, List, Optional

from openai import OpenAI


DEFAULT_BASE_URL = "https://models.sjtu.edu.cn/api/v1"
DEFAULT_MODEL = "deepseek-chat"

SYSTEM_PROMPT = """\
你是一个专业的Sanger测序质控分析员。根据以下比对证据，为每个样本输出判读结论。

输出格式（每样本一行，严格遵守）：
<样本ID> gene is ok/wrong [AA变异列表] [中文错误原因]

判读规则（作为参考，请综合判断）：
1. identity≈1.0 且 cds_coverage>0.55 且无移码 且无AA变化 → gene is ok
2. 有AA变异（如S334L）且identity>0.95 且变异数量少(1-5) → gene is wrong + 列出所有AA变异（空格分隔）
3. 有移码(frameshift=True) 且 cds_coverage>0.55 → gene is wrong 移码错误
4. cds_coverage<0.55 且无AA变异 且无移码 → gene is ok 未测通（测序长度不足以覆盖完整CDS）
5. cds_coverage<0.55 且有移码(frameshift=True) → gene is wrong 移码错误（即使覆盖度低，frameshift已被生物信息学确认为CDS区域内非3倍数indel）
6. identity<0.85 且 aa_changes_n>40 → gene is wrong 重叠峰，比对失败
7. identity<0.85 且 aa_changes_n在25-40之间 → gene is wrong 重叠峰
8. cds_coverage在0.55-0.80之间 且 aa_changes_n>=5 且变异集中在某一连续区域 → gene is wrong 片段缺失（AA变异集中在一段连续位置说明该区域序列有结构性问题）
9. identity<0.30 或 seq_length<50 → gene is wrong 测序失败
10. identity在0.85-0.95之间 且 aa_changes_n>15 且变异分散在整个CDS → gene is ok 生工重叠峰（测序峰图混乱但非真实突变，注意区分：变异分散=重叠峰，变异集中=片段缺失）

区分「重叠峰」与「AA变异」的关键：
- identity<0.85 + 大量AA变异(>20) → 重叠峰（测序质量差导致）
- identity≈1.0（>0.95）+ 少量AA变异(1-5) → 真实AA变异

多读段(dual_read)判断（优先级高，必须先检查）：
- 如果样本有dual_read=True且other_read_issues不为空（其他读段发现了AA变异或移码），则该样本 gene is wrong
- 即使主读段identity=1.0且无AA变异，只要other_read_issues显示其他读段存在问题，就应判定为wrong
- total_cds_coverage表示所有读段合并后的覆盖率，如果显著高于单条读段说明多条读段覆盖了不同区域

重要：
- 严格每样本一行输出
- 不要输出任何额外解释或表头
- AA变异直接写如 S334L K171M，用空格分隔
- 中文原因写在最后
- 如果gene is ok但有注释（如未测通），注释写在ok后面
"""


def normalize_llm_base_url(value: Optional[str]) -> str:
    raw = (value or "").strip()
    if not raw:
        return DEFAULT_BASE_URL

    normalized = raw.rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    return normalized or DEFAULT_BASE_URL


def call_llm(evidence_text: str,
             model: str = DEFAULT_MODEL,
             temperature: float = 0.0,
             max_tokens: int = 4096,
             system_prompt: Optional[str] = None) -> str:
    """Call LLM API with evidence text. Returns raw response text."""
    api_key = os.environ.get("LLM_API_KEY")
    base_url = normalize_llm_base_url(os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL))

    if not api_key or api_key == "your-api-key-here":
        raise RuntimeError(
            "LLM_API_KEY not configured.\n"
            "Please set your API key in the Settings tab."
        )

    client = OpenAI(api_key=api_key, base_url=base_url)

    messages = [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
        {"role": "user", "content": evidence_text},
    ]

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            result = response.choices[0].message.content
            if result is None:
                raise RuntimeError("LLM returned empty content")
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                raise


def parse_llm_result(raw_text: str) -> List[str]:
    """Parse LLM response into individual result lines."""
    lines = []
    for line in raw_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^C\d+", line):
            lines.append(line)
    return lines


def parse_llm_result_map(raw_text: str) -> Dict[str, dict]:
    result_map: Dict[str, dict] = {}
    for line in parse_llm_result(raw_text):
        match = re.match(r"^(?P<sid>\S+)\s+gene\s+is\s+(?P<status>ok|wrong)(?:\s+(?P<reason>.*))?$", line, re.IGNORECASE)
        if not match:
            continue
        sid = match.group("sid").strip().upper()
        result_map[sid] = {
            "status": match.group("status").lower(),
            "reason": (match.group("reason") or "").strip(),
            "line": line.strip(),
        }
    return result_map
