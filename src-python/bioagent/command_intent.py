from __future__ import annotations

import re
from typing import Any

CONFIRMATION_ACTIONS = {"run_analysis", "export_report", "open_export_folder", "import_dataset"}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def interpret_command(text: str) -> dict[str, Any]:
    raw_text = str(text or "").strip()
    normalized = _normalize(raw_text)
    actions: list[dict[str, Any]] = []

    if "pet15b" in normalized:
        actions.append({"id": "set_plasmid", "args": {"plasmid": "pet15b"}})
    elif "pet22b" in normalized:
        actions.append({"id": "set_plasmid", "args": {"plasmid": "pet22b"}})

    if ("导入" in raw_text or "导入" in normalized) and ("数据集" in raw_text or "数据集" in normalized):
        actions.append({"id": "import_dataset", "args": {}})

    if "分析" in raw_text or "分析" in normalized:
        actions.append({"id": "run_analysis", "args": {}})

    if "wrong" in normalized or "错误" in raw_text or "异常" in raw_text:
        actions.append({"id": "filter_results", "args": {"status": "wrong"}})
    elif "ok" in normalized or "正常" in raw_text or "通过" in raw_text:
        actions.append({"id": "filter_results", "args": {"status": "ok"}})

    if ("导出" in raw_text or "导出" in normalized) and ("报告" in raw_text or "报告" in normalized):
        actions.append({"id": "export_report", "args": {}})

    if ("打开" in raw_text or "打开" in normalized) and ("导出" in raw_text or "导出" in normalized) and (
        "目录" in raw_text or "目录" in normalized
    ):
        actions.append({"id": "open_export_folder", "args": {}})

    summary = " -> ".join(action["id"] for action in actions) or "reply_only"
    return {
        "summary": summary,
        "actions": actions,
        "needsConfirmation": any(action["id"] in CONFIRMATION_ACTIONS for action in actions),
    }
