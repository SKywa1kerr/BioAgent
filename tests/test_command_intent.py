import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src-python"))

from bioagent.command_intent import interpret_command


def test_interpret_command_parses_analysis_request_with_plasmid_and_wrong_filter():
    plan = interpret_command("分析这个数据集，用 pet15b，只看 wrong 样本")

    assert plan["summary"] == "set_plasmid -> run_analysis -> filter_results"
    assert plan["actions"] == [
        {"id": "set_plasmid", "args": {"plasmid": "pet15b"}},
        {"id": "run_analysis", "args": {}},
        {"id": "filter_results", "args": {"status": "wrong"}},
    ]
    assert plan["needsConfirmation"] is True
    assert json.loads(json.dumps(plan, ensure_ascii=False)) == plan


def test_interpret_command_parses_export_and_open_folder_request():
    plan = interpret_command("导出当前报告并打开导出目录")

    assert plan["summary"] == "export_report -> open_export_folder"
    assert plan["actions"] == [
        {"id": "export_report", "args": {}},
        {"id": "open_export_folder", "args": {}},
    ]
    assert plan["needsConfirmation"] is True


def test_interpret_command_parses_import_and_analysis_request():
    plan = interpret_command("导入新的数据集并开始分析")

    assert plan["summary"] == "import_dataset -> run_analysis"
    assert plan["actions"] == [
        {"id": "import_dataset", "args": {}},
        {"id": "run_analysis", "args": {}},
    ]
    assert plan["needsConfirmation"] is True


def test_interpret_command_cli_handles_empty_string():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "bioagent.main",
            "--interpret-command",
            "",
        ],
        cwd=Path(__file__).parent.parent / "src-python",
        capture_output=True,
        text=True,
        check=True,
    )

    output = json.loads(result.stdout.strip())
    assert output["summary"] == "reply_only"
    assert output["actions"] == []
    assert output["needsConfirmation"] is False
