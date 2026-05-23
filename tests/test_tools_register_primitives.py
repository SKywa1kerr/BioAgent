"""Integration tests verifying the 6 primitive tools are registered and
dispatchable via the standard mcp_tools.call_tool() interface."""
import bioagent.mcp_tools as mcp_tools
import bioagent.tools_register as tools_register


def setup_function(_fn):
    # conftest already resets the registry between tests; just rebuild.
    tools_register.register_initial_tools()


def test_inspect_path_is_callable(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    result = mcp_tools.call_tool("inspect_path", {"paths": str(tmp_path / "a.txt")})
    assert result["ok"] is True
    assert result["data"]["ok"] is True


def test_translate_sequence_is_callable():
    result = mcp_tools.call_tool("translate_sequence", {"dna": "ATGGAATAA"})
    assert result["ok"] is True
    assert result["data"]["protein"] == "ME*"


def test_compare_sequences_is_callable():
    result = mcp_tools.call_tool("compare_sequences", {
        "ref_seq": "ATGGAATAA",
        "query_seq": "ATGGGATAA",
    })
    assert result["ok"] is True
    assert result["data"]["ok"] is True


def test_read_sequence_file_is_callable(tmp_path):
    fa = tmp_path / "x.fa"
    fa.write_text(">s\nATGC\n")
    result = mcp_tools.call_tool("read_sequence_file", {"path": str(fa)})
    assert result["ok"] is True
    assert result["data"]["format"] == "fasta"


def test_analyze_files_adhoc_is_callable():
    result = mcp_tools.call_tool("analyze_files_adhoc", {
        "ab1_dir": "/missing",
        "gb_dir": "/missing",
    })
    # call dispatched successfully (ok=True at the call_tool envelope)
    assert result["ok"] is True
    # but the tool itself returned an error envelope
    assert result["data"]["ok"] is False


def test_export_analysis_html_is_callable():
    result = mcp_tools.call_tool("export_analysis_html", {
        "analysis_id": "nope",
        "sample_id": "nope",
    })
    assert result["ok"] is True
    assert result["data"]["ok"] is False


def test_list_tools_includes_all_six_primitives():
    names = {t["name"] for t in mcp_tools.list_tools()}
    expected = {
        "inspect_path", "read_sequence_file", "compare_sequences",
        "translate_sequence", "analyze_files_adhoc", "export_analysis_html",
    }
    assert expected.issubset(names), f"missing: {expected - names}"
