"""End-to-end smoke test for the 6 v0.3.0 primitive tools.

Spawns the real sidecar (python -m bioagent.main --mcp-server), sends
JSON-RPC requests for each primitive over stdin, prints results.

This is what electron/agent_harness.mjs talks to in production, so a
pass here proves the protocol-level integration works end-to-end —
schema, dispatch, return shape, the whole pipeline. The only thing not
exercised is the LLM deciding to call these (which is a Task-9
user-driven manual test).
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def rpc(method, params=None, _id=1):
    return json.dumps({"jsonrpc": "2.0", "id": _id, "method": method,
                       "params": params or {}}) + "\n"


def main():
    repo = Path(__file__).resolve().parents[1]

    # Build a tiny on-disk fixture for read_sequence_file
    fixture_dir = Path(tempfile.mkdtemp(prefix="bioagent_smoke_"))
    (fixture_dir / "tiny.fa").write_text(">seq1\nATGGAATAA\n")

    requests = [
        rpc("tools/list", _id=1),
        rpc("tools/call", {"name": "translate_sequence",
                           "arguments": {"dna": "ATGGAATAA"}}, _id=2),
        rpc("tools/call", {"name": "inspect_path",
                           "arguments": {"paths": str(fixture_dir / "tiny.fa")}}, _id=3),
        rpc("tools/call", {"name": "read_sequence_file",
                           "arguments": {"path": str(fixture_dir / "tiny.fa")}}, _id=4),
        rpc("tools/call", {"name": "compare_sequences",
                           "arguments": {"ref_seq": "ATGGAATAA",
                                         "query_seq": "ATGAAATAA",
                                         "ref_cds_start": 1,
                                         "ref_cds_end": 9}}, _id=5),
        rpc("tools/call", {"name": "analyze_files_adhoc",
                           "arguments": {"ab1_dir": "/missing",
                                         "gb_dir": "/missing"}}, _id=6),
        rpc("tools/call", {"name": "export_analysis_html",
                           "arguments": {"analysis_id": "nope",
                                         "sample_id": "nope"}}, _id=7),
    ]

    proc = subprocess.Popen(
        [sys.executable, "-m", "bioagent.main", "--mcp-server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=str(repo / "src-python"),
        text=True, bufsize=1,
        env={**__import__("os").environ, "PYTHONPATH": str(repo / "src-python")},
    )

    try:
        stdout, stderr = proc.communicate(input="".join(requests), timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        print("[FAIL] sidecar timed out")
        return 1

    print(f"--- stderr ---\n{stderr[:1000]}\n")
    print(f"--- stdout (response lines) ---")

    expected_ids = {1, 2, 3, 4, 5, 6, 7}
    seen_ids = set()
    failures = []
    primitive_names = {"inspect_path", "read_sequence_file", "compare_sequences",
                       "translate_sequence", "analyze_files_adhoc",
                       "export_analysis_html"}

    for line in stdout.strip().splitlines():
        try:
            resp = json.loads(line)
        except json.JSONDecodeError:
            print(f"  [skip non-json] {line[:120]}")
            continue
        rid = resp.get("id")
        seen_ids.add(rid)

        if rid == 1:
            # tools/list — check all 6 new primitives are present
            tools = resp.get("result", {}).get("tools", [])
            names = {t["name"] for t in tools}
            missing = primitive_names - names
            if missing:
                failures.append(f"tools/list missing: {missing}")
                print(f"  [id=1 tools/list] FAIL — missing {missing}")
            else:
                print(f"  [id=1 tools/list] OK — {len(tools)} tools total, all 6 primitives present")
        else:
            # This sidecar returns the tool dict directly as `result`,
            # not wrapped in the MCP content/text envelope.
            result = resp.get("result", {})
            if not isinstance(result, dict):
                result = {}
            ok = result.get("ok")
            short = json.dumps(result)[:120]
            print(f"  [id={rid}] ok={ok}  {short}")
            if rid in (2, 3, 4, 5) and ok is not True:
                failures.append(f"id={rid} expected ok=True, got {ok}")
            if rid in (6, 7) and ok is not False:
                failures.append(f"id={rid} expected ok=False (graceful error), got {ok}")

    missing_ids = expected_ids - seen_ids
    if missing_ids:
        failures.append(f"no response received for ids: {missing_ids}")

    print(f"\n--- summary ---")
    if failures:
        print(f"FAIL: {len(failures)} issues")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS: all 6 primitives responded correctly through MCP protocol")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
