"""Unit tests for bio_primitives wrappers used by the MCP tool layer."""
from bioagent.bio_primitives import (
    analyze_files_adhoc,
    compare_sequences,
    export_analysis_html,
    inspect_path,
    read_sequence_file,
    translate_sequence,
)


def test_translate_sequence_full_orf():
    result = translate_sequence(dna="ATGGAATAA")  # M-E-*
    assert result["ok"] is True
    assert result["protein"] == "ME*"
    assert result["stop_codons"] == [3]  # stop at amino acid position 3 (1-based)


def test_translate_sequence_handles_frame_shift_arg():
    result = translate_sequence(dna="GATGGAATAA", frame=1)  # frame 1: ATG GAA TAA
    assert result["ok"] is True
    assert result["protein"] == "ME*"


def test_translate_sequence_rejects_empty():
    result = translate_sequence(dna="")
    assert result["ok"] is False
    assert "empty" in result["error"].lower()


def test_translate_sequence_handles_ambiguous_codon():
    # 'NNN' should produce 'X' (unknown), not crash
    result = translate_sequence(dna="NNNATG")
    assert result["ok"] is True
    assert result["protein"].startswith("X")


def test_inspect_path_handles_single_string(tmp_path):
    f = tmp_path / "sample.ab1"
    f.write_bytes(b"\x00\x00")

    result = inspect_path(paths=str(f))
    assert result["ok"] is True
    assert len(result["items"]) == 1
    item = result["items"][0]
    assert item["type"] == "file"
    assert item["ext"] == ".ab1"
    assert item["size"] == 2


def test_inspect_path_detects_dataset_layout_subdirs(tmp_path):
    ab1 = tmp_path / "ab1"
    gb = tmp_path / "gb"
    ab1.mkdir()
    gb.mkdir()
    (ab1 / "a.ab1").write_bytes(b"")
    (gb / "a.gb").write_text("LOCUS x")

    result = inspect_path(paths=str(tmp_path))
    assert result["ok"] is True
    assert result["items"][0]["dataset_hint"] == "subdirs"


def test_inspect_path_marks_missing(tmp_path):
    result = inspect_path(paths=str(tmp_path / "does_not_exist"))
    assert result["ok"] is True
    assert result["items"][0]["type"] == "missing"


def test_inspect_path_accepts_list(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b.txt").write_text("y")

    result = inspect_path(paths=[str(tmp_path / "a.txt"), str(tmp_path / "b.txt")])
    assert result["ok"] is True
    assert len(result["items"]) == 2


def test_read_sequence_file_genbank(tmp_path):
    gb = tmp_path / "tiny.gb"
    gb.write_text(
        "LOCUS       tiny    9 bp    DNA     linear   UNK 01-JAN-2026\n"
        "FEATURES             Location/Qualifiers\n"
        "     CDS             1..9\n"
        '                     /gene="X"\n'
        "ORIGIN\n"
        "        1 atggaataa\n"
        "//\n"
    )
    result = read_sequence_file(path=str(gb))
    assert result["ok"] is True
    assert result["format"] == "genbank"
    assert result["length"] == 9
    assert result["sequence_preview"].startswith("ATGGAATAA")
    assert result["features"], "expected at least one CDS feature"


def test_read_sequence_file_fasta(tmp_path):
    fa = tmp_path / "x.fa"
    fa.write_text(">seq1\nATGCATGC\n")
    result = read_sequence_file(path=str(fa))
    assert result["ok"] is True
    assert result["format"] == "fasta"
    assert result["length"] == 8


def test_read_sequence_file_missing(tmp_path):
    result = read_sequence_file(path=str(tmp_path / "missing.gb"))
    assert result["ok"] is False
    assert "not found" in result["error"].lower()


def test_read_sequence_file_unknown_extension(tmp_path):
    f = tmp_path / "weird.xyz"
    f.write_text("x")
    result = read_sequence_file(path=str(f))
    assert result["ok"] is False
    assert "unsupported" in result["error"].lower()


def test_compare_sequences_finds_single_substitution():
    # ref:  ATGGAATAA
    # qry:  ATGGGATAA  (position 5: A -> G)
    result = compare_sequences(ref_seq="ATGGAATAA", query_seq="ATGGGATAA")
    assert result["ok"] is True
    assert result["length"] >= 8
    subs = [m for m in result["mutations"] if m.get("type", "").lower() == "substitution"]
    assert subs, f"expected substitution, got {result['mutations']}"
    sub = subs[0]
    assert sub.get("ref_pos") == 5
    assert sub.get("ref_base") == "A"
    assert sub.get("qry_base") == "G"


def test_compare_sequences_with_cds_yields_aa_changes():
    # ref ATGGAATAA encodes M-E-*; query ATGAAATAA changes codon 2 GAA->AAA
    # which is E -> K (a real amino-acid substitution).
    result = compare_sequences(
        ref_seq="ATGGAATAA",
        query_seq="ATGAAATAA",
        ref_cds_start=1,
        ref_cds_end=9,
    )
    assert result["ok"] is True
    assert result["aa_changes"] == ["E2K"]


def test_compare_sequences_rejects_empty():
    result = compare_sequences(ref_seq="", query_seq="ATG")
    assert result["ok"] is False


def test_compare_sequences_identical_inputs_zero_mutations():
    result = compare_sequences(ref_seq="ATGGAATAACGTACGT", query_seq="ATGGAATAACGTACGT")
    assert result["ok"] is True
    assert result["mutations"] == []
    assert result["identity"] == 1.0


def test_analyze_files_adhoc_missing_dirs():
    result = analyze_files_adhoc(ab1_dir="/no/such", gb_dir="/no/such")
    assert result["ok"] is False
    assert "not found" in result["error"].lower() or "not a directory" in result["error"].lower()


def test_analyze_files_adhoc_empty_dirs_returns_zero(tmp_path):
    """Two existing-but-empty dirs should NOT crash; analyze_dirs will
    raise FileNotFoundError because there are no .gb files. Wrapper must
    catch and return an error envelope."""
    ab1 = tmp_path / "ab1"
    gb = tmp_path / "gb"
    ab1.mkdir()
    gb.mkdir()

    result = analyze_files_adhoc(ab1_dir=str(ab1), gb_dir=str(gb))
    assert result["ok"] is False
    assert "no .gb" in result["error"].lower() or "not found" in result["error"].lower()


def test_export_analysis_html_unknown_analysis():
    import bioagent.tools_register  # ensure module loaded so conftest reset hits it
    result = export_analysis_html(analysis_id="nope", sample_id="x")
    assert result["ok"] is False
    assert "analysis" in result["error"].lower()


def test_export_analysis_html_writes_file(tmp_path):
    import bioagent.tools_register as tools_register
    tools_register._ANALYSIS_DETAILS["test_aid"] = {
        "analysis_id": "test_aid",
        "samples": [
            {
                "id": "S1",
                "ref_sequence": "ATGGAATAACGTACGT",
                "query_sequence": "ATGGAATAACGTACGT",
                "ref_length": 16,
                "aligned_ref_g": "ATGGAATAACGTACGT",
                "aligned_query_g": "ATGGAATAACGTACGT",
            },
        ],
    }

    out_path = tmp_path / "alignment.html"
    result = export_analysis_html(
        analysis_id="test_aid",
        sample_id="S1",
        output_path=str(out_path),
    )
    assert result["ok"] is True, result
    assert out_path.exists()
    body = out_path.read_text()
    assert "ATGGAATAA" in body or "REF" in body
