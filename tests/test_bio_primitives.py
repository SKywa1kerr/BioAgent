"""Unit tests for bio_primitives wrappers used by the MCP tool layer."""
from bioagent.bio_primitives import inspect_path, read_sequence_file, translate_sequence


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
