"""Unit tests for bio_primitives wrappers used by the MCP tool layer."""
from bioagent.bio_primitives import translate_sequence


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
