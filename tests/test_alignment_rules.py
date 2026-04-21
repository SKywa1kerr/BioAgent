from core.alignment import is_edge_ignored, label_synonymous_mutations


def test_substitution_within_edge_of_cds_end_is_ignored():
    # cds_end=6614, EDGE_IGNORE_BP=20 default -> pos 6600 ignored, 6593 not
    assert is_edge_ignored(pos=6600, cds_start=100, cds_end=6614) is True
    assert is_edge_ignored(pos=6593, cds_start=100, cds_end=6614) is False


def test_substitution_within_edge_of_cds_start_is_ignored():
    assert is_edge_ignored(pos=105, cds_start=100, cds_end=6614) is True
    assert is_edge_ignored(pos=125, cds_start=100, cds_end=6614) is False


def test_none_cds_bounds_never_ignored():
    assert is_edge_ignored(pos=100, cds_start=None, cds_end=None) is False


def test_synonymous_substitution_is_labeled():
    # GAA (Glu) at codon 4-6 -> GAG (Glu) = synonymous (pos 6: A -> G)
    ref_seq = "ATGGAATAA"  # M-E-*
    mutations = [
        {"type": "substitution", "position": 6, "refBase": "A", "queryBase": "G",
         "effect": ""},
    ]
    result = label_synonymous_mutations(
        mutations, ref_seq=ref_seq, cds_start=1, cds_end=9,
        cds_positions={4: "G", 5: "A", 6: "G", 7: "T", 8: "A", 9: "A"},
    )
    assert result[0]["effect"] == "synonymous"


def test_missense_substitution_is_not_labeled():
    # GAA (Glu) -> GTA (Val) at codon pos 5: A -> T
    ref_seq = "ATGGAATAA"
    mutations = [
        {"type": "substitution", "position": 5, "refBase": "A", "queryBase": "T",
         "effect": ""},
    ]
    result = label_synonymous_mutations(
        mutations, ref_seq=ref_seq, cds_start=1, cds_end=9,
        cds_positions={4: "G", 5: "T", 6: "A", 7: "T", 8: "A", 9: "A"},
    )
    assert result[0]["effect"] != "synonymous"


def test_indel_mutations_passthrough_unchanged():
    mutations = [
        {"type": "insertion", "position": 5, "refBase": "-", "queryBase": "A",
         "effect": ""},
        {"type": "deletion", "position": 7, "refBase": "T", "queryBase": "-",
         "effect": ""},
    ]
    result = label_synonymous_mutations(
        mutations, ref_seq="ATGGAATAA", cds_start=1, cds_end=9, cds_positions={},
    )
    assert all(m["effect"] == "" for m in result)
