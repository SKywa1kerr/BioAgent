from core.alignment import (
    is_edge_ignored,
    label_synonymous_mutations,
    apply_dual_read_consensus,
    decide_bucket,
)


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


def test_single_read_mutation_demoted_when_other_read_covers_with_ref_base():
    best = {
        "sid": "C123-1",
        "mutations": [
            {"position": 200, "refBase": "A", "queryBase": "T",
             "type": "substitution", "effect": ""},
        ],
        "_cds_positions": {200: "T", 201: "G", 202: "C"},
    }
    other = {
        "sid": "C123-1",
        "mutations": [],
        "_cds_positions": {200: "A", 199: "C"},  # same position, ref base
    }
    merged = apply_dual_read_consensus(best, [other])
    assert merged["mutations"][0]["effect"] == "single_read"


def test_consensus_mutation_keeps_effect_unchanged():
    best = {
        "sid": "C123-1",
        "mutations": [
            {"position": 200, "refBase": "A", "queryBase": "T",
             "type": "substitution", "effect": ""},
        ],
        "_cds_positions": {200: "T"},
    }
    other = {
        "sid": "C123-1",
        "mutations": [
            {"position": 200, "refBase": "A", "queryBase": "T",
             "type": "substitution", "effect": ""},
        ],
        "_cds_positions": {200: "T"},
    }
    merged = apply_dual_read_consensus(best, [other])
    assert merged["mutations"][0]["effect"] == ""


def test_mutation_uncovered_by_other_read_not_demoted():
    best = {
        "sid": "C123-1",
        "mutations": [
            {"position": 500, "refBase": "A", "queryBase": "T",
             "type": "substitution", "effect": ""},
        ],
        "_cds_positions": {500: "T"},
    }
    other = {
        "sid": "C123-1",
        "mutations": [],
        "_cds_positions": {},  # other read never covers pos 500
    }
    merged = apply_dual_read_consensus(best, [other])
    assert merged["mutations"][0]["effect"] == ""


def test_bucket_untested_when_coverage_below_threshold():
    assert decide_bucket(
        cds_coverage=0.3, avg_qry_quality=30.0, frameshift=False,
        aa_changes=[], mutations=[], has_single_read=False,
    ) == "untested"


def test_bucket_untested_when_avg_quality_very_low():
    assert decide_bucket(
        cds_coverage=0.9, avg_qry_quality=10.0, frameshift=False,
        aa_changes=[], mutations=[], has_single_read=False,
    ) == "untested"


def test_bucket_wrong_when_aa_changes_present():
    assert decide_bucket(
        cds_coverage=0.95, avg_qry_quality=35.0, frameshift=False,
        aa_changes=["K171M"], mutations=[], has_single_read=False,
    ) == "wrong"


def test_bucket_wrong_when_frameshift():
    assert decide_bucket(
        cds_coverage=0.95, avg_qry_quality=35.0, frameshift=True,
        aa_changes=[], mutations=[], has_single_read=False,
    ) == "wrong"


def test_bucket_uncertain_when_single_read_mutations_present():
    assert decide_bucket(
        cds_coverage=0.95, avg_qry_quality=35.0, frameshift=False,
        aa_changes=[], mutations=[], has_single_read=True,
    ) == "uncertain"


def test_bucket_uncertain_for_mid_coverage():
    assert decide_bucket(
        cds_coverage=0.6, avg_qry_quality=35.0, frameshift=False,
        aa_changes=[], mutations=[], has_single_read=False,
    ) == "uncertain"


def test_bucket_ok_for_clean_sample():
    assert decide_bucket(
        cds_coverage=0.95, avg_qry_quality=35.0, frameshift=False,
        aa_changes=[], mutations=[], has_single_read=False,
    ) == "ok"
