from core.alignment import is_edge_ignored


def test_substitution_within_edge_of_cds_end_is_ignored():
    # cds_end=6614, EDGE_IGNORE_BP=20 default -> pos 6600 ignored, 6593 not
    assert is_edge_ignored(pos=6600, cds_start=100, cds_end=6614) is True
    assert is_edge_ignored(pos=6593, cds_start=100, cds_end=6614) is False


def test_substitution_within_edge_of_cds_start_is_ignored():
    assert is_edge_ignored(pos=105, cds_start=100, cds_end=6614) is True
    assert is_edge_ignored(pos=125, cds_start=100, cds_end=6614) is False


def test_none_cds_bounds_never_ignored():
    assert is_edge_ignored(pos=100, cds_start=None, cds_end=None) is False
