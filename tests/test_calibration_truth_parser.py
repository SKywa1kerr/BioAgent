from scripts.calibrate import parse_truth_line, TruthRecord


def test_parses_ok():
    rec = parse_truth_line("C379-a gene is ok")
    assert rec == TruthRecord(sid="C379-a", status="ok", aa=[], note=None)


def test_parses_wrong_with_aa_list():
    rec = parse_truth_line("C402-2 gene is wrong R171M L334S")
    assert rec.status == "wrong"
    assert rec.aa == ["R171M", "L334S"]


def test_parses_single_aa():
    rec = parse_truth_line("C397-2 gene is wrong K171M")
    assert rec.aa == ["K171M"]


def test_chinese_note_maps_to_uncertain_for_overlap():
    rec = parse_truth_line("C410-1 重叠峰")
    assert rec.status == "uncertain"


def test_chinese_note_maps_to_untested_for_failed():
    rec = parse_truth_line("C410-2 测序失败")
    assert rec.status == "untested"


def test_frameshift_keyword_wrongs_the_call():
    rec = parse_truth_line("C411-3 移码突变")
    assert rec.status == "wrong"


def test_blank_line_returns_none():
    assert parse_truth_line("") is None
    assert parse_truth_line("   ") is None


def test_comment_line_returns_none():
    assert parse_truth_line("# header") is None
