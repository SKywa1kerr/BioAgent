import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / "src-python"))

from bioagent.alignment import extract_mutations, ref2pos_to_refpos


def test_ref2pos_to_refpos_does_not_wrap_for_linear_reference():
    assert ref2pos_to_refpos(7, 4, is_circular=False) == 8


def test_extract_mutations_treats_base_case_as_match():
    mutations = extract_mutations("AaT", "AAT", ref2_start=0, ref_len=3)

    assert mutations == []
