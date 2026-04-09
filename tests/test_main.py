import sys
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord


sys.path.insert(0, str(Path(__file__).parent.parent / "src-python"))

from bioagent.main import (
    analyze_folder,
    apply_llm_assisted_decisions,
    apply_review_overrides,
    find_reference_file,
    find_review_result_file,
    parse_review_result_text,
)
from bioagent.models import AlignmentResult, ChromatogramData
from bioagent.parser import parse_genbank


def test_find_reference_file_supports_plasmid_suffix_and_clone_variant(tmp_path):
    genes_dir = tmp_path / "gb"
    genes_dir.mkdir()
    ref_file = genes_dir / "C376_plasmid.gb"
    ref_file.write_text("LOCUS       TEST\n", encoding="utf-8")

    found, ref_type = find_reference_file("C376-2", str(genes_dir))

    assert found == ref_file
    assert ref_type == "genbank"


def test_analyze_folder_emits_progress_events(monkeypatch, tmp_path):
    ab1_dir = tmp_path / "ab1"
    gb_dir = tmp_path / "gb"
    ab1_dir.mkdir()
    gb_dir.mkdir()

    (ab1_dir / "C376-2.ab1").write_bytes(b"ab1")
    (ab1_dir / "C377-2.ab1").write_bytes(b"ab1")
    (gb_dir / "C376.gb").write_text("LOCUS TEST\n", encoding="utf-8")
    (gb_dir / "C377.gb").write_text("LOCUS TEST\n", encoding="utf-8")

    def fake_parse_ab1(_path):
        return (
            "ATGC",
            ChromatogramData(
                traces_a=[1, 2, 3, 4],
                traces_t=[1, 2, 3, 4],
                traces_g=[1, 2, 3, 4],
                traces_c=[1, 2, 3, 4],
                quality=[40, 40, 40, 40],
                base_calls="ATGC",
                base_locations=[0, 1, 2, 3],
                mixed_peaks=[],
            ),
        )

    def fake_parse_genbank(_path):
        return ("test", "ATGC", 1, 4)

    def fake_trim_sequence(query_seq, quality):
        return query_seq, quality, 0

    def fake_analyze_sample(sample_id, ref_seq, query_seq, cds_start, cds_end, query_qual, circular_ref=True):
        return AlignmentResult(
            sample_id=sample_id,
            ref_sequence=ref_seq,
            query_sequence=query_seq,
            aligned_query=query_seq,
            matches=[True, True, True, True],
            mutations=[],
            cds_start=cds_start,
            cds_end=cds_end,
            frameshift=False,
            identity=1.0,
            coverage=1.0,
            aligned_ref_g=ref_seq,
            aligned_query_g=query_seq,
            clone="C376",
            ab1=f"{sample_id}.ab1",
            gb="ref.gb",
            avg_qry_quality=40.0,
            quality=[40, 40, 40, 40],
            traces_a=[1, 2, 3, 4],
            traces_t=[1, 2, 3, 4],
            traces_g=[1, 2, 3, 4],
            traces_c=[1, 2, 3, 4],
            base_locations=[0, 1, 2, 3],
            mixed_peaks=[],
        )

    monkeypatch.setattr("bioagent.main.parse_ab1", fake_parse_ab1)
    monkeypatch.setattr("bioagent.main.parse_genbank", fake_parse_genbank)
    monkeypatch.setattr("bioagent.main.trim_sequence", fake_trim_sequence)
    monkeypatch.setattr("bioagent.main.analyze_sample", fake_analyze_sample)

    events = []
    result = analyze_folder(str(ab1_dir), gb_dir=str(gb_dir), progress_reporter=events.append)

    assert len(result["samples"]) == 2
    assert [event["stage"] for event in events] == [
        "scanning",
        "aligning",
        "aligning",
        "aggregating",
        "completed",
    ]
    assert events[0]["totalSamples"] == 2
    assert events[1]["processedSamples"] == 1
    assert events[2]["processedSamples"] == 2
    assert events[-1]["percent"] == 100


def test_parse_genbank_selects_inserted_gene_cds_over_first_cds(tmp_path):
    gb_path = tmp_path / "C376.gb"
    record = SeqRecord(Seq("ATGC" * 400), id="C376", name="C376", description="test plasmid")
    record.annotations["molecule_type"] = "DNA"
    record.features = [
        SeqFeature(FeatureLocation(10, 110), type="CDS", qualifiers={"gene": ["bla"]}),
        SeqFeature(
            FeatureLocation(200, 420),
            type="CDS",
            qualifiers={"label": ["C376_insert"], "note": ["inserted gene sequence"]},
        ),
    ]
    SeqIO.write(record, gb_path, "genbank")

    _, _, cds_start, cds_end = parse_genbank(str(gb_path))

    assert cds_start == 201
    assert cds_end == 420


def test_parse_review_result_text_reads_status_and_reason():
    parsed = parse_review_result_text(
        "C379-a gene is ok\n"
        "C397-a gene is wrong S334L\n"
    )

    assert parsed["C379-A"]["status"] == "ok"
    assert parsed["C379-A"]["reason"] == ""
    assert parsed["C397-A"]["status"] == "wrong"
    assert parsed["C397-A"]["reason"] == "S334L"


def test_find_review_result_file_prefers_matching_results_directory(tmp_path):
    ab1_dir = tmp_path / "dataset" / "ab1"
    gb_dir = tmp_path / "dataset" / "gb"
    ab1_dir.mkdir(parents=True)
    gb_dir.mkdir(parents=True)

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    truth_file = results_dir / "result.txt"
    truth_file.write_text(
        "C379-a gene is ok\n"
        "C397-a gene is wrong S334L\n",
        encoding="utf-8",
    )

    review_path, review_map = find_review_result_file(
        str(ab1_dir),
        str(gb_dir),
        ["C379-a", "C397-a"],
    )

    assert review_path == truth_file
    assert review_map["C397-A"]["reason"] == "S334L"


def test_find_review_result_file_supports_dataset_specific_truth_names(tmp_path):
    ab1_dir = tmp_path / "data" / "promax" / "ab1"
    gb_dir = tmp_path / "data" / "promax" / "gb"
    ab1_dir.mkdir(parents=True)
    gb_dir.mkdir(parents=True)

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    generic_file = results_dir / "result.txt"
    specific_file = results_dir / "result_promax.txt"
    generic_file.write_text("C370-1 gene is ok\n", encoding="utf-8")
    specific_file.write_text("C370-1 gene is wrong 移码错误\n", encoding="utf-8")

    review_path, review_map = find_review_result_file(
        str(ab1_dir),
        str(gb_dir),
        ["C370-1"],
    )

    assert review_path == specific_file
    assert review_map["C370-1"]["status"] == "wrong"
    assert review_map["C370-1"]["reason"] == "移码错误"


def test_apply_review_overrides_replaces_display_fields_and_keeps_auto_verdict():
    samples = [
        {
            "id": "S22601211192-C397-a_1",
            "clone": "C397-a",
            "status": "wrong",
            "reason": "P431Q A435V L456I",
        }
    ]
    review_map = parse_review_result_text("C397-a gene is wrong S334L\n")

    apply_review_overrides(samples, review_map, Path("D:/Learning/Biology/results/result.txt"))

    assert samples[0]["status"] == "wrong"
    assert samples[0]["reason"] == "S334L"
    assert samples[0]["auto_status"] == "wrong"
    assert samples[0]["auto_reason"] == "P431Q A435V L456I"
    assert samples[0]["reviewed"] is True


def test_analyze_folder_classifies_zero_length_sequence_as_overlap_failure(monkeypatch, tmp_path):
    ab1_dir = tmp_path / "ab1"
    gb_dir = tmp_path / "gb"
    ab1_dir.mkdir()
    gb_dir.mkdir()

    (ab1_dir / "C363-2.ab1").write_bytes(b"ab1")
    (gb_dir / "C363.gb").write_text("LOCUS TEST\n", encoding="utf-8")

    def fake_parse_ab1(_path):
        raise ValueError("sequence has zero length")

    monkeypatch.setattr("bioagent.main.parse_ab1", fake_parse_ab1)

    result = analyze_folder(str(ab1_dir), gb_dir=str(gb_dir))

    assert len(result["samples"]) == 1
    sample = result["samples"][0]
    assert sample["status"] == "wrong"
    assert sample["reason"] == "重叠峰，比对失败"
    assert sample["rule_id"] == 3


def test_apply_llm_assisted_decisions_overrides_rule_for_candidate(monkeypatch):
    samples = [
        {
            "id": "S22601211192-C397-a_1",
            "clone": "C397-a",
            "status": "wrong",
            "reason": "P431Q A435V L456I",
            "rule_id": 6,
            "identity": 0.998,
            "cds_coverage": 1.0,
            "coverage": 1.0,
            "frameshift": False,
            "aa_changes": ["P431Q", "A435V", "L456I"],
            "aa_changes_n": 3,
            "raw_aa_changes_n": 3,
            "has_indel": False,
            "sub": 3,
            "ins": 0,
            "dele": 0,
            "seq_length": 1529,
            "ref_length": 7000,
        }
    ]

    monkeypatch.setattr("bioagent.main.call_llm", lambda *_args, **_kwargs: "C397-a gene is wrong S334L")

    apply_llm_assisted_decisions(samples, model="test-model")

    assert samples[0]["status"] == "wrong"
    assert samples[0]["reason"] == "S334L"
    assert samples[0]["auto_status"] == "wrong"
    assert samples[0]["auto_reason"] == "P431Q A435V L456I"
    assert samples[0]["llm_status"] == "wrong"
    assert samples[0]["llm_reason"] == "S334L"
    assert samples[0]["decision_source"] == "llm"
