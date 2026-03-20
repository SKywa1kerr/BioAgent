import pytest
from backend.core.rules import judge_sample, load_thresholds

@pytest.fixture
def thresholds():
    return load_thresholds()

class TestIndividualRules:
    def test_rule1_multi_read_conflict(self, thresholds):
        sample = {"sid": "X1", "identity": 1.0, "cds_coverage": 1.0, "frameshift": False, "aa_changes": [], "aa_changes_n": 0, "seq_length": 1000, "other_read_issues": ["read2(aa:S334L)"]}
        assert judge_sample(sample, thresholds)["rule"] == 1

    def test_rule2_seq_failure_low_identity(self, thresholds):
        sample = {"sid": "X2", "identity": 0.20, "cds_coverage": 0.5, "frameshift": False, "aa_changes": [], "aa_changes_n": 0, "seq_length": 500}
        assert judge_sample(sample, thresholds)["rule"] == 2

    def test_rule2_seq_failure_short(self, thresholds):
        sample = {"sid": "X3", "identity": 0.95, "cds_coverage": 0.1, "frameshift": False, "aa_changes": [], "aa_changes_n": 0, "seq_length": 30}
        assert judge_sample(sample, thresholds)["rule"] == 2

    def test_rule3_overlap_alignment_failure(self, thresholds):
        sample = {"sid": "X4", "identity": 0.50, "cds_coverage": 0.3, "frameshift": True, "aa_changes": [], "aa_changes_n": 109, "seq_length": 459}
        assert judge_sample(sample, thresholds)["rule"] == 3

    def test_rule4_overlap_moderate(self, thresholds):
        sample = {"sid": "X5", "identity": 0.80, "cds_coverage": 0.6, "frameshift": False, "aa_changes": ["A1B"]*30, "aa_changes_n": 30, "seq_length": 800}
        assert judge_sample(sample, thresholds)["rule"] == 4

    def test_rule5_frameshift(self, thresholds):
        sample = {"sid": "X6", "identity": 1.0, "cds_coverage": 0.511, "frameshift": True, "aa_changes": [], "aa_changes_n": 0, "seq_length": 717}
        assert judge_sample(sample, thresholds)["rule"] == 5

    def test_rule6_real_aa_mutations(self, thresholds):
        sample = {"sid": "X7", "identity": 0.998, "cds_coverage": 1.0, "frameshift": False, "aa_changes": ["Q131T"], "aa_changes_n": 1, "seq_length": 1529}
        r = judge_sample(sample, thresholds)
        assert r["rule"] == 6 and "Q131T" in r["reason"]

    def test_rule7_segment_deletion(self, thresholds):
        sample = {"sid": "X8", "identity": 0.96, "cds_coverage": 0.622, "frameshift": False, "aa_changes": ["R176L","Y177*","V178E","I179G","E180D"], "aa_changes_n": 15, "seq_length": 986}
        assert judge_sample(sample, thresholds)["rule"] == 7

    def test_rule9_low_coverage_ok(self, thresholds):
        sample = {"sid": "X10", "identity": 1.0, "cds_coverage": 0.445, "frameshift": False, "aa_changes": [], "aa_changes_n": 0, "seq_length": 624}
        r = judge_sample(sample, thresholds)
        assert r["rule"] == 9 and r["status"] == "ok"

    def test_rule10_normal(self, thresholds):
        sample = {"sid": "X11", "identity": 1.0, "cds_coverage": 1.0, "frameshift": False, "aa_changes": [], "aa_changes_n": 0, "seq_length": 1523}
        r = judge_sample(sample, thresholds)
        assert r["rule"] == 10 and r["status"] == "ok"

    def test_fallback_uncertain(self, thresholds):
        sample = {"sid": "X12", "identity": 0.93, "cds_coverage": 0.9, "frameshift": False, "aa_changes": ["A1B","C2D","E3F"], "aa_changes_n": 3, "seq_length": 1000}
        r = judge_sample(sample, thresholds)
        assert r["status"] == "uncertain" and r["rule"] == -1

class TestTruthDataBase:
    BASE_EVIDENCE = [
        {"sid": "C373-2", "identity": 1.0, "cds_coverage": 1.0, "frameshift": False, "aa_changes": [], "aa_changes_n": 0, "seq_length": 1523},
        {"sid": "C376-2", "identity": 1.0, "cds_coverage": 1.0, "frameshift": False, "aa_changes": [], "aa_changes_n": 0, "seq_length": 1529},
        {"sid": "C379-2", "identity": 0.998, "cds_coverage": 1.0, "frameshift": False, "aa_changes": ["Q131T"], "aa_changes_n": 1, "seq_length": 1529},
        {"sid": "C379-a", "identity": 1.0, "cds_coverage": 1.0, "frameshift": False, "aa_changes": [], "aa_changes_n": 0, "seq_length": 1529},
        {"sid": "C397-2", "identity": 0.9967, "cds_coverage": 1.0, "frameshift": False, "aa_changes": ["K171M","L334S"], "aa_changes_n": 2, "seq_length": 1529},
        {"sid": "C397-a", "identity": 0.998, "cds_coverage": 1.0, "frameshift": False, "aa_changes": ["P431Q","A435V","L456I"], "aa_changes_n": 3, "seq_length": 1529},
        {"sid": "C402-2", "identity": 0.9948, "cds_coverage": 1.0, "frameshift": False, "aa_changes": ["R171M","L334S","S335A"], "aa_changes_n": 3, "seq_length": 1529},
        {"sid": "C405-2", "identity": 1.0, "cds_coverage": 1.0, "frameshift": False, "aa_changes": [], "aa_changes_n": 0, "seq_length": 1523},
        {"sid": "C406-2", "identity": 1.0, "cds_coverage": 1.0, "frameshift": False, "aa_changes": [], "aa_changes_n": 0, "seq_length": 1523},
        {"sid": "C410-a", "identity": 1.0, "cds_coverage": 1.0, "frameshift": False, "aa_changes": [], "aa_changes_n": 0, "seq_length": 1529},
    ]
    BASE_TRUTH = {"C373-2":"ok","C376-2":"ok","C379-2":"wrong","C379-a":"ok","C397-2":"wrong","C397-a":"wrong","C402-2":"wrong","C405-2":"ok","C406-2":"ok","C410-a":"ok"}

    def test_base_dataset_accuracy(self, thresholds):
        for s in self.BASE_EVIDENCE:
            r = judge_sample(s, thresholds)
            expected = self.BASE_TRUTH[s["sid"]]
            assert r["status"] == expected, f"{s['sid']}: expected={expected}, got={r['status']} (rule={r['rule']})"

class TestTruthDataPromax:
    PROMAX_EVIDENCE = [
        {"sid":"C351-2","identity":1.0,"cds_coverage":1.0,"frameshift":False,"aa_changes":[],"aa_changes_n":0,"seq_length":1412},
        {"sid":"C358-1","identity":1.0,"cds_coverage":0.502,"frameshift":False,"aa_changes":[],"aa_changes_n":0,"seq_length":732},
        {"sid":"C363-2","identity":0.4967,"cds_coverage":0.323,"frameshift":True,"aa_changes":[],"aa_changes_n":0,"seq_length":459},
        {"sid":"C364-2","identity":0.9626,"cds_coverage":0.622,"frameshift":False,"aa_changes":["R176L","Y177*","V178E","I179G","E180D","L181I","F182H","V183M","T185S","F186M","K187N","K188V","P192V","L195I","Y196E"],"aa_changes_n":15,"seq_length":986},
        {"sid":"C366-2","identity":0.9987,"cds_coverage":1.0,"frameshift":False,"aa_changes":["P431Q","A435V"],"aa_changes_n":2,"seq_length":1529},
        {"sid":"C367-1","identity":0.0,"cds_coverage":0.0,"frameshift":False,"aa_changes":[],"aa_changes_n":0,"seq_length":0},
        {"sid":"C368-1","identity":1.0,"cds_coverage":0.517,"frameshift":False,"aa_changes":[],"aa_changes_n":0,"seq_length":746},
        {"sid":"C369-1","identity":1.0,"cds_coverage":0.500,"frameshift":False,"aa_changes":[],"aa_changes_n":0,"seq_length":724},
        {"sid":"C370-1","identity":1.0,"cds_coverage":0.511,"frameshift":True,"aa_changes":[],"aa_changes_n":0,"seq_length":717},
        {"sid":"C371-1","identity":1.0,"cds_coverage":0.509,"frameshift":False,"aa_changes":[],"aa_changes_n":0,"seq_length":729},
        {"sid":"C373-1","identity":1.0,"cds_coverage":0.463,"frameshift":False,"aa_changes":[],"aa_changes_n":0,"seq_length":673},
        {"sid":"C374-1","identity":1.0,"cds_coverage":0.445,"frameshift":False,"aa_changes":[],"aa_changes_n":0,"seq_length":615},
    ]
    PROMAX_TRUTH = {"C351-2":"ok","C358-1":"ok","C363-2":"wrong","C364-2":"wrong","C366-2":"wrong","C367-1":"wrong","C368-1":"ok","C369-1":"ok","C370-1":"wrong","C371-1":"ok","C373-1":"ok","C374-1":"ok"}

    def test_promax_dataset_accuracy(self, thresholds):
        mismatches = []
        for s in self.PROMAX_EVIDENCE:
            r = judge_sample(s, thresholds)
            expected = self.PROMAX_TRUTH[s["sid"]]
            actual = r["status"]
            if actual != expected:
                mismatches.append(f"{s['sid']}: expected={expected}, got={actual} (rule={r['rule']})")
        accuracy = (len(self.PROMAX_EVIDENCE) - len(mismatches)) / len(self.PROMAX_EVIDENCE)
        assert accuracy >= 0.90, f"Accuracy {accuracy:.0%}. Mismatches: {mismatches}"

class TestTruthDataPro:
    PRO_EVIDENCE = [
        {"sid":"C363-3","identity":1.0,"cds_coverage":0.445,"frameshift":False,"aa_changes":[],"aa_changes_n":0,"seq_length":624},
        {"sid":"C364-6","identity":1.0,"cds_coverage":0.518,"frameshift":False,"aa_changes":[],"aa_changes_n":0,"seq_length":725},
        {"sid":"C366-3","identity":1.0,"cds_coverage":0.502,"frameshift":False,"aa_changes":[],"aa_changes_n":0,"seq_length":731},
        {"sid":"C370-2","identity":1.0,"cds_coverage":0.518,"frameshift":False,"aa_changes":[],"aa_changes_n":0,"seq_length":753},
        {"sid":"C377-1","identity":1.0,"cds_coverage":0.445,"frameshift":False,"aa_changes":[],"aa_changes_n":0,"seq_length":624},
        {"sid":"C381-1","identity":1.0,"cds_coverage":0.511,"frameshift":False,"aa_changes":[],"aa_changes_n":0,"seq_length":746},
        {"sid":"C403-1","identity":1.0,"cds_coverage":0.444,"frameshift":False,"aa_changes":[],"aa_changes_n":0,"seq_length":646},
    ]
    PRO_TRUTH = {"C363-3":"ok","C364-6":"ok","C366-3":"wrong","C370-2":"ok","C377-1":"ok","C381-1":"ok","C403-1":"ok"}

    def test_pro_dataset_accuracy(self, thresholds):
        mismatches = []
        for s in self.PRO_EVIDENCE:
            r = judge_sample(s, thresholds)
            expected = self.PRO_TRUTH[s["sid"]]
            actual = r["status"]
            if actual != expected:
                mismatches.append(f"{s['sid']}: expected={expected}, got={actual}")
        assert len(mismatches) <= 1, f"Too many mismatches: {mismatches}"
