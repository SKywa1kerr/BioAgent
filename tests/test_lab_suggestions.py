from bioagent.lab_suggestions import generate_lab_suggestions


def test_generate_lab_suggestions_flags_quality_and_frameshift_issues():
    samples = [
        {
            "id": "S1",
            "clone": "C201-1",
            "status": "wrong",
            "identity": 0.42,
            "coverage": 0.38,
            "avg_quality": 14,
            "frameshift": True,
            "mutations": [{"position": 10}],
        },
        {
            "id": "S2",
            "clone": "C201-2",
            "status": "wrong",
            "identity": 0.82,
            "coverage": 0.51,
            "avg_quality": 20,
            "frameshift": False,
            "mutations": [],
        },
    ]

    result = generate_lab_suggestions(samples)

    assert result["overall_health"] in {"critical", "needs_attention"}
    assert len(result["diagnoses"]) >= 2
    assert any("移码" in item["issue"] for item in result["diagnoses"])
    assert any("模板" in item["suggestion"] or "引物" in item["suggestion"] for item in result["diagnoses"])
    assert result["suggestions"]


def test_generate_lab_suggestions_reports_healthy_dataset():
    samples = [
        {
            "id": "S1",
            "clone": "C1",
            "status": "ok",
            "identity": 0.99,
            "coverage": 0.96,
            "avg_quality": 38,
            "frameshift": False,
            "mutations": [],
        }
    ]

    result = generate_lab_suggestions(samples)

    assert result["overall_health"] == "good"
    assert result["diagnoses"] == []
    assert "良好" in result["summary"] or "未发现" in result["summary"]
