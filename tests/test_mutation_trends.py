from bioagent.mutation_trends import analyze_mutation_trends


def test_analyze_mutation_trends_finds_hotspots_and_insights():
    samples = [
        {
            "id": "S1",
            "clone": "C101-1",
            "status": "wrong",
            "mutations": [
                {"position": 12, "refBase": "A", "queryBase": "G", "type": "substitution", "effect": "missense"},
                {"position": 30, "refBase": "C", "queryBase": "T", "type": "substitution", "effect": "synonymous"},
            ],
        },
        {
            "id": "S2",
            "clone": "C101-2",
            "status": "wrong",
            "mutations": [
                {"position": 12, "refBase": "A", "queryBase": "G", "type": "substitution", "effect": "missense"},
            ],
        },
        {
            "id": "S3",
            "clone": "C101-3",
            "status": "ok",
            "mutations": [],
        },
    ]

    result = analyze_mutation_trends(samples)

    assert result["total_samples"] == 3
    assert result["total_mutations"] == 3
    assert result["status_summary"] == {"wrong": 2, "ok": 1}
    assert result["mutation_hotspots"][0]["position"] == 12
    assert result["mutation_hotspots"][0]["count"] == 2
    assert result["common_effects"][0]["effect"] == "missense"
    assert any("热点" in insight for insight in result["insights"])


def test_analyze_mutation_trends_reports_clean_dataset():
    samples = [
        {"id": "S1", "clone": "C1", "status": "ok", "mutations": []},
        {"id": "S2", "clone": "C2", "status": "ok", "mutations": []},
    ]

    result = analyze_mutation_trends(samples)

    assert result["total_mutations"] == 0
    assert result["mutation_hotspots"] == []
    assert any("未发现" in insight or "一致性良好" in insight for insight in result["insights"])
