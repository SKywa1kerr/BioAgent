def generate_lab_suggestions(samples):
    diagnoses = []
    suggestion_tags = set()

    for sample in samples:
        clone = sample.get("clone") or sample.get("id") or "?"
        identity = sample.get("identity", 1.0)
        coverage = sample.get("coverage", sample.get("cds_coverage", 1.0))
        avg_quality = sample.get("avg_quality", sample.get("avg_qry_quality", 40))
        frameshift = sample.get("frameshift", False)

        if identity < 0.5:
            diagnoses.append(
                {
                    "clone": clone,
                    "issue": "测序信号极弱",
                    "severity": "critical",
                    "suggestion": "模板质量或浓度不足，建议重新提取模板并提高送样浓度。",
                }
            )
            suggestion_tags.add("模板浓度优化")

        elif identity < 0.85:
            diagnoses.append(
                {
                    "clone": clone,
                    "issue": "疑似重叠峰或非特异扩增",
                    "severity": "high",
                    "suggestion": "建议重新检查引物特异性，并适当提高退火温度。",
                }
            )
            suggestion_tags.add("引物与退火温度优化")

        if coverage < 0.6:
            diagnoses.append(
                {
                    "clone": clone,
                    "issue": "覆盖度不足",
                    "severity": "high",
                    "suggestion": "建议更换测序引物位置，提升 CDS 覆盖范围。",
                }
            )
            suggestion_tags.add("覆盖度优化")

        if avg_quality < 18:
            diagnoses.append(
                {
                    "clone": clone,
                    "issue": "峰图质量偏低",
                    "severity": "high",
                    "suggestion": "建议纯化 PCR 产物后再测，减少背景噪音。",
                }
            )
            suggestion_tags.add("纯化后重测")

        if frameshift:
            diagnoses.append(
                {
                    "clone": clone,
                    "issue": "检测到移码风险",
                    "severity": "critical",
                    "suggestion": "建议优先复核该样本并重新测序确认是否为真实移码。",
                }
            )
            suggestion_tags.add("复核移码样本")

    if any(item["severity"] == "critical" for item in diagnoses):
        overall_health = "critical"
        summary = "发现严重实验风险，建议优先处理异常样本。"
    elif diagnoses:
        overall_health = "needs_attention"
        summary = "发现若干需要关注的问题，建议优化实验条件后重测。"
    else:
        overall_health = "good"
        summary = "整体质量良好，未发现明显实验体系问题。"

    return {
        "diagnoses": diagnoses,
        "overall_health": overall_health,
        "summary": summary,
        "suggestions": sorted(suggestion_tags),
    }
