from collections import defaultdict


def analyze_mutation_trends(samples):
    status_summary = defaultdict(int)
    hotspot_counts = defaultdict(int)
    effect_counts = defaultdict(int)
    per_clone = []
    total_mutations = 0

    for sample in samples:
        status = sample.get("status", "unknown")
        status_summary[status] += 1
        clone = sample.get("clone") or sample.get("id") or "?"
        mutations = sample.get("mutations") or []
        per_clone.append(
            {
                "clone": clone,
                "mutation_count": len(mutations),
                "mutations": mutations,
            }
        )

        for mutation in mutations:
            position = mutation.get("position")
            ref_base = mutation.get("refBase")
            query_base = mutation.get("queryBase")
            effect = mutation.get("effect")
            hotspot_counts[(position, ref_base, query_base)] += 1
            if effect:
                effect_counts[effect] += 1
            total_mutations += 1

    mutation_hotspots = []
    for (position, ref_base, query_base), count in sorted(hotspot_counts.items(), key=lambda item: (-item[1], item[0][0] or 0)):
        if count < 2:
            continue
        mutation_hotspots.append(
            {
                "position": position,
                "ref_base": ref_base,
                "query_base": query_base,
                "count": count,
                "frequency": count / max(len(samples), 1),
            }
        )

    common_effects = [
        {"effect": effect, "count": count}
        for effect, count in sorted(effect_counts.items(), key=lambda item: (-item[1], item[0]))
    ]

    insights = []
    wrong_count = status_summary.get("wrong", 0)
    if len(samples):
        wrong_ratio = wrong_count / len(samples)
        if wrong_ratio > 0.5:
            insights.append(f"异常率较高，{wrong_count}/{len(samples)} 个样本判定为 wrong。")
    if mutation_hotspots:
        top = mutation_hotspots[0]
        insights.append(f"发现突变热点：位置 {top['position']} 在 {top['count']} 个样本中重复出现。")
    if common_effects:
        top_effect = common_effects[0]
        insights.append(f"最常见变异效应为 {top_effect['effect']}，共出现 {top_effect['count']} 次。")
    if total_mutations == 0:
        insights.append("未发现明显突变热点，整体序列一致性良好。")

    return {
        "total_samples": len(samples),
        "total_mutations": total_mutations,
        "mutation_hotspots": mutation_hotspots,
        "status_summary": dict(status_summary),
        "common_effects": common_effects,
        "per_clone_trends": per_clone,
        "insights": insights,
    }
