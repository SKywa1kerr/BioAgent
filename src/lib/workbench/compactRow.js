const MAX_PILLS = 3;

function parseAa(value) {
  if (Array.isArray(value)) {
    return value.filter((x) => typeof x === "string" && x.trim().length > 0);
  }
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      if (Array.isArray(parsed)) {
        return parsed.filter((x) => typeof x === "string" && x.trim().length > 0);
      }
    } catch {
      return value.trim() ? [value.trim()] : [];
    }
  }
  return [];
}

export function compactRowView(sample) {
  const aa = parseAa(sample && sample.aa_changes);
  const pills = aa.slice(0, MAX_PILLS);
  const overflow = Math.max(0, aa.length - MAX_PILLS);
  const mutTypes = new Set();
  const mutations = Array.isArray(sample && sample.mutations) ? sample.mutations : [];
  for (const m of mutations) {
    if (m && m.type) mutTypes.add(String(m.type).toLowerCase());
    if (m && m.effect) mutTypes.add(String(m.effect).toLowerCase());
  }
  return { aaPills: pills, aaOverflow: overflow, mutationTypes: Array.from(mutTypes) };
}
