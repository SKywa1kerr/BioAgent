export function fuzzyScore(text, query) {
  if (typeof text !== "string" || typeof query !== "string") return -1;
  if (query.length === 0) return 0;
  if (query.length > text.length) return -1;
  const t = text.toLowerCase();
  const q = query.toLowerCase();
  let ti = 0;
  let qi = 0;
  let score = 0;
  let lastMatch = -2;
  while (ti < t.length && qi < q.length) {
    if (t[ti] === q[qi]) {
      score += ti - lastMatch === 1 ? 3 : 1;
      score += Math.max(0, 10 - ti);
      lastMatch = ti;
      qi++;
    }
    ti++;
  }
  if (qi < q.length) return -1;
  return score;
}
