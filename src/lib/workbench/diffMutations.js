// Pure helper for the workbench compare-view "Diff highlight" toggle.
//
// Given two arrays of mutations (left/right sample), partition the union of
// their numeric `position` values into three sets:
//   - uniqueLeftPositions  : positions present only in `left`
//   - uniqueRightPositions : positions present only in `right`
//   - sharedPositions      : positions present in both
//
// Membership is "any mutation at that position" — we deliberately ignore
// refBase / queryBase / type / effect for now. Entries with missing or
// non-finite `position` are skipped. The consumer uses the resulting Sets
// for keyed row → highlight class lookups in the side-by-side mutation table.

function collectPositions(arr) {
  const set = new Set();
  if (!Array.isArray(arr)) return set;
  for (const m of arr) {
    if (!m) continue;
    const p = m.position;
    if (typeof p === "number" && Number.isFinite(p)) {
      set.add(p);
    }
  }
  return set;
}

export function diffMutations(left, right) {
  const leftSet = collectPositions(left);
  const rightSet = collectPositions(right);

  const uniqueLeftPositions = new Set();
  const uniqueRightPositions = new Set();
  const sharedPositions = new Set();

  for (const p of leftSet) {
    if (rightSet.has(p)) {
      sharedPositions.add(p);
    } else {
      uniqueLeftPositions.add(p);
    }
  }
  for (const p of rightSet) {
    if (!leftSet.has(p)) {
      uniqueRightPositions.add(p);
    }
  }

  return { uniqueLeftPositions, uniqueRightPositions, sharedPositions };
}
