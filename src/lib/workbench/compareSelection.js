// Pure helper for the workbench compare-selection state.
//
// The selection is an array of sample ids. The order encodes insertion order:
// the FIRST entry is the oldest pick, the LAST entry is the newest. When the
// user clicks an id we apply these rules:
//
//   - if the id is already in the list, remove it (toggle off)
//   - else if there is room (length < max), append it
//   - else drop the OLDEST (index 0) and append the new id (FIFO replacement)
//
// `max` defaults to 2 — the compare view only shows two samples.

const DEFAULT_MAX = 2;

export function nextCompareSelection(prev, id, max) {
  const limit = typeof max === "number" && max > 0 ? Math.floor(max) : DEFAULT_MAX;
  const arr = Array.isArray(prev) ? prev.filter((x) => typeof x === "string" && x.length > 0) : [];
  if (typeof id !== "string" || id.length === 0) return arr;

  const idx = arr.indexOf(id);
  if (idx >= 0) {
    return arr.filter((x) => x !== id);
  }
  if (arr.length < limit) {
    return [...arr, id];
  }
  return [...arr.slice(arr.length - limit + 1), id];
}
