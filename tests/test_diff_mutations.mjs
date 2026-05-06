import test from "node:test";
import assert from "node:assert/strict";
import { diffMutations } from "../src/lib/workbench/diffMutations.js";

test("returns empty sets when both inputs are empty", () => {
  const r = diffMutations([], []);
  assert.equal(r.uniqueLeftPositions.size, 0);
  assert.equal(r.uniqueRightPositions.size, 0);
  assert.equal(r.sharedPositions.size, 0);
});

test("treats every position as unique-left when right is empty", () => {
  const left = [{ position: 5 }, { position: 12 }, { position: 47 }];
  const r = diffMutations(left, []);
  assert.deepEqual([...r.uniqueLeftPositions].sort((a, b) => a - b), [5, 12, 47]);
  assert.equal(r.uniqueRightPositions.size, 0);
  assert.equal(r.sharedPositions.size, 0);
});

test("treats every position as unique-right when left is empty", () => {
  const right = [{ position: 3 }, { position: 9 }];
  const r = diffMutations([], right);
  assert.equal(r.uniqueLeftPositions.size, 0);
  assert.deepEqual([...r.uniqueRightPositions].sort((a, b) => a - b), [3, 9]);
  assert.equal(r.sharedPositions.size, 0);
});

test("collects shared positions when both sides match exactly", () => {
  const left = [{ position: 7 }, { position: 19 }];
  const right = [{ position: 7 }, { position: 19 }];
  const r = diffMutations(left, right);
  assert.equal(r.uniqueLeftPositions.size, 0);
  assert.equal(r.uniqueRightPositions.size, 0);
  assert.deepEqual([...r.sharedPositions].sort((a, b) => a - b), [7, 19]);
});

test("partitions a mixed set into left-only, right-only, and shared", () => {
  // shared: 10, 30
  // left-only: 5
  // right-only: 22, 40
  const left = [{ position: 5 }, { position: 10 }, { position: 30 }];
  const right = [{ position: 10 }, { position: 22 }, { position: 30 }, { position: 40 }];
  const r = diffMutations(left, right);
  assert.deepEqual([...r.uniqueLeftPositions].sort((a, b) => a - b), [5]);
  assert.deepEqual([...r.uniqueRightPositions].sort((a, b) => a - b), [22, 40]);
  assert.deepEqual([...r.sharedPositions].sort((a, b) => a - b), [10, 30]);
});

test("ignores ref/query base / effect — membership is by position only", () => {
  // Same position with different refBase / queryBase / effect counts as shared.
  const left = [{ position: 12, refBase: "A", queryBase: "G", effect: "missense" }];
  const right = [{ position: 12, refBase: "T", queryBase: "C", effect: "synonymous" }];
  const r = diffMutations(left, right);
  assert.equal(r.uniqueLeftPositions.size, 0);
  assert.equal(r.uniqueRightPositions.size, 0);
  assert.deepEqual([...r.sharedPositions], [12]);
});

test("skips entries with missing, NaN, or non-numeric position", () => {
  const left = [
    { position: 5 },
    { position: undefined },
    { position: null },
    { position: NaN },
    { position: Infinity },
    { position: "8" }, // non-numeric (string) — also skipped
    { /* no position field at all */ },
  ];
  const right = [{ position: 5 }, { position: NaN }];
  const r = diffMutations(left, right);
  // Only position 5 is finite numeric on both sides, so it is shared.
  assert.equal(r.uniqueLeftPositions.size, 0);
  assert.equal(r.uniqueRightPositions.size, 0);
  assert.deepEqual([...r.sharedPositions], [5]);
});

test("deduplicates repeated positions within the same side", () => {
  // Two mutations at position 11 on the left, one at 11 on the right →
  // one shared entry, no duplicates leaking into either Set.
  const left = [{ position: 11 }, { position: 11 }, { position: 14 }];
  const right = [{ position: 11 }];
  const r = diffMutations(left, right);
  assert.deepEqual([...r.uniqueLeftPositions], [14]);
  assert.equal(r.uniqueRightPositions.size, 0);
  assert.deepEqual([...r.sharedPositions], [11]);
});

test("handles null / non-array inputs without throwing", () => {
  // The helper is fed via WorkbenchSample.mutations which can be undefined.
  // @ts-expect-error: deliberately malformed input
  const r1 = diffMutations(null, undefined);
  assert.equal(r1.uniqueLeftPositions.size, 0);
  assert.equal(r1.uniqueRightPositions.size, 0);
  assert.equal(r1.sharedPositions.size, 0);
  // @ts-expect-error: deliberately malformed input
  const r2 = diffMutations(undefined, [{ position: 7 }]);
  assert.equal(r2.uniqueLeftPositions.size, 0);
  assert.deepEqual([...r2.uniqueRightPositions], [7]);
});
