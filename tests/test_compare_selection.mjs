import test from "node:test";
import assert from "node:assert/strict";
import { nextCompareSelection } from "../src/lib/workbench/compareSelection.js";

test("adds an id to an empty selection", () => {
  assert.deepEqual(nextCompareSelection([], "A"), ["A"]);
});

test("adds a second id when the selection has one entry", () => {
  assert.deepEqual(nextCompareSelection(["A"], "B"), ["A", "B"]);
});

test("FIFO replaces the older id when at capacity", () => {
  // ["A", "B"] means A came first (older), B came second (newer).
  // Adding "C" should drop A and keep B → ["B", "C"].
  assert.deepEqual(nextCompareSelection(["A", "B"], "C"), ["B", "C"]);
});

test("toggles off when the id is already selected", () => {
  assert.deepEqual(nextCompareSelection(["A", "B"], "A"), ["B"]);
  assert.deepEqual(nextCompareSelection(["A", "B"], "B"), ["A"]);
  assert.deepEqual(nextCompareSelection(["A"], "A"), []);
});

test("ignores non-string ids and preserves order", () => {
  assert.deepEqual(nextCompareSelection(["A"], ""), ["A"]);
  // @ts-expect-error: deliberately malformed id
  assert.deepEqual(nextCompareSelection(["A"], null), ["A"]);
});

test("respects custom max when supplied", () => {
  // max=3 → second add still appends, third add appends, fourth FIFO-replaces.
  assert.deepEqual(nextCompareSelection(["A", "B", "C"], "D", 3), ["B", "C", "D"]);
  assert.deepEqual(nextCompareSelection(["A", "B"], "C", 3), ["A", "B", "C"]);
});
