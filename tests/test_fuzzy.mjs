import test from "node:test";
import assert from "node:assert/strict";
import { fuzzyScore } from "../src/lib/commands/fuzzy.js";

test("exact prefix scores higher than sparse match", () => {
  const exact = fuzzyScore("export csv", "exp");
  const sparse = fuzzyScore("export csv", "ecv");
  assert.ok(exact > sparse, `exact ${exact} should exceed sparse ${sparse}`);
});

test("non-subsequence returns -1", () => {
  assert.equal(fuzzyScore("export csv", "zzz"), -1);
});

test("subsequence match is positive", () => {
  assert.ok(fuzzyScore("export csv", "ecv") > 0);
});

test("case-insensitive", () => {
  assert.ok(fuzzyScore("Export CSV", "csv") > 0);
  assert.ok(fuzzyScore("EXPORT", "exp") > 0);
});

test("empty query scores 0", () => {
  assert.equal(fuzzyScore("anything", ""), 0);
});

test("query longer than text returns -1", () => {
  assert.equal(fuzzyScore("abc", "abcd"), -1);
});
