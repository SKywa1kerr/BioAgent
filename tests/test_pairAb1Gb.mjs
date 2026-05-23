import test from "node:test";
import assert from "node:assert/strict";
import { pairAb1Gb } from "../src/lib/files/pairAb1Gb.js";

test("returns empty pairs/unpaired arrays when both inputs are empty", () => {
  const r = pairAb1Gb([], []);
  assert.deepEqual(r.pairs, []);
  assert.deepEqual(r.unpairedAb1, []);
  assert.deepEqual(r.unpairedGb, []);
});

test("pairs one-to-one matches and leaves unpaired arrays empty", () => {
  const r = pairAb1Gb(["data/ab1/foo.ab1"], ["data/gb/foo.gb"]);
  assert.equal(r.pairs.length, 1);
  assert.equal(r.pairs[0].basename, "foo");
  assert.equal(r.pairs[0].ab1, "data/ab1/foo.ab1");
  assert.equal(r.pairs[0].gb, "data/gb/foo.gb");
  assert.deepEqual(r.unpairedAb1, []);
  assert.deepEqual(r.unpairedGb, []);
});

test("AB1 with no matching GB lands in unpairedAb1", () => {
  const r = pairAb1Gb(["a/orphan.ab1"], []);
  assert.deepEqual(r.pairs, []);
  assert.deepEqual(r.unpairedAb1, ["a/orphan.ab1"]);
  assert.deepEqual(r.unpairedGb, []);
});

test("GB with no matching AB1 lands in unpairedGb", () => {
  const r = pairAb1Gb([], ["a/lonely.gbk"]);
  assert.deepEqual(r.pairs, []);
  assert.deepEqual(r.unpairedAb1, []);
  assert.deepEqual(r.unpairedGb, ["a/lonely.gbk"]);
});

test("matches basenames case-insensitively", () => {
  const r = pairAb1Gb(["data/Foo.ab1"], ["other/foo.gb"]);
  assert.equal(r.pairs.length, 1);
  assert.equal(r.pairs[0].ab1, "data/Foo.ab1");
  assert.equal(r.pairs[0].gb, "other/foo.gb");
});

test("normalises backslash paths so Windows-style inputs pair correctly", () => {
  const r = pairAb1Gb(["C:\\data\\ab1\\foo.ab1"], ["C:\\data\\gb\\foo.gb"]);
  assert.equal(r.pairs.length, 1);
  assert.equal(r.pairs[0].basename, "foo");
  assert.equal(r.pairs[0].ab1, "C:\\data\\ab1\\foo.ab1");
  assert.equal(r.pairs[0].gb, "C:\\data\\gb\\foo.gb");
});

test("files with the wrong extension fall into the matching unpaired array", () => {
  const r = pairAb1Gb(
    ["a/foo.txt", "a/bar.ab1"],
    ["b/bar.fasta", "b/bar.gb"],
  );
  assert.equal(r.pairs.length, 1);
  assert.equal(r.pairs[0].basename, "bar");
  assert.deepEqual(r.unpairedAb1, ["a/foo.txt"]);
  assert.deepEqual(r.unpairedGb, ["b/bar.fasta"]);
});

test("duplicate ab1 basenames keep the last write and demote earlier ones to unpaired", () => {
  const r = pairAb1Gb(
    ["one/foo.ab1", "two/foo.ab1"],
    ["ref/foo.gb"],
  );
  assert.equal(r.pairs.length, 1);
  assert.equal(r.pairs[0].ab1, "two/foo.ab1");
  assert.equal(r.pairs[0].gb, "ref/foo.gb");
  assert.deepEqual(r.unpairedAb1, ["one/foo.ab1"]);
  assert.deepEqual(r.unpairedGb, []);
});

test("pairs are sorted by basename for deterministic output", () => {
  const r = pairAb1Gb(
    ["a/zeta.ab1", "a/alpha.ab1", "a/mike.ab1"],
    ["b/zeta.gb", "b/alpha.gb", "b/mike.gb"],
  );
  assert.deepEqual(
    r.pairs.map((p) => p.basename),
    ["alpha", "mike", "zeta"],
  );
});

test("single .gb reference pairs with every .ab1 when no stems match (Sanger 1-to-many)", () => {
  // Typical Sanger workflow: one plasmid reference, many trace files whose
  // basenames bear no relation to the reference filename.
  const r = pairAb1Gb(
    ["traces/S1-C376.ab1", "traces/S2-C379.ab1", "traces/S3-C397.ab1"],
    ["ref/plasmid.gb"],
  );
  assert.equal(r.pairs.length, 3, `expected 3 pairs, got ${r.pairs.length}`);
  for (const p of r.pairs) {
    assert.equal(p.gb, "ref/plasmid.gb", "all ab1s should reuse the single gb reference");
  }
  assert.deepEqual(r.unpairedAb1, []);
  assert.deepEqual(r.unpairedGb, []);
});

test("1-to-many fallback does NOT trigger when at least one stem matches", () => {
  // If even one ab1 matches the gb by stem, prefer stem-based pairing to
  // avoid accidentally collapsing distinct references into one.
  const r = pairAb1Gb(
    ["traces/plasmid.ab1", "traces/unrelated.ab1"],
    ["ref/plasmid.gb"],
  );
  assert.equal(r.pairs.length, 1, "only the stem-matched pair should be created");
  assert.equal(r.pairs[0].ab1, "traces/plasmid.ab1");
  assert.deepEqual(r.unpairedAb1, ["traces/unrelated.ab1"]);
});

test("1-to-many fallback skipped when there are multiple .gb references", () => {
  // Multiple gb references => ambiguous which one to reuse, fall back to
  // strict stem matching.
  const r = pairAb1Gb(
    ["traces/foo.ab1", "traces/bar.ab1"],
    ["ref/a.gb", "ref/b.gb"],
  );
  assert.equal(r.pairs.length, 0);
  assert.equal(r.unpairedAb1.length, 2);
  assert.equal(r.unpairedGb.length, 2);
});
