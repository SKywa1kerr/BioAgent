import test from "node:test";
import assert from "node:assert/strict";
import {
  buildAlignmentViewModel,
  buildCoordinateMap,
  parseAaChanges,
} from "../src/components/workbench/alignmentView.js";

test("parseAaChanges accepts arrays as-is and trims entries", () => {
  assert.deepEqual(parseAaChanges([" K171M ", "L334S", ""]), ["K171M", "L334S"]);
});

test("parseAaChanges parses JSON strings of arrays", () => {
  assert.deepEqual(parseAaChanges('["K171M","L334S"]'), ["K171M", "L334S"]);
});

test("parseAaChanges returns plain strings as single-element list", () => {
  assert.deepEqual(parseAaChanges("frameshift"), ["frameshift"]);
});

test("parseAaChanges returns empty array for null/undefined", () => {
  assert.deepEqual(parseAaChanges(null), []);
  assert.deepEqual(parseAaChanges(undefined), []);
  assert.deepEqual(parseAaChanges(""), []);
});

test("buildCoordinateMap maps ref-only positions and skips gaps", () => {
  const map = buildCoordinateMap("ATG-C", "ATGTC");
  assert.deepEqual(map.refToGapped, [0, 1, 2, 4]);
  assert.deepEqual(map.gappedToRef, [0, 1, 2, null, 3]);
  assert.deepEqual(map.queryToGapped, [0, 1, 2, 3, 4]);
  assert.deepEqual(map.gappedToQuery, [0, 1, 2, 3, 4]);
});

test("buildAlignmentViewModel returns null when sequences are missing", () => {
  assert.equal(buildAlignmentViewModel({ id: "x" }), null);
  assert.equal(buildAlignmentViewModel({ id: "x", ref_sequence: "ATG" }), null);
});

test("buildAlignmentViewModel pads sequences and exposes match/tick lines", () => {
  const view = buildAlignmentViewModel({
    id: "S1",
    aligned_ref_g: "ATGCATGCAT",
    aligned_query_g: "ATGCATGCAT",
    cds_start: 1,
    cds_end: 10,
  });
  assert.ok(view);
  assert.equal(view.refLine.length, 10);
  assert.equal(view.matchLine, "||||||||||");
  assert.deepEqual(view.cdsRange, { start: 0, end: 10 });
  assert.equal(view.tickLine.length, 10);
  assert.equal(view.tickLine.endsWith("+"), true);
});

test("buildAlignmentViewModel maps insertion mutations onto the gap column in ref", () => {
  const view = buildAlignmentViewModel({
    id: "S1",
    aligned_ref_g: "ATG-C",
    aligned_query_g: "ATGTC",
    mutations: [
      { position: 4, refBase: "-", queryBase: "T", type: "insertion", effect: "frameshift" },
    ],
  });
  assert.ok(view);
  assert.equal(view.mutationRanges.length, 1);
  const [range] = view.mutationRanges;
  assert.equal(range.start, 3);
  assert.equal(range.end, 4);
  assert.equal(range.label, "->T");
  assert.equal(range.type, "insertion");
  assert.equal(range.effect, "frameshift");
});

test("buildAlignmentViewModel produces match line from explicit matches mask when provided", () => {
  const view = buildAlignmentViewModel({
    id: "S1",
    aligned_ref_g: "ATGC",
    aligned_query_g: "ATGT",
    matches: [true, true, true, false],
  });
  assert.ok(view);
  assert.equal(view.matchLine, "||| ");
});
