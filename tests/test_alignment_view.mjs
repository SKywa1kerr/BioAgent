import test from "node:test";
import assert from "node:assert/strict";
import {
  buildAlignmentViewModel,
  buildCoordinateMap,
  parseAaChanges,
} from "../src/components/workbench/alignmentView.js";

test("buildCoordinateMap maps gapped ref and query coordinates", () => {
  const map = buildCoordinateMap("ATG-C", "A-GTC");
  assert.deepEqual(map.refToGapped, [0, 1, 2, 4]);
  assert.deepEqual(map.queryToGapped, [0, 2, 3, 4]);
  assert.deepEqual(map.gappedToRef, [0, 1, 2, null, 3]);
  assert.deepEqual(map.gappedToQuery, [0, null, 1, 2, 3]);
});

test("buildAlignmentViewModel maps one-based CDS and mutation positions into gapped ranges", () => {
  const view = buildAlignmentViewModel({
    id: "S1",
    ref_sequence: "ATGC",
    query_sequence: "AGTC",
    aligned_ref_g: "ATG-C",
    aligned_query_g: "A-GTC",
    matches: [true, false, true, false, true],
    cds_start: 2,
    cds_end: 4,
    mutations: [{ position: 4, refBase: "C", queryBase: "T", type: "substitution" }],
    aa_changes: ["S2L"],
  });

  assert.equal(view.refLine, "ATG-C");
  assert.equal(view.queryLine, "A-GTC");
  assert.equal(view.matchLine, "| | |");
  assert.deepEqual(view.cdsRange, { start: 1, end: 5 });
  assert.deepEqual(view.mutationRanges, [{ start: 4, end: 5, label: "C>T", type: "substitution", effect: undefined }]);
  assert.deepEqual(view.aaChanges, ["S2L"]);
});

test("buildAlignmentViewModel returns null without usable alignment strings", () => {
  assert.equal(buildAlignmentViewModel({ id: "S1" }), null);
});

test("parseAaChanges accepts arrays, JSON strings, plain strings, and blanks", () => {
  assert.deepEqual(parseAaChanges(["S2L", ""]), ["S2L"]);
  assert.deepEqual(parseAaChanges('["S2L","K4M"]'), ["S2L", "K4M"]);
  assert.deepEqual(parseAaChanges("S2L K4M"), ["S2L K4M"]);
  assert.deepEqual(parseAaChanges(""), []);
});
