import test from "node:test";
import assert from "node:assert/strict";
import { compactRowView } from "../src/lib/workbench/compactRow.js";

test("empty aa_changes collapses to dash sentinel", () => {
  const view = compactRowView({ id: "C1-1", aa_changes: [] });
  assert.deepEqual(view.aaPills, []);
  assert.equal(view.aaOverflow, 0);
});

test("up to 3 aa changes render as pills", () => {
  const view = compactRowView({ id: "C1-1", aa_changes: ["K171M", "L334S", "Q131T"] });
  assert.deepEqual(view.aaPills, ["K171M", "L334S", "Q131T"]);
  assert.equal(view.aaOverflow, 0);
});

test("more than 3 aa changes show +N overflow", () => {
  const view = compactRowView({
    id: "C1-1",
    aa_changes: ["K171M", "L334S", "Q131T", "R200W", "V300A"],
  });
  assert.deepEqual(view.aaPills, ["K171M", "L334S", "Q131T"]);
  assert.equal(view.aaOverflow, 2);
});

test("string aa_changes json is parsed", () => {
  const view = compactRowView({ id: "C1-1", aa_changes: '["K171M","L334S"]' });
  assert.deepEqual(view.aaPills, ["K171M", "L334S"]);
});

test("synonymous mutations feed the mutation type set", () => {
  const view = compactRowView({
    id: "C1-1",
    aa_changes: [],
    mutations: [
      { type: "substitution", effect: "synonymous" },
      { type: "insertion", effect: "" },
    ],
  });
  assert.ok(view.mutationTypes.includes("synonymous"));
  assert.ok(view.mutationTypes.includes("insertion"));
});
