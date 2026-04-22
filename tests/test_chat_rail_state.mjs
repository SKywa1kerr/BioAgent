import test from "node:test";
import assert from "node:assert/strict";
import { nextRailState, loadRailState, saveRailState } from "../src/lib/ui/chatRailState.js";

test("cycles wide -> narrow -> hidden -> wide", () => {
  assert.equal(nextRailState("wide"), "narrow");
  assert.equal(nextRailState("narrow"), "hidden");
  assert.equal(nextRailState("hidden"), "wide");
});

test("load from storage returns wide by default", () => {
  const store = {};
  const fake = { getItem: (k) => store[k] ?? null, setItem: (k, v) => { store[k] = v; } };
  assert.equal(loadRailState(fake), "wide");
});

test("round-trips through save/load", () => {
  const store = {};
  const fake = { getItem: (k) => store[k] ?? null, setItem: (k, v) => { store[k] = v; } };
  saveRailState("narrow", fake);
  assert.equal(loadRailState(fake), "narrow");
});

test("invalid stored value falls back to wide", () => {
  const fake = { getItem: () => "lol", setItem: () => {} };
  assert.equal(loadRailState(fake), "wide");
});
