import test from "node:test";
import assert from "node:assert/strict";
import {
  readControls,
  writeControls,
  DEFAULT_CONTROLS,
  CONTROLS_STORAGE_KEY,
} from "../src/hooks/useWorkbenchControls.js";

function makeStore() {
  const data = new Map();
  return {
    getItem: (k) => (data.has(k) ? data.get(k) : null),
    setItem: (k, v) => data.set(k, v),
    removeItem: (k) => data.delete(k),
    _data: data,
  };
}

test("readControls returns defaults when storage empty", () => {
  const store = makeStore();
  assert.deepEqual(readControls(store), DEFAULT_CONTROLS);
});

test("writeControls + readControls round trips", () => {
  const store = makeStore();
  const next = { statusFilter: "wrong", searchQuery: "abc", sortKey: "identity", summaryScope: "all" };
  writeControls(store, next);
  assert.deepEqual(readControls(store), next);
});

test("readControls falls back to defaults on invalid schema", () => {
  const store = makeStore();
  store.setItem(CONTROLS_STORAGE_KEY, JSON.stringify({ statusFilter: "bogus" }));
  assert.deepEqual(readControls(store), DEFAULT_CONTROLS);
});

test("readControls tolerates malformed JSON", () => {
  const store = makeStore();
  store.setItem(CONTROLS_STORAGE_KEY, "{not json");
  assert.deepEqual(readControls(store), DEFAULT_CONTROLS);
});

test("writeControls ignores invalid input silently", () => {
  const store = makeStore();
  writeControls(store, { statusFilter: "nope", searchQuery: "", sortKey: "status", summaryScope: "filtered" });
  assert.equal(store.getItem(CONTROLS_STORAGE_KEY), null);
});
