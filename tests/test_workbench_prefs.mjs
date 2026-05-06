import test from "node:test";
import assert from "node:assert/strict";
import {
  loadWorkbenchPrefs,
  saveWorkbenchPrefs,
  defaultWorkbenchPrefs,
} from "../src/lib/ui/workbenchPrefs.js";

const STORAGE_KEY = "bioagent.workbench.prefs.v1";

function makeStore() {
  const data = new Map();
  return {
    getItem: (k) => (data.has(k) ? data.get(k) : null),
    setItem: (k, v) => data.set(k, v),
    removeItem: (k) => data.delete(k),
    _data: data,
  };
}

test("loadWorkbenchPrefs returns defaults when storage is empty", () => {
  const store = makeStore();
  assert.deepEqual(loadWorkbenchPrefs(store), defaultWorkbenchPrefs);
});

test("loadWorkbenchPrefs falls back to defaults on malformed JSON", () => {
  const store = makeStore();
  store.setItem(STORAGE_KEY, "{not json");
  assert.deepEqual(loadWorkbenchPrefs(store), defaultWorkbenchPrefs);
});

test("loadWorkbenchPrefs clamps unknown enum values to defaults", () => {
  const store = makeStore();
  store.setItem(
    STORAGE_KEY,
    JSON.stringify({
      sortKey: "bogus",
      sortDir: "sideways",
      density: "ultra",
      statusFilter: "weird",
      summaryScope: "moonbeams",
    }),
  );
  assert.deepEqual(loadWorkbenchPrefs(store), defaultWorkbenchPrefs);
});

test("loadWorkbenchPrefs clamps individual unknown fields while keeping valid ones", () => {
  const store = makeStore();
  store.setItem(
    STORAGE_KEY,
    JSON.stringify({
      sortKey: "identity",
      sortDir: "garbage",
      density: "detailed",
      statusFilter: "wrong",
      summaryScope: "all",
    }),
  );
  assert.deepEqual(loadWorkbenchPrefs(store), {
    sortKey: "identity",
    sortDir: defaultWorkbenchPrefs.sortDir,
    density: "detailed",
    statusFilter: "wrong",
    summaryScope: "all",
  });
});

test("save then load round-trips the prefs", () => {
  const store = makeStore();
  const next = {
    sortKey: "coverage",
    sortDir: "asc",
    density: "detailed",
    statusFilter: "uncertain",
    summaryScope: "all",
  };
  saveWorkbenchPrefs(next, store);
  assert.deepEqual(loadWorkbenchPrefs(store), next);
});

test("save then load falls back when summaryScope is missing from saved blob", () => {
  // Older builds wrote prefs without summaryScope. Simulate that.
  const store = makeStore();
  store.setItem(
    STORAGE_KEY,
    JSON.stringify({
      sortKey: "identity",
      sortDir: "asc",
      density: "compact",
      statusFilter: "all",
    }),
  );
  const loaded = loadWorkbenchPrefs(store);
  assert.equal(loaded.summaryScope, defaultWorkbenchPrefs.summaryScope);
});

test("saveWorkbenchPrefs swallows setItem failures silently", () => {
  const failing = {
    getItem: () => null,
    setItem: () => {
      throw new Error("quota");
    },
  };
  assert.doesNotThrow(() =>
    saveWorkbenchPrefs(
      {
        sortKey: "identity",
        sortDir: "desc",
        density: "compact",
        statusFilter: "ok",
        summaryScope: "filtered",
      },
      failing,
    ),
  );
});

test("saveWorkbenchPrefs ignores invalid inputs without writing", () => {
  const store = makeStore();
  saveWorkbenchPrefs(null, store);
  saveWorkbenchPrefs(undefined, store);
  saveWorkbenchPrefs("nope", store);
  assert.equal(store.getItem(STORAGE_KEY), null);
});

test("loadWorkbenchPrefs falls back when storage.getItem throws", () => {
  const failing = {
    getItem: () => {
      throw new Error("denied");
    },
    setItem: () => {},
  };
  assert.deepEqual(loadWorkbenchPrefs(failing), defaultWorkbenchPrefs);
});
