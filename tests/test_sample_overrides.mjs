import test from "node:test";
import assert from "node:assert/strict";
import {
  loadSampleOverrides,
  saveSampleOverrides,
  getOverrideKey,
} from "../src/lib/ui/sampleOverrides.js";

const STORAGE_KEY = "bioagent.sample.overrides.v1";

function makeStore() {
  const data = new Map();
  return {
    getItem: (k) => (data.has(k) ? data.get(k) : null),
    setItem: (k, v) => data.set(k, v),
    removeItem: (k) => data.delete(k),
    _data: data,
  };
}

test("loadSampleOverrides returns empty map when storage is empty", () => {
  const store = makeStore();
  assert.deepEqual(loadSampleOverrides(store), {});
});

test("loadSampleOverrides returns empty map on malformed JSON", () => {
  const store = makeStore();
  store.setItem(STORAGE_KEY, "{not json");
  assert.deepEqual(loadSampleOverrides(store), {});
});

test("loadSampleOverrides drops entries with unknown status", () => {
  const store = makeStore();
  store.setItem(
    STORAGE_KEY,
    JSON.stringify({
      "a::s1": { status: "ok", reason: "looks fine", updatedAt: "2026-05-06T00:00:00Z" },
      "a::s2": { status: "bogus", reason: "x", updatedAt: "2026-05-06T00:00:00Z" },
      "a::s3": { status: "untested", reason: "x", updatedAt: "2026-05-06T00:00:00Z" },
    }),
  );
  const loaded = loadSampleOverrides(store);
  assert.deepEqual(Object.keys(loaded).sort(), ["a::s1"]);
  assert.equal(loaded["a::s1"].status, "ok");
});

test("loadSampleOverrides drops entries with missing analysisId or sampleId", () => {
  const store = makeStore();
  store.setItem(
    STORAGE_KEY,
    JSON.stringify({
      "::s1": { status: "ok", reason: "", updatedAt: "2026-05-06T00:00:00Z" },
      "a::": { status: "ok", reason: "", updatedAt: "2026-05-06T00:00:00Z" },
      "no-separator": { status: "ok", reason: "", updatedAt: "2026-05-06T00:00:00Z" },
      "a::s1": { status: "ok", reason: "", updatedAt: "2026-05-06T00:00:00Z" },
    }),
  );
  const loaded = loadSampleOverrides(store);
  assert.deepEqual(Object.keys(loaded), ["a::s1"]);
});

test("loadSampleOverrides coerces missing reason / updatedAt to safe defaults", () => {
  const store = makeStore();
  store.setItem(
    STORAGE_KEY,
    JSON.stringify({ "a::s1": { status: "wrong" } }),
  );
  const loaded = loadSampleOverrides(store);
  assert.equal(loaded["a::s1"].status, "wrong");
  assert.equal(loaded["a::s1"].reason, "");
  assert.equal(typeof loaded["a::s1"].updatedAt, "string");
});

test("save then load round-trips overrides", () => {
  const store = makeStore();
  const next = {
    "ana1::s1": { status: "ok", reason: "manual review", updatedAt: "2026-05-06T01:00:00Z" },
    "ana1::s2": { status: "wrong", reason: "bad signal", updatedAt: "2026-05-06T01:00:00Z" },
  };
  saveSampleOverrides(next, store);
  assert.deepEqual(loadSampleOverrides(store), next);
});

test("saveSampleOverrides drops invalid entries before writing", () => {
  const store = makeStore();
  saveSampleOverrides(
    {
      "ana1::s1": { status: "ok", reason: "", updatedAt: "2026-05-06T00:00:00Z" },
      "ana1::s2": { status: "weird", reason: "", updatedAt: "2026-05-06T00:00:00Z" },
    },
    store,
  );
  const loaded = loadSampleOverrides(store);
  assert.deepEqual(Object.keys(loaded), ["ana1::s1"]);
});

test("saveSampleOverrides swallows setItem failures silently", () => {
  const failing = {
    getItem: () => null,
    setItem: () => {
      throw new Error("quota");
    },
  };
  assert.doesNotThrow(() =>
    saveSampleOverrides(
      { "a::s1": { status: "ok", reason: "", updatedAt: "2026-05-06T00:00:00Z" } },
      failing,
    ),
  );
});

test("loadSampleOverrides falls back when storage.getItem throws", () => {
  const failing = {
    getItem: () => {
      throw new Error("denied");
    },
    setItem: () => {},
  };
  assert.deepEqual(loadSampleOverrides(failing), {});
});

test("getOverrideKey produces stable keys for same input", () => {
  assert.equal(getOverrideKey("ana1", "s1"), "ana1::s1");
  assert.equal(getOverrideKey("ana1", "s1"), getOverrideKey("ana1", "s1"));
  assert.notEqual(getOverrideKey("ana1", "s1"), getOverrideKey("ana2", "s1"));
});
