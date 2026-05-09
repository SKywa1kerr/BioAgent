import test from "node:test";
import assert from "node:assert/strict";
import {
  computeNextWidth,
  loadChatWidthState,
  saveChatWidthState,
  SPLITTER_CONSTANTS,
} from "../src/lib/ui/chatColumnWidth.js";

test("computeNextWidth keeps width inside bounds", () => {
  const r = computeNextWidth(400, 1200, 360);
  assert.equal(r.width, 400);
  assert.equal(r.collapsed, false);
});

test("computeNextWidth collapses below threshold", () => {
  const r = computeNextWidth(80, 1200, 360);
  assert.equal(r.collapsed, true);
  assert.equal(r.width, SPLITTER_CONSTANTS.RAIL_WIDTH);
});

test("computeNextWidth clamps to container max minus canvas min", () => {
  const r = computeNextWidth(2000, 1000, 360);
  assert.equal(r.width, 640);
  assert.equal(r.collapsed, false);
});

test("computeNextWidth threshold edge — exactly threshold expands", () => {
  const r = computeNextWidth(SPLITTER_CONSTANTS.COLLAPSE_THRESHOLD, 1200, 360);
  assert.equal(r.collapsed, false);
  assert.equal(r.width, SPLITTER_CONSTANTS.COLLAPSE_THRESHOLD);
});

test("computeNextWidth NaN falls back to default", () => {
  const r = computeNextWidth(NaN, 1200, 360);
  assert.equal(r.width, SPLITTER_CONSTANTS.DEFAULT_WIDTH);
  assert.equal(r.collapsed, false);
});

test("load returns defaults when storage empty", () => {
  const fake = { getItem: () => null, setItem: () => {} };
  const s = loadChatWidthState(fake);
  assert.equal(s.width, SPLITTER_CONSTANTS.DEFAULT_WIDTH);
  assert.equal(s.collapsed, false);
  assert.equal(s.lastExpandedWidth, SPLITTER_CONSTANTS.DEFAULT_WIDTH);
});

test("save then load round-trips", () => {
  const store = {};
  const fake = { getItem: (k) => store[k] ?? null, setItem: (k, v) => { store[k] = v; } };
  saveChatWidthState({ width: 32, collapsed: true, lastExpandedWidth: 420 }, fake);
  const s = loadChatWidthState(fake);
  assert.equal(s.width, 32);
  assert.equal(s.collapsed, true);
  assert.equal(s.lastExpandedWidth, 420);
});

test("load malformed JSON falls back to defaults", () => {
  const fake = { getItem: () => "not-json", setItem: () => {} };
  const s = loadChatWidthState(fake);
  assert.equal(s.width, SPLITTER_CONSTANTS.DEFAULT_WIDTH);
  assert.equal(s.collapsed, false);
});
