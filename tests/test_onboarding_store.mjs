import test from "node:test";
import assert from "node:assert/strict";
import {
  readOnboarding,
  writeOnboarding,
  ONBOARDING_STORAGE_KEY,
} from "../src/hooks/onboardingStore.js";

function makeStore() {
  const data = new Map();
  return {
    getItem: (k) => (data.has(k) ? data.get(k) : null),
    setItem: (k, v) => data.set(k, v),
    removeItem: (k) => data.delete(k),
  };
}

test("readOnboarding returns false when storage empty", () => {
  assert.equal(readOnboarding(makeStore()), false);
});

test("writeOnboarding(true) then read returns true", () => {
  const s = makeStore();
  writeOnboarding(s, true);
  assert.equal(readOnboarding(s), true);
});

test("readOnboarding tolerates invalid values", () => {
  const s = makeStore();
  s.setItem(ONBOARDING_STORAGE_KEY, "garbage");
  assert.equal(readOnboarding(s), false);
});

test("writeOnboarding(false) removes the key", () => {
  const s = makeStore();
  writeOnboarding(s, true);
  writeOnboarding(s, false);
  assert.equal(s.getItem(ONBOARDING_STORAGE_KEY), null);
});
