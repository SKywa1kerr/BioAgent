import test from "node:test";
import assert from "node:assert/strict";
import {
  toastReducer,
  initialToastState,
  pushToast,
  dismissToast,
  clearAllToasts,
} from "../src/lib/ui/toastReducer.js";

test("initial state has no toasts", () => {
  assert.deepEqual(initialToastState, { toasts: [], nextId: 1 });
});

test("pushToast adds a toast with auto-assigned id and default duration", () => {
  const state = toastReducer(initialToastState, pushToast({ kind: "info", title: "Hello" }));
  assert.equal(state.toasts.length, 1);
  assert.equal(state.toasts[0].id, "toast-1");
  assert.equal(state.toasts[0].title, "Hello");
  assert.equal(state.toasts[0].kind, "info");
  assert.equal(state.toasts[0].durationMs, 5000);
  assert.equal(state.nextId, 2);
});

test("pushToast preserves explicit duration and description", () => {
  const state = toastReducer(initialToastState, pushToast({
    kind: "error",
    title: "Boom",
    description: "Stack...",
    durationMs: 0,
  }));
  assert.equal(state.toasts[0].durationMs, 0);
  assert.equal(state.toasts[0].description, "Stack...");
});

test("pushToast appends to the tail and caps at MAX_VISIBLE (5) by dropping oldest", () => {
  let state = initialToastState;
  for (let i = 0; i < 7; i += 1) {
    state = toastReducer(state, pushToast({ kind: "info", title: `t${i}` }));
  }
  assert.equal(state.toasts.length, 5);
  assert.deepEqual(state.toasts.map((t) => t.title), ["t2", "t3", "t4", "t5", "t6"]);
});

test("dismissToast removes by id and is a no-op for unknown ids", () => {
  let state = toastReducer(initialToastState, pushToast({ kind: "info", title: "a" }));
  state = toastReducer(state, pushToast({ kind: "info", title: "b" }));
  state = toastReducer(state, dismissToast("toast-1"));
  assert.equal(state.toasts.length, 1);
  assert.equal(state.toasts[0].id, "toast-2");
  state = toastReducer(state, dismissToast("nope"));
  assert.equal(state.toasts.length, 1);
});

test("clearAllToasts empties the queue without resetting nextId", () => {
  let state = toastReducer(initialToastState, pushToast({ kind: "info", title: "a" }));
  state = toastReducer(state, pushToast({ kind: "info", title: "b" }));
  const cleared = toastReducer(state, clearAllToasts());
  assert.equal(cleared.toasts.length, 0);
  assert.equal(cleared.nextId, 3);
});

test("pushToast carries through optional action descriptor", () => {
  const action = { label: "Retry", actionId: "retry-1" };
  const state = toastReducer(initialToastState, pushToast({ kind: "error", title: "x", action }));
  assert.deepEqual(state.toasts[0].action, action);
});
