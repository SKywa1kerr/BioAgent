import test from "node:test";
import assert from "node:assert/strict";
import {
  registerCommand,
  getCommands,
  filterCommands,
  clearCommands,
} from "../src/lib/commands/registry.js";

function makeCmd(id, overrides = {}) {
  return {
    id,
    title: overrides.title ?? id,
    group: overrides.group ?? "nav",
    keywords: overrides.keywords,
    when: overrides.when,
    run: overrides.run ?? (() => {}),
  };
}

test("registerCommand stores commands in insertion order", () => {
  clearCommands();
  registerCommand(makeCmd("a"));
  registerCommand(makeCmd("b"));
  assert.deepEqual(getCommands().map((c) => c.id), ["a", "b"]);
});

test("registerCommand returns an unregister function", () => {
  clearCommands();
  const off = registerCommand(makeCmd("a"));
  off();
  assert.equal(getCommands().length, 0);
});

test("registerCommand with duplicate id replaces the earlier entry", () => {
  clearCommands();
  registerCommand(makeCmd("a", { title: "first" }));
  registerCommand(makeCmd("a", { title: "second" }));
  const cmds = getCommands();
  assert.equal(cmds.length, 1);
  assert.equal(cmds[0].title, "second");
});

test("filterCommands hides entries whose when() returns false", () => {
  clearCommands();
  registerCommand(makeCmd("a", { when: () => true }));
  registerCommand(makeCmd("b", { when: () => false }));
  const ids = filterCommands("").map((c) => c.id);
  assert.deepEqual(ids, ["a"]);
});

test("clearCommands empties the registry", () => {
  registerCommand(makeCmd("a"));
  clearCommands();
  assert.equal(getCommands().length, 0);
});
