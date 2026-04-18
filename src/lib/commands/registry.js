import { fuzzyScore } from "./fuzzy.js";

const commands = new Map();

export function registerCommand(cmd) {
  if (!cmd || typeof cmd.id !== "string" || typeof cmd.run !== "function") {
    throw new Error("registerCommand: id and run are required");
  }
  commands.set(cmd.id, cmd);
  return function unregister() {
    const existing = commands.get(cmd.id);
    if (existing === cmd) commands.delete(cmd.id);
  };
}

export function getCommands() {
  return Array.from(commands.values());
}

export function filterCommands(query) {
  const visible = getCommands().filter((c) => (typeof c.when === "function" ? c.when() : true));
  const q = (query ?? "").trim();
  if (!q) return visible;
  const scored = [];
  for (const cmd of visible) {
    const hay = [cmd.title, ...(cmd.keywords ?? [])].join(" ");
    const s = fuzzyScore(hay, q);
    if (s >= 0) scored.push({ cmd, s });
  }
  scored.sort((a, b) => b.s - a.s);
  return scored.map((x) => x.cmd);
}

export function clearCommands() {
  commands.clear();
}
