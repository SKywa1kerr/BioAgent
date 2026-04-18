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
  return visible;
}

export function clearCommands() {
  commands.clear();
}
