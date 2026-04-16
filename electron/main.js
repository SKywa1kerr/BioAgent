const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");
const fs = require("fs");
const { pathToFileURL } = require("url");

let mainWindow = null;
let agentHarness = null;

const INIT_TIMEOUT_MS = 30000;
const MAX_DEBUG_ENTRIES = 4000;
const debugEntries = [];

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1100,
    minHeight: 760,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  if (app.isPackaged) {
    mainWindow.loadFile(path.join(__dirname, "..", "dist", "index.html"));
  } else {
    mainWindow.loadURL("http://127.0.0.1:1420");
  }
}

function getPythonConfig() {
  const cmd = process.platform === "win32" ? "python" : "python3";
  const cwd = path.resolve(__dirname, "..");
  const baseArgs = ["-m", "bioagent.main"];
  const env = {
    ...process.env,
    PYTHONPATH: path.resolve(__dirname, "../src-python"),
  };
  return { cmd, cwd, baseArgs, env };
}

function appendDebug(entry) {
  debugEntries.push({ timestamp: new Date().toISOString(), ...entry });
  if (debugEntries.length > MAX_DEBUG_ENTRIES) {
    debugEntries.splice(0, debugEntries.length - MAX_DEBUG_ENTRIES);
  }
}

function exportDebugLog() {
  const root = path.resolve(__dirname, "..", "outputs", "debug");
  fs.mkdirSync(root, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const target = path.join(root, `debug-${stamp}.log`);
  const lines = debugEntries.map((entry) => JSON.stringify(entry));
  fs.writeFileSync(target, lines.join("\n"), "utf8");
  return target;
}

function listFilesSafe(dirPath) {
  try {
    if (!fs.existsSync(dirPath)) return [];
    return fs.readdirSync(dirPath, { withFileTypes: true }).filter((d) => d.isFile()).map((d) => d.name);
  } catch {
    return [];
  }
}

function checkDatasetLayout() {
  const root = path.resolve(__dirname, "..", "data");
  const datasets = ["base", "pro", "promax"];

  const result = datasets.map((name) => {
    const ds = path.join(root, name);
    const ab1Dir = path.join(ds, "ab1");
    const gbDir = path.join(ds, "gb");
    const ab1Files = listFilesSafe(ab1Dir).filter((f) => f.toLowerCase().endsWith(".ab1"));
    const gbFiles = listFilesSafe(gbDir).filter((f) => f.toLowerCase().endsWith(".gb") || f.toLowerCase().endsWith(".gbk"));
    return {
      dataset: name,
      path: ds,
      exists: fs.existsSync(ds),
      ab1Dir,
      gbDir,
      ab1Count: ab1Files.length,
      gbCount: gbFiles.length,
      ok: fs.existsSync(ab1Dir) && fs.existsSync(gbDir) && ab1Files.length > 0 && gbFiles.length > 0,
    };
  });

  return {
    root,
    datasets: result,
    allOk: result.every((x) => x.ok),
  };
}

function withTimeout(promise, timeoutMs, label) {
  let timeoutId;
  const timeout = new Promise((_, reject) => {
    timeoutId = setTimeout(() => reject(new Error(`${label} timed out after ${timeoutMs}ms`)), timeoutMs);
  });

  return Promise.race([promise, timeout]).finally(() => {
    clearTimeout(timeoutId);
  });
}

function makeLifecycle(phase, message, extra = {}) {
  return {
    type: "lifecycle",
    phase,
    message,
    timestamp: new Date().toISOString(),
    ...extra,
  };
}

function emitLifecycle(sender, entry) {
  if (!sender || sender.isDestroyed()) return;
  sender.send("agent-event", entry);
}

function pushLifecycle(trace, sender, phase, message, extra = {}) {
  const entry = makeLifecycle(phase, message, extra);
  trace.push(entry);
  appendDebug({ kind: "lifecycle", entry });
  emitLifecycle(sender, entry);
  return entry;
}

app.whenReady().then(async () => {
  createWindow();

  const mod = await import(pathToFileURL(path.join(__dirname, "agent_harness.mjs")).href);
  const AgentHarness = mod.AgentHarness;

  ipcMain.handle("agent-export-debug-log", async () => {
    try {
      const filePath = exportDebugLog();
      return { ok: true, filePath, entries: debugEntries.length };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return { ok: false, error: message };
    }
  });

  ipcMain.handle("agent-harness-init", async (event, settings) => {
    const trace = [];
    pushLifecycle(trace, event.sender, "init", "Init request received");

    try {
      if (!agentHarness) {
        pushLifecycle(trace, event.sender, "init", "Creating harness instance");
        agentHarness = new AgentHarness({
          ...settings,
          pythonConfig: getPythonConfig(),
        });

        pushLifecycle(trace, event.sender, "init", "Starting MCP server");
        const tools = await withTimeout(agentHarness.initMcpServer(), INIT_TIMEOUT_MS, "MCP init");
        pushLifecycle(trace, event.sender, "ready", `Agent ready. Tools loaded: ${tools.length}`);
        return { ok: true, tools, trace };
      }

      pushLifecycle(trace, event.sender, "ready", "Agent already initialized");
      return { ok: true, tools: agentHarness.mcpTools || [], trace };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      pushLifecycle(trace, event.sender, "error", `Init failed: ${message}`);
      return { ok: false, error: message, tools: [], trace };
    }
  });

  ipcMain.handle("dataset-precheck", async (event) => {
    const trace = [];
    try {
      const report = checkDatasetLayout();
      const summary = report.datasets
        .map((d) => `${d.dataset}: ab1=${d.ab1Count}, gb=${d.gbCount}, ok=${d.ok}`)
        .join(" | ");
      pushLifecycle(trace, event.sender, "run", `Dataset precheck: ${summary}`);
      return { ok: true, report, trace };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      pushLifecycle(trace, event.sender, "error", `Dataset precheck failed: ${message}`);
      return { ok: false, error: message, trace };
    }
  });

  ipcMain.handle("agent-harness-run", async (event, userMessage) => {
    const trace = [];

    if (!agentHarness) {
      const tools = [];
      pushLifecycle(trace, event.sender, "error", "Run rejected: harness not initialized");
      return { ok: false, error: "Agent harness not initialized", tools, trace };
    }

    const events = [];
    pushLifecycle(trace, event.sender, "run", "Run started");

    try {
      await agentHarness.runTurn(userMessage, (payload) => {
        events.push(payload);
        appendDebug({ kind: "agent-event", payload });
        if (!event.sender.isDestroyed()) {
          event.sender.send("agent-event", payload);
        }
      });
      pushLifecycle(trace, event.sender, "run", `Run finished with ${events.length} event(s)`);
      return { ok: true, events, trace };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      pushLifecycle(trace, event.sender, "error", `Run failed: ${message}`);
      return { ok: false, error: message, events, trace };
    }
  });

  ipcMain.handle("agent-harness-get-analysis-detail", async (event, analysisId) => {
    const trace = [];

    if (!agentHarness) {
      pushLifecycle(trace, event.sender, "error", "Detail request rejected: harness not initialized");
      return { ok: false, error: "Agent harness not initialized", trace };
    }

    try {
      const detail = await agentHarness.callMcpTool("get_analysis_detail", { analysis_id: analysisId });
      if (detail?.ok === false) {
        const msg = detail?.error || "Failed to fetch analysis detail";
        pushLifecycle(trace, event.sender, "error", msg);
        return { ok: false, error: msg, trace };
      }
      pushLifecycle(trace, event.sender, "run", `Loaded analysis detail: ${analysisId}`);
      return { ok: true, detail, trace };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      pushLifecycle(trace, event.sender, "error", `Detail fetch failed: ${message}`);
      return { ok: false, error: message, trace };
    }
  });

  ipcMain.handle("agent-harness-shutdown", async (event) => {
    if (agentHarness) {
      const entry = makeLifecycle("shutdown", "Shutting down harness");
      appendDebug({ kind: "lifecycle", entry });
      emitLifecycle(event.sender, entry);
      agentHarness.shutdown();
      agentHarness = null;
    }
    return { ok: true };
  });
});

app.on("window-all-closed", () => {
  if (agentHarness) {
    agentHarness.shutdown();
    agentHarness = null;
  }
  if (process.platform !== "darwin") {
    app.quit();
  }
});
