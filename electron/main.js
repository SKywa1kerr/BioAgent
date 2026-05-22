const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const path = require("path");
const fs = require("fs");
const { pathToFileURL } = require("url");
const { autoUpdater } = require("electron-updater");

let mainWindow = null;
let agentHarness = null;

const INIT_TIMEOUT_MS = 30000;
const MAX_DEBUG_ENTRIES = 4000;
const debugEntries = [];
let flushTimer = null;

function createWindow() {
  const isMac = process.platform === "darwin";
  const chromeOptions = isMac
    ? { titleBarStyle: "hiddenInset", trafficLightPosition: { x: 16, y: 12 } }
    : { frame: false };

  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1100,
    minHeight: 760,
    ...chromeOptions,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  // Broadcast maximize state changes so the renderer can keep its
  // titlebar maximize/restore icon in sync without polling.
  const broadcastWindowState = () => {
    if (!mainWindow || mainWindow.isDestroyed()) return;
    mainWindow.webContents.send("window-state", {
      isMaximized: mainWindow.isMaximized(),
    });
  };
  mainWindow.on("maximize", broadcastWindowState);
  mainWindow.on("unmaximize", broadcastWindowState);

  if (app.isPackaged) {
    mainWindow.loadFile(path.join(__dirname, "..", "dist", "index.html"));
  } else {
    mainWindow.loadURL("http://127.0.0.1:1420");
  }
}

function getPythonConfig() {
  const persistRoot = path.join(app.getPath("userData"), "bioagent");
  fs.mkdirSync(persistRoot, { recursive: true });
  const env = {
    ...process.env,
    BIOAGENT_DATA_DIR: persistRoot,
    // Force UTF-8 on Python stdio so JSON over MCP carries Chinese characters
    // verbatim. Without this, Windows defaults to cp936/gbk and the renderer
    // shows garbled trends/suggestions text.
    PYTHONIOENCODING: "utf-8",
    PYTHONUTF8: "1",
  };

  if (app.isPackaged) {
    // Production: spawn the standalone PyInstaller sidecar that ships
    // alongside the Electron binary. extraResources in package.json drops
    // it under <resources>/sidecar/bioagent-sidecar/.
    const sidecarDir = path.join(process.resourcesPath, "sidecar", "bioagent-sidecar");
    const exeName = process.platform === "win32" ? "bioagent-sidecar.exe" : "bioagent-sidecar";
    const cmd = path.join(sidecarDir, exeName);
    return { cmd, cwd: sidecarDir, baseArgs: [], env };
  }

  // Dev: fall back to the user's system Python with our source on PYTHONPATH.
  const cmd = process.platform === "win32" ? "python" : "python3";
  const cwd = path.resolve(__dirname, "..");
  const baseArgs = ["-m", "bioagent.main"];
  env.PYTHONPATH = path.resolve(__dirname, "../src-python");
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

const AUTO_FLUSH_INTERVAL_MS = 5 * 60 * 1000; // 5 minutes
const AUTO_FLUSH_MIN_ENTRIES = 50;
let lastFlushedCount = 0;

function autoFlushDebugLog() {
  if (debugEntries.length <= lastFlushedCount || debugEntries.length < AUTO_FLUSH_MIN_ENTRIES) return;
  try {
    const root = path.resolve(__dirname, "..", "outputs", "debug");
    fs.mkdirSync(root, { recursive: true });
    const target = path.join(root, "debug-latest.log");
    const lines = debugEntries.map((entry) => JSON.stringify(entry));
    fs.writeFileSync(target, lines.join("\n"), "utf8");
    lastFlushedCount = debugEntries.length;
  } catch {
    // ignore flush errors
  }
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

function setupAutoUpdater(mainWindow) {
  // No-op in dev — autoUpdater requires a packaged app.
  if (!app.isPackaged) return;

  autoUpdater.autoDownload = true;
  autoUpdater.autoInstallOnAppQuit = true;

  function broadcast(channel, payload) {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send(channel, payload);
    }
  }

  autoUpdater.on("checking-for-update", () => broadcast("updater-state", { phase: "checking" }));
  autoUpdater.on("update-available", (info) => broadcast("updater-state", { phase: "available", version: info.version }));
  autoUpdater.on("update-not-available", () => broadcast("updater-state", { phase: "up-to-date" }));
  autoUpdater.on("download-progress", (progress) => broadcast("updater-state", {
    phase: "downloading",
    percent: Math.round(progress.percent),
    bytesPerSecond: progress.bytesPerSecond,
  }));
  autoUpdater.on("update-downloaded", (info) => broadcast("updater-state", { phase: "ready", version: info.version }));
  autoUpdater.on("error", (err) => broadcast("updater-state", { phase: "error", message: err?.message ?? String(err) }));

  // Kick off the first check 5s after launch so the renderer is mounted.
  setTimeout(() => { autoUpdater.checkForUpdatesAndNotify().catch(() => {}); }, 5000);
}

app.whenReady().then(async () => {
  createWindow();
  flushTimer = setInterval(autoFlushDebugLog, AUTO_FLUSH_INTERVAL_MS);
  setupAutoUpdater(mainWindow);

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

  ipcMain.handle("window-minimize", (event) => {
    const win = BrowserWindow.fromWebContents(event.sender);
    if (win && !win.isDestroyed()) win.minimize();
    return { ok: true };
  });

  ipcMain.handle("window-maximize-toggle", (event) => {
    const win = BrowserWindow.fromWebContents(event.sender);
    if (!win || win.isDestroyed()) return { isMaximized: false };
    if (win.isMaximized()) {
      win.unmaximize();
    } else {
      win.maximize();
    }
    return { isMaximized: win.isMaximized() };
  });

  ipcMain.handle("window-close", (event) => {
    const win = BrowserWindow.fromWebContents(event.sender);
    if (win && !win.isDestroyed()) win.close();
    return { ok: true };
  });

  ipcMain.handle("updater-quit-and-install", async (event) => {
    const trace = [];
    pushLifecycle(trace, event.sender, "run", "Updater: quit-and-install requested");
    try {
      autoUpdater.quitAndInstall();
      return { ok: true, trace };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      pushLifecycle(trace, event.sender, "error", `Updater quit-and-install failed: ${message}`);
      return { ok: false, error: message, trace };
    }
  });

  ipcMain.handle("export-save-file", async (_event, payload) => {
    if (!mainWindow) return { canceled: true, error: "no-window" };
    const { defaultPath, filters, data, encoding } = payload || {};
    try {
      const result = await dialog.showSaveDialog(mainWindow, { defaultPath, filters });
      if (result.canceled || !result.filePath) return { canceled: true };
      const buffer = encoding === "base64"
        ? Buffer.from(String(data ?? ""), "base64")
        : Buffer.from(String(data ?? ""), "utf8");
      fs.writeFileSync(result.filePath, buffer);
      return { canceled: false, filePath: result.filePath };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return { canceled: true, error: message };
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
        if (settings?.language) agentHarness.setLanguage(settings.language);
        pushLifecycle(trace, event.sender, "ready", `Agent ready. Tools loaded: ${tools.length}`);
        return { ok: true, tools, trace };
      }

      if (settings?.language) agentHarness.setLanguage(settings.language);
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

  ipcMain.handle("agent-harness-run", async (event, userMessage, runtimeSettings) => {
    const trace = [];

    if (!agentHarness) {
      const tools = [];
      pushLifecycle(trace, event.sender, "error", "Run rejected: harness not initialized");
      return { ok: false, error: "Agent harness not initialized", tools, trace };
    }

    if (runtimeSettings?.language) agentHarness.setLanguage(runtimeSettings.language);

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

  ipcMain.handle("agent-harness-get-analysis-bundle", async (event, analysisId) => {
    const trace = [];

    if (!agentHarness) {
      pushLifecycle(trace, event.sender, "error", "Bundle request rejected: harness not initialized");
      return { ok: false, error: "Agent harness not initialized", trace };
    }

    try {
      const detail = await agentHarness.callMcpTool("get_analysis_detail", { analysis_id: analysisId });
      if (detail?.ok === false) {
        const msg = detail?.error || "Failed to fetch analysis detail";
        pushLifecycle(trace, event.sender, "error", msg);
        return { ok: false, error: msg, trace };
      }

      let trends = detail?.trends ?? null;
      let suggestions = detail?.suggestions ?? null;

      if (!trends) {
        const trendsResult = await agentHarness.callMcpTool("detect_mutation_trends", { analysis_id: analysisId });
        if (trendsResult?.ok !== false) trends = trendsResult;
      }
      if (!suggestions) {
        const suggestionsResult = await agentHarness.callMcpTool("generate_lab_suggestions", { analysis_id: analysisId });
        if (suggestionsResult?.ok !== false) suggestions = suggestionsResult;
      }

      pushLifecycle(trace, event.sender, "run", `Loaded analysis bundle: ${analysisId}`);
      return { ok: true, detail, trends, suggestions, trace };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      pushLifecycle(trace, event.sender, "error", `Bundle fetch failed: ${message}`);
      return { ok: false, error: message, trace };
    }
  });

  ipcMain.handle("datasets-list", async (event) => {
    const trace = [];
    if (!agentHarness) {
      pushLifecycle(trace, event.sender, "error", "Datasets list rejected: harness not initialized");
      return { ok: false, error: "Agent harness not initialized", trace };
    }
    try {
      const result = await agentHarness.callMcpTool("list_datasets", {});
      if (result?.ok === false) {
        return { ok: false, error: result?.error || "list_datasets failed", trace };
      }
      return { ok: true, builtin: result?.builtin || [], user: result?.user || [], trace };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return { ok: false, error: message, trace };
    }
  });

  ipcMain.handle("dataset-register", async (event, payload) => {
    const trace = [];
    if (!agentHarness) {
      pushLifecycle(trace, event.sender, "error", "Dataset register rejected: harness not initialized");
      return { ok: false, error: "Agent harness not initialized", trace };
    }
    try {
      const result = await agentHarness.callMcpTool("register_dataset", {
        label: payload?.label,
        ab1_dir: payload?.ab1Dir,
        gb_dir: payload?.gbDir,
      });
      if (result?.ok === false) {
        return { ok: false, error: result?.error || "register_dataset failed", trace };
      }
      await agentHarness.refreshDatasetList();
      pushLifecycle(trace, event.sender, "run", `Registered dataset: ${result?.id}`);
      return { ok: true, dataset: result, trace };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return { ok: false, error: message, trace };
    }
  });

  ipcMain.handle("dataset-delete", async (event, id) => {
    const trace = [];
    if (!agentHarness) {
      return { ok: false, error: "Agent harness not initialized", trace };
    }
    try {
      const result = await agentHarness.callMcpTool("delete_dataset", { id });
      if (result?.ok === false) {
        return { ok: false, error: result?.error || "delete_dataset failed", trace };
      }
      await agentHarness.refreshDatasetList();
      pushLifecycle(trace, event.sender, "run", `Deleted dataset: ${id}`);
      return { ok: true, trace };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return { ok: false, error: message, trace };
    }
  });

  ipcMain.handle("dialog-pick-folder", async (_event, options) => {
    if (!mainWindow) return { canceled: true, error: "no-window" };
    try {
      const result = await dialog.showOpenDialog(mainWindow, {
        title: options?.title,
        defaultPath: options?.defaultPath,
        properties: ["openDirectory"],
      });
      if (result.canceled || result.filePaths.length === 0) return { canceled: true };
      return { canceled: false, path: result.filePaths[0] };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return { canceled: true, error: message };
    }
  });

  ipcMain.handle("inspect-dropped-paths", async (_event, paths) => {
    const dirs = [];
    const files = [];
    for (const p of Array.isArray(paths) ? paths : []) {
      try {
        const stat = fs.statSync(p);
        if (stat.isDirectory()) dirs.push(p);
        else if (stat.isFile()) files.push(p);
      } catch {
        // ignore
      }
    }
    return { ok: true, dirs, files };
  });

  ipcMain.handle("inspect-dataset-folder", async (_event, folderPath) => {
    try {
      if (!folderPath || !fs.existsSync(folderPath)) {
        return { ok: false, error: "not found" };
      }
      const stat = fs.statSync(folderPath);
      if (!stat.isDirectory()) return { ok: false, error: "not a directory" };

      const entries = fs.readdirSync(folderPath, { withFileTypes: true });
      const subdirs = entries.filter((e) => e.isDirectory());
      const ab1Sub = subdirs.find((d) => /^ab1$|^ab1_files$/i.test(d.name));
      const gbSub = subdirs.find((d) => /^gb$|^gbk$|^genbank$/i.test(d.name));

      if (ab1Sub && gbSub) {
        return {
          ok: true,
          layout: "subdirs",
          ab1Dir: path.join(folderPath, ab1Sub.name),
          gbDir: path.join(folderPath, gbSub.name),
        };
      }

      const files = entries.filter((e) => e.isFile());
      const hasAb1 = files.some((f) => /\.ab1$/i.test(f.name));
      const hasGb = files.some((f) => /\.gbk?$/i.test(f.name));
      if (hasAb1 && hasGb) {
        return { ok: true, layout: "flat", ab1Dir: folderPath, gbDir: folderPath };
      }
      // Couldn't auto-detect — return root for both, caller can prefill the
      // dialog and let the user fix.
      return {
        ok: true,
        layout: "ambiguous",
        ab1Dir: folderPath,
        gbDir: folderPath,
        hasAb1,
        hasGb,
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return { ok: false, error: message };
    }
  });

  ipcMain.handle("agent-query-history", async (event, args) => {
    const trace = [];

    if (!agentHarness) {
      pushLifecycle(trace, event.sender, "error", "History request rejected: harness not initialized");
      return { ok: false, error: "Agent harness not initialized", trace };
    }

    try {
      const result = await agentHarness.callMcpTool("query_history", args || { limit: 20 });
      if (result?.ok === false) {
        const msg = result?.error || "Failed to query history";
        pushLifecycle(trace, event.sender, "error", msg);
        return { ok: false, error: msg, trace };
      }
      return { ok: true, result, trace };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      pushLifecycle(trace, event.sender, "error", `History query failed: ${message}`);
      return { ok: false, error: message, trace };
    }
  });

  ipcMain.handle("analyze-dropped-files", async (event, payload) => {
    const trace = [];

    if (!agentHarness) {
      pushLifecycle(trace, event.sender, "error", "Drop analyze rejected: harness not initialized");
      return { ok: false, error: "Agent harness not initialized", trace };
    }

    const ab1Paths = Array.isArray(payload?.ab1Paths) ? payload.ab1Paths : [];
    const gbPaths = Array.isArray(payload?.gbPaths) ? payload.gbPaths : [];

    let pairs = [];
    let unpairedAb1 = [];
    let unpairedGb = [];
    try {
      const helperUrl = pathToFileURL(
        path.resolve(__dirname, "..", "src", "lib", "files", "pairAb1Gb.js"),
      ).href;
      const { pairAb1Gb } = await import(helperUrl);
      const pairResult = pairAb1Gb(ab1Paths, gbPaths);
      pairs = pairResult.pairs;
      unpairedAb1 = pairResult.unpairedAb1;
      unpairedGb = pairResult.unpairedGb;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      pushLifecycle(trace, event.sender, "error", `Drop analyze pairing failed: ${message}`);
      return { ok: false, error: message, trace };
    }

    if (pairs.length === 0) {
      pushLifecycle(trace, event.sender, "error", "Drop analyze: no matching .ab1/.gb pairs found");
      return {
        ok: false,
        error: "No matching .ab1/.gb pairs found",
        pairs: 0,
        unpairedAb1,
        unpairedGb,
        trace,
      };
    }

    const ab1Dir = path.dirname(pairs[0].ab1);
    const gbDir = path.dirname(pairs[0].gb);
    const sameDirs = pairs.every(
      (pair) => path.dirname(pair.ab1) === ab1Dir && path.dirname(pair.gb) === gbDir,
    );
    if (!sameDirs) {
      pushLifecycle(trace, event.sender, "error", "Drop analyze: paired files span multiple directories");
      return {
        ok: false,
        error: "Dropped files span multiple directories",
        pairs: pairs.length,
        unpairedAb1,
        unpairedGb,
        trace,
      };
    }

    pushLifecycle(
      trace,
      event.sender,
      "run",
      `Drop analyze: ${pairs.length} pair(s) → analyze_sequences`,
      { ab1Dir, gbDir },
    );

    try {
      const result = await agentHarness.callMcpTool("analyze_sequences", {
        ab1_dir: ab1Dir,
        gb_dir: gbDir,
      });
      if (result?.ok === false) {
        const msg = result?.error || "analyze_sequences failed";
        pushLifecycle(trace, event.sender, "error", msg);
        return { ok: false, error: msg, pairs: pairs.length, unpairedAb1, unpairedGb, trace };
      }
      pushLifecycle(
        trace,
        event.sender,
        "run",
        `Drop analyze complete (${pairs.length} pair(s))`,
      );
      return {
        ok: true,
        result,
        pairs: pairs.length,
        unpairedAb1,
        unpairedGb,
        trace,
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      pushLifecycle(trace, event.sender, "error", `Drop analyze failed: ${message}`);
      return { ok: false, error: message, pairs: pairs.length, unpairedAb1, unpairedGb, trace };
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
  if (flushTimer) clearInterval(flushTimer);
  autoFlushDebugLog();
  if (agentHarness) {
    agentHarness.shutdown();
    agentHarness = null;
  }
  if (process.platform !== "darwin") {
    app.quit();
  }
});
