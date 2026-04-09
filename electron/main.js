const { app, BrowserWindow, dialog, ipcMain, shell } = require("electron");
const path = require("path");
const fs = require("fs");
const { execFile, spawn } = require("child_process");

const isDev = !app.isPackaged;

// --------------- Python sidecar helpers ---------------

/**
 * Returns the command + args prefix to invoke the Python sidecar.
 * In dev: uses system python -m bioagent.main
 * In production: uses the PyInstaller-bundled executable
 */
function getPythonCommand() {
  if (isDev) {
    const pythonPath = process.platform === "win32" ? "py" : "python3";
    const pythonDir = path.join(__dirname, "../src-python");
    const baseArgs =
      process.platform === "win32" ? ["-3", "-m", "bioagent.main"] : ["-m", "bioagent.main"];
    return { cmd: pythonPath, baseArgs, cwd: pythonDir };
  }
  // Production: bundled sidecar next to the app
  const ext = process.platform === "win32" ? ".exe" : "";
  const sidecarDir = path.join(process.resourcesPath, "sidecar", "bioagent-sidecar");
  const sidecarExe = path.join(sidecarDir, `bioagent-sidecar${ext}`);
  return { cmd: sidecarExe, baseArgs: [], cwd: sidecarDir };
}

/**
 * Build environment variables for the Python subprocess.
 * Injects API key, base URL from saved settings, PYTHONPATH (dev only),
 * and the unified DB path.
 */
function getAnalysisEnv() {
  const env = { ...process.env };

  // In dev mode, set PYTHONPATH so python -m works
  if (isDev) {
    env.PYTHONPATH = path.join(__dirname, "../src-python");
  }

  // Load saved settings and inject as env vars
  try {
    const raw = fs.readFileSync(getSettingsPath(), "utf-8");
    const settings = JSON.parse(raw);
    if (settings.llmApiKey) env.LLM_API_KEY = settings.llmApiKey;
    if (settings.llmBaseUrl) env.LLM_BASE_URL = settings.llmBaseUrl;
    if (settings.llmModel) env.LLM_MODEL = settings.llmModel;
  } catch {
    // No settings file yet — that's fine
  }

  return env;
}

/** Unified DB path inside userData */
function getDbPath() {
  return path.join(app.getPath("userData"), "bioagent.db");
}

// --------------- Window ---------------

function createWindow() {
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  if (isDev) {
    win.loadURL("http://127.0.0.1:1420");
    win.webContents.openDevTools();
  } else {
    win.loadFile(path.join(__dirname, "../dist/index.html"));
  }
}

// --------------- IPC Handlers ---------------

ipcMain.handle("open-folder-dialog", async () => {
  const result = await dialog.showOpenDialog({
    properties: ["openDirectory"],
  });
  if (result.canceled || result.filePaths.length === 0) return null;
  return result.filePaths[0];
});

ipcMain.handle("inspect-dataset-directory", async (_event, folder) => {
  if (!folder) {
    return [];
  }

  return ["ab1", "gb"]
    .map((name) => path.join(folder, name))
    .filter((candidate) => {
      try {
        return fs.existsSync(candidate) && fs.statSync(candidate).isDirectory();
      } catch {
        return false;
      }
    });
});

ipcMain.handle("list-ab1-files", async (_event, folder) => {
  const entries = fs.readdirSync(folder);
  return entries
    .filter((f) => f.endsWith(".ab1"))
    .map((f) => path.join(folder, f));
});

ipcMain.handle("list-genbank-files", async (_event, folder) => {
  const entries = fs.readdirSync(folder);
  return entries
    .filter((f) => f.endsWith(".gb") || f.endsWith(".gbk"))
    .map((f) => path.join(folder, f));
});

ipcMain.handle("run-analysis", async (_event, ab1Dir, genesDir, options = {}) => {
  return new Promise((resolve, reject) => {
    const { cmd, baseArgs, cwd } = getPythonCommand();
    const args = [...baseArgs];
    const sendProgress = (payload) => {
      if (!_event.sender.isDestroyed()) {
        _event.sender.send("analysis-progress", payload);
      }
    };

    if (options.autoImport) {
      args.push("--auto-import");
    } else {
      args.push("--ab1-dir", ab1Dir);
    }

    if (genesDir) args.push("--genes-dir", genesDir);
    if (options.useLLM) args.push("--llm");
    if (options.plasmid) args.push("--plasmid", options.plasmid);
    if (options.model) args.push("--model", options.model);

    args.push("--db-path", getDbPath());

    const env = getAnalysisEnv();
    const child = spawn(cmd, args, { cwd, env, windowsHide: true });
    let stdout = "";
    let stderr = "";
    let stderrBuffer = "";

    sendProgress({
      stage: "preparing",
      percent: 2,
      processedSamples: 0,
      totalSamples: 0,
      message: "Preparing analysis runtime.",
    });

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on("data", (chunk) => {
      const text = chunk.toString();
      stderr += text;
      stderrBuffer += text;

      const lines = stderrBuffer.split(/\r?\n/);
      stderrBuffer = lines.pop() || "";
      for (const line of lines) {
        if (line.startsWith("__BIOAGENT_PROGRESS__")) {
          const rawPayload = line.slice("__BIOAGENT_PROGRESS__".length);
          try {
            sendProgress(JSON.parse(rawPayload));
          } catch (error) {
            console.warn("Failed to parse progress payload:", rawPayload, error);
          }
          continue;
        }

        if (line.trim()) {
          console.log("Python Warning/Info:", line);
        }
      }
    });

    child.on("error", (error) => {
      sendProgress({
        stage: "failed",
        percent: 100,
        processedSamples: 0,
        totalSamples: 0,
        message: error.message,
      });
      reject(`Analysis failed: ${error.message}`);
    });

    child.on("close", (code) => {
      if (stderrBuffer.trim()) {
        if (stderrBuffer.startsWith("__BIOAGENT_PROGRESS__")) {
          const rawPayload = stderrBuffer.slice("__BIOAGENT_PROGRESS__".length);
          try {
            sendProgress(JSON.parse(rawPayload));
          } catch (error) {
            console.warn("Failed to parse trailing progress payload:", rawPayload, error);
          }
        } else {
          console.log("Python Warning/Info:", stderrBuffer);
        }
      }

      if (code !== 0) {
        const errorMessage = stderr || `Process exited with code ${code}`;
        sendProgress({
          stage: "failed",
          percent: 100,
          processedSamples: 0,
          totalSamples: 0,
          message: errorMessage.trim(),
        });
        console.error("Python Error:", errorMessage);
        reject(`Analysis failed: ${errorMessage}`);
      } else {
        resolve(stdout);
      }
    });
  });
});

// History handler
ipcMain.handle("get-history", async () => {
  return new Promise((resolve, reject) => {
    const { cmd, baseArgs, cwd } = getPythonCommand();
    const args = [...baseArgs, "--history", "--db-path", getDbPath()];
    const env = getAnalysisEnv();

    execFile(cmd, args, { cwd, env }, (err, stdout, stderr) => {
      if (err) {
        console.error("History error:", stderr);
        reject(stderr || err.message);
      } else {
        if (stderr) console.log("History info:", stderr);
        resolve(stdout);
      }
    });
  });
});

ipcMain.handle("agent-chat", async (_event, payload) => {
  return new Promise((resolve, reject) => {
    const { cmd, baseArgs, cwd } = getPythonCommand();
    const args = [...baseArgs, "--agent-chat", JSON.stringify(payload)];
    const env = getAnalysisEnv();

    execFile(cmd, args, { cwd, env, maxBuffer: 20 * 1024 * 1024 }, (err, stdout, stderr) => {
      if (err) {
        console.error("Agent chat error:", stderr);
        reject(stderr || err.message);
      } else {
        if (stderr) console.log("Agent chat info:", stderr);
        resolve(stdout);
      }
    });
  });
});

ipcMain.handle("interpret-command", async (_event, text) => {
  return new Promise((resolve, reject) => {
    const { cmd, baseArgs, cwd } = getPythonCommand();
    const args = [...baseArgs, "--interpret-command", String(text)];
    const env = getAnalysisEnv();

    execFile(cmd, args, { cwd, env }, (err, stdout, stderr) => {
      if (err) {
        console.error("Interpret command error:", stderr);
        reject(stderr || err.message);
      } else {
        if (stderr) console.log("Interpret command info:", stderr);
        try {
          resolve(JSON.parse(stdout));
        } catch (parseError) {
          reject(`Interpret command returned invalid JSON: ${parseError.message}`);
        }
      }
    });
  });
});

// Settings handlers
function getSettingsPath() {
  return path.join(app.getPath("userData"), "settings.json");
}

ipcMain.handle("load-settings", async () => {
  try {
    return fs.readFileSync(getSettingsPath(), "utf-8");
  } catch {
    return null;
  }
});

ipcMain.handle("save-settings", async (_event, settingsJson) => {
  fs.writeFileSync(getSettingsPath(), settingsJson, "utf-8");
  return true;
});

// Export Excel handler
ipcMain.handle("export-excel", async (_event, samples, sourcePath) => {
  const { filePath } = await dialog.showSaveDialog({
    defaultPath: "BioAgent-Report.xlsx",
    filters: [{ name: "Excel", extensions: ["xlsx"] }],
  });
  if (!filePath) return null;

  const tmpFile = path.join(app.getPath("temp"), "bioagent-export.json");
  fs.writeFileSync(tmpFile, JSON.stringify({ samples, source_path: sourcePath || "" }));

  const { cmd, baseArgs, cwd } = getPythonCommand();
  const args = [...baseArgs, "--export-excel", filePath, "--export-data", tmpFile];
  const env = getAnalysisEnv();

  return new Promise((resolve, reject) => {
    execFile(cmd, args, { cwd, env }, (err, stdout, stderr) => {
      if (err) {
        console.error("Export error:", stderr);
        reject(stderr || err.message);
      } else {
        if (stderr) console.log("Export info:", stderr);
        resolve(JSON.parse(stdout));
      }
    });
  });
});

ipcMain.handle("open-export-folder", async (_event, exportedPath) => {
  if (typeof exportedPath !== "string") return false;
  const normalizedPath = exportedPath.trim();
  if (!normalizedPath) return false;

  try {
    return Boolean(shell.showItemInFolder(normalizedPath));
  } catch {
    return false;
  }
});

// --------------- App lifecycle ---------------

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
