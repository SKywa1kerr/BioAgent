const { app, BrowserWindow, dialog, ipcMain } = require("electron");
const path = require("path");
const fs = require("fs");
const { execFile } = require("child_process");

const isDev = !app.isPackaged;

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

// IPC Handlers

ipcMain.handle("open-folder-dialog", async () => {
  const result = await dialog.showOpenDialog({
    properties: ["openDirectory"],
  });
  if (result.canceled || result.filePaths.length === 0) return null;
  return result.filePaths[0];
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
    const pythonPath = process.platform === "win32" ? "python" : "python3";
    const pythonDir = path.join(__dirname, "../src-python");

    const args = ["-m", "bioagent.main"];
    if (options.autoImport) {
      args.push("--auto-import");
    } else {
      args.push("--ab1-dir", ab1Dir);
    }

    if (genesDir) {
      args.push("--genes-dir", genesDir);
    }

    if (options.useLLM) {
      args.push("--llm");
    }

    if (options.plasmid) {
      args.push("--plasmid", options.plasmid);
    }

    const env = { ...process.env, PYTHONPATH: pythonDir };

    // Increase maxBuffer to 100MB to handle large chromatogram data
    execFile(pythonPath, args, { cwd: pythonDir, env, maxBuffer: 100 * 1024 * 1024 }, (error, stdout, stderr) => {
      if (error) {
        // Only reject if there's an actual error code, not just stderr output
        console.error("Python Error:", stderr);
        reject(`Analysis failed: ${stderr || error.message}`);
      } else {
        // Some Python libraries might print to stderr even on success
        if (stderr) {
          console.log("Python Warning/Info:", stderr);
        }
        resolve(stdout);
      }
    });
  });
});

// History handler
ipcMain.handle("get-history", async () => {
  return new Promise((resolve, reject) => {
    const pythonPath = process.platform === "win32" ? "python" : "python3";
    const pythonDir = path.join(__dirname, "../src-python");
    const env = { ...process.env, PYTHONPATH: pythonDir };
    execFile(pythonPath, ["-m", "bioagent.main", "--history"], { cwd: pythonDir, env }, (err, stdout, stderr) => {
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

  const pythonPath = process.platform === "win32" ? "python" : "python3";
  const pythonDir = path.join(__dirname, "../src-python");
  const env = { ...process.env, PYTHONPATH: pythonDir };
  const args = ["-m", "bioagent.main", "--export-excel", filePath, "--export-data", tmpFile];

  return new Promise((resolve, reject) => {
    execFile(pythonPath, args, { cwd: pythonDir, env }, (err, stdout, stderr) => {
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

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
