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

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
