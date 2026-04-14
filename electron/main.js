const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");
const { pathToFileURL } = require("url");

let mainWindow = null;
let agentHarness = null;

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

  mainWindow.loadURL("http://127.0.0.1:1420");
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

app.whenReady().then(async () => {
  createWindow();

  const mod = await import(pathToFileURL(path.join(__dirname, "agent_harness.mjs")).href);
  const AgentHarness = mod.AgentHarness;

  ipcMain.handle("agent-harness-init", async (_event, settings) => {
    if (!agentHarness) {
      agentHarness = new AgentHarness({
        ...settings,
        pythonConfig: getPythonConfig(),
      });
      const tools = await agentHarness.initMcpServer();
      return { ok: true, tools };
    }
    return { ok: true, tools: agentHarness.mcpTools || [] };
  });

  ipcMain.handle("agent-harness-run", async (event, userMessage, settings) => {
    if (!agentHarness) {
      const tools = [];
      return { ok: false, error: "Agent harness not initialized", tools };
    }

    const events = [];
    await agentHarness.runTurn(userMessage, (payload) => {
      events.push(payload);
      if (!event.sender.isDestroyed()) {
        event.sender.send("agent-event", payload);
      }
    });
    return { ok: true, events };
  });

  ipcMain.handle("agent-harness-shutdown", async () => {
    if (agentHarness) {
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
