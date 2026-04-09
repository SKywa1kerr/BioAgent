const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
  invoke: (channel, ...args) => ipcRenderer.invoke(channel, ...args),
  interpretCommand: (text) => ipcRenderer.invoke("interpret-command", text),
  openExportFolder: (exportedPath) => ipcRenderer.invoke("open-export-folder", exportedPath),
  onAnalysisProgress: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on("analysis-progress", listener);
    return () => ipcRenderer.removeListener("analysis-progress", listener);
  },
});
