const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
  invoke: (channel, ...args) => ipcRenderer.invoke(channel, ...args),
  onAnalysisProgress: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on("analysis-progress", listener);
    return () => ipcRenderer.removeListener("analysis-progress", listener);
  },
});
