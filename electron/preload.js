const { contextBridge, ipcRenderer, webUtils } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
  invoke: (channel, ...args) => ipcRenderer.invoke(channel, ...args),
  onAgentEvent: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on("agent-event", listener);
    return () => ipcRenderer.removeListener("agent-event", listener);
  },
  // Electron 32+ removed File.path for security. The renderer must call
  // webUtils.getPathForFile(file) instead to recover the absolute path of
  // a dropped/selected file. Expose it through the bridge because webUtils
  // isn't available in the renderer's global scope under contextIsolation.
  getDroppedFilePath: (file) => {
    try {
      return webUtils.getPathForFile(file) || "";
    } catch {
      return "";
    }
  },
});

contextBridge.exposeInMainWorld("electronWindow", {
  minimize: () => ipcRenderer.invoke("window-minimize"),
  toggleMaximize: () => ipcRenderer.invoke("window-maximize-toggle"),
  close: () => ipcRenderer.invoke("window-close"),
  onStateChange: (callback) => {
    const listener = (_event, state) => callback(state);
    ipcRenderer.on("window-state", listener);
    return () => ipcRenderer.removeListener("window-state", listener);
  },
  platform: process.platform,
});

contextBridge.exposeInMainWorld("electronUpdater", {
  quitAndInstall: () => ipcRenderer.invoke("updater-quit-and-install"),
  onState: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on("updater-state", listener);
    return () => ipcRenderer.removeListener("updater-state", listener);
  },
});
