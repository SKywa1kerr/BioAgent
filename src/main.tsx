import React from "react";
import ReactDOM from "react-dom/client";
import "@fontsource/inter/400.css";
import "@fontsource/inter/500.css";
import "@fontsource/inter/600.css";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/500.css";
import "@fontsource/source-serif-pro/400.css";
import "@fontsource/source-serif-pro/600.css";
import "./styles/tokens.css";
import { App } from "./App";
import { ToastProvider } from "./components/ui/ToastProvider";
import "./styles.css";

// Browser-only stub so the renderer mounts in pure-Vite dev/e2e (no Electron
// preload). The real preload bridge replaces this when running inside Electron.
// Keeping the surface as no-ops means UI flows that don't hit the agent still
// work — and any IPC call resolves to { ok: false } rather than crashing.
if (typeof window !== "undefined" && !(window as { electronAPI?: unknown }).electronAPI) {
  (window as { electronAPI: unknown }).electronAPI = {
    invoke: async () => ({ ok: false, error: "electronAPI unavailable (browser mode)" }),
    onAgentEvent: () => () => {},
    onUpdaterEvent: () => () => {},
    send: () => {},
    on: () => () => {},
  };
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ToastProvider>
      <App />
    </ToastProvider>
  </React.StrictMode>
);

