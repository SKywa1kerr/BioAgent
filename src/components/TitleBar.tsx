import { useEffect, useState } from "react";
import { Icon } from "./ui/Icon";
import "./TitleBar.css";

declare global {
  interface Window {
    electronWindow?: {
      minimize: () => Promise<void>;
      toggleMaximize: () => Promise<{ isMaximized: boolean }>;
      close: () => Promise<void>;
      onStateChange: (cb: (state: { isMaximized: boolean }) => void) => () => void;
      platform: string;
    };
  }
}

export interface TitleBarProps {
  title: string;
  /** Optional aria-labels for window control buttons. */
  labels?: {
    minimize?: string;
    maximize?: string;
    restore?: string;
    close?: string;
  };
}

export function TitleBar({ title, labels }: TitleBarProps): JSX.Element {
  const electronWindow = typeof window !== "undefined" ? window.electronWindow : undefined;
  const platform = electronWindow?.platform ?? "browser";
  const isDarwin = platform === "darwin";

  const [isMaximized, setIsMaximized] = useState(false);

  useEffect(() => {
    if (!electronWindow) return;
    const off = electronWindow.onStateChange((state) => {
      setIsMaximized(Boolean(state?.isMaximized));
    });
    return () => {
      try { off(); } catch { /* ignore */ }
    };
  }, [electronWindow]);

  const handleMinimize = () => {
    void electronWindow?.minimize();
  };
  const handleToggleMaximize = async () => {
    if (!electronWindow) return;
    try {
      const result = await electronWindow.toggleMaximize();
      setIsMaximized(Boolean(result?.isMaximized));
    } catch {
      /* ignore */
    }
  };
  const handleClose = () => {
    void electronWindow?.close();
  };

  const minimizeLabel = labels?.minimize ?? "Minimize";
  const maximizeLabel = labels?.maximize ?? "Maximize";
  const restoreLabel = labels?.restore ?? "Restore";
  const closeLabel = labels?.close ?? "Close";

  return (
    <div className={`titlebar${isDarwin ? " is-darwin" : " is-other"}`} role="presentation">
      {!isDarwin ? (
        <div className="titlebar-brand" aria-hidden="true">
          <Icon name="lab" size={14} />
        </div>
      ) : null}
      <div className="titlebar-title" aria-label={title}>{title}</div>
      {!isDarwin ? (
        <div className="titlebar-controls" role="group" aria-label="Window controls">
          <button
            type="button"
            className="titlebar-button titlebar-min"
            onClick={handleMinimize}
            aria-label={minimizeLabel}
            title={minimizeLabel}
          >
            <Icon name="window-minimize" size={14} aria-hidden="true" />
          </button>
          <button
            type="button"
            className="titlebar-button titlebar-max"
            onClick={() => void handleToggleMaximize()}
            aria-label={isMaximized ? restoreLabel : maximizeLabel}
            title={isMaximized ? restoreLabel : maximizeLabel}
          >
            <Icon
              name={isMaximized ? "window-restore" : "window-maximize"}
              size={12}
              aria-hidden="true"
            />
          </button>
          <button
            type="button"
            className="titlebar-button titlebar-close"
            onClick={handleClose}
            aria-label={closeLabel}
            title={closeLabel}
          >
            <Icon name="close" size={14} aria-hidden="true" />
          </button>
        </div>
      ) : null}
    </div>
  );
}
