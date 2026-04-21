import { useState, useEffect } from "react";
import type { AppSettings, AppLanguage, AppTheme } from "../../../shared/types";
import "./SettingsPanel.css";

const { invoke } = window.electronAPI;

export function SettingsPanel() {
  const [settings, setSettings] = useState<AppSettings>({
    language: "en",
    theme: "system",
    aiReviewEnabled: true,
    analysisDecisionMode: "manual",
  });
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState("");

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      const raw = (await invoke("get-settings")) as string;
      const data = JSON.parse(raw) as AppSettings;
      setSettings(data);
    } catch (error) {
      console.error("Failed to load settings:", error);
    }
  };

  const saveSettings = async () => {
    setIsLoading(true);
    try {
      await invoke("save-settings", JSON.stringify(settings));
      setMessage("Settings saved successfully");
      setTimeout(() => setMessage(""), 3000);
    } catch (error) {
      console.error("Failed to save settings:", error);
      setMessage("Failed to save settings");
    } finally {
      setIsLoading(false);
    }
  };

  const handleLanguageChange = (language: AppLanguage) => {
    setSettings((s) => ({ ...s, language }));
  };

  const handleThemeChange = (theme: AppTheme) => {
    setSettings((s) => ({ ...s, theme }));
    // Apply theme immediately
    document.documentElement.setAttribute("data-theme", theme);
  };

  return (
    <div className="settings-panel">
      <header className="settings-panel-header">
        <h3>Settings</h3>
      </header>

      <div className="settings-content">
        <section className="settings-section">
          <h4>Appearance</h4>
          <div className="setting-item">
            <label>Language</label>
            <div className="setting-options">
              <button
                className={settings.language === "en" ? "active" : ""}
                onClick={() => handleLanguageChange("en")}
              >
                English
              </button>
              <button
                className={settings.language === "zh" ? "active" : ""}
                onClick={() => handleLanguageChange("zh")}
              >
                中文
              </button>
            </div>
          </div>

          <div className="setting-item">
            <label>Theme</label>
            <div className="setting-options">
              <button
                className={settings.theme === "light" ? "active" : ""}
                onClick={() => handleThemeChange("light")}
              >
                Light
              </button>
              <button
                className={settings.theme === "dark" ? "active" : ""}
                onClick={() => handleThemeChange("dark")}
              >
                Dark
              </button>
              <button
                className={settings.theme === "system" ? "active" : ""}
                onClick={() => handleThemeChange("system")}
              >
                System
              </button>
            </div>
          </div>
        </section>

        <section className="settings-section">
          <h4>Analysis</h4>
          <div className="setting-item">
            <label>AI Review</label>
            <div className="setting-toggle">
              <input
                type="checkbox"
                checked={settings.aiReviewEnabled}
                onChange={(e) =>
                  setSettings((s) => ({ ...s, aiReviewEnabled: e.target.checked }))
                }
              />
              <span>Enable AI-powered result review</span>
            </div>
          </div>

          <div className="setting-item">
            <label>Decision Mode</label>
            <div className="setting-options">
              <button
                className={settings.analysisDecisionMode === "manual" ? "active" : ""}
                onClick={() =>
                  setSettings((s) => ({ ...s, analysisDecisionMode: "manual" }))
                }
              >
                Manual
              </button>
              <button
                className={settings.analysisDecisionMode === "auto" ? "active" : ""}
                onClick={() =>
                  setSettings((s) => ({ ...s, analysisDecisionMode: "auto" }))
                }
              >
                Auto
              </button>
            </div>
          </div>
        </section>

        <section className="settings-section">
          <h4>AI Configuration</h4>
          <div className="setting-item">
            <label>API Key</label>
            <input
              type="password"
              value={settings.llmApiKey || ""}
              onChange={(e) =>
                setSettings((s) => ({ ...s, llmApiKey: e.target.value }))
              }
              placeholder="Enter your API key"
            />
          </div>

          <div className="setting-item">
            <label>Model</label>
            <input
              type="text"
              value={settings.llmModel || ""}
              onChange={(e) =>
                setSettings((s) => ({ ...s, llmModel: e.target.value }))
              }
              placeholder="e.g., gpt-4"
            />
          </div>
        </section>

        {message && <div className="settings-message">{message}</div>}

        <div className="settings-actions">
          <button
            onClick={saveSettings}
            disabled={isLoading}
            className="btn-primary"
          >
            {isLoading ? "Saving..." : "Save Settings"}
          </button>
        </div>
      </div>
    </div>
  );
}
