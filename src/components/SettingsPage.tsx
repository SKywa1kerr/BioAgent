import React, { useState, useEffect } from "react";
import { AppSettings } from "../types";
import "./SettingsPage.css";

const { invoke } = window.electronAPI;

const DEFAULT_SETTINGS: AppSettings = {
  llmApiKey: "",
  llmBaseUrl: "https://api.chatanywhere.tech/v1",
  plasmid: "pet22b",
  qualityThreshold: 20,
};

export const SettingsPage: React.FC = () => {
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const result = await invoke("load-settings") as string;
        if (result) setSettings({ ...DEFAULT_SETTINGS, ...JSON.parse(result) });
      } catch (e) {
        console.error("Failed to load settings:", e);
      }
    })();
  }, []);

  const handleSave = async () => {
    try {
      await invoke("save-settings", JSON.stringify(settings));
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (e) {
      alert(`Save failed: ${e}`);
    }
  };

  return (
    <div className="settings-page">
      <h2>Settings</h2>
      <div className="settings-group">
        <h3>LLM API Configuration</h3>
        <label>
          API Key
          <input
            type="password"
            value={settings.llmApiKey}
            onChange={(e) => setSettings({ ...settings, llmApiKey: e.target.value })}
            placeholder="sk-..."
          />
        </label>
        <label>
          Base URL
          <input
            type="text"
            value={settings.llmBaseUrl}
            onChange={(e) => setSettings({ ...settings, llmBaseUrl: e.target.value })}
          />
        </label>
      </div>
      <div className="settings-group">
        <h3>Analysis Settings</h3>
        <label>
          Plasmid Template
          <select
            value={settings.plasmid}
            onChange={(e) => setSettings({ ...settings, plasmid: e.target.value })}
          >
            <option value="pet22b">pET-22b</option>
            <option value="pet15b">pET-15b</option>
            <option value="none">None</option>
          </select>
        </label>
        <label>
          Quality Threshold (Phred)
          <input
            type="number"
            value={settings.qualityThreshold}
            onChange={(e) => setSettings({ ...settings, qualityThreshold: parseInt(e.target.value) || 20 })}
            min={0}
            max={60}
          />
        </label>
      </div>
      <button className="btn-primary" onClick={handleSave}>
        {saved ? "Saved!" : "Save Settings"}
      </button>
    </div>
  );
};
