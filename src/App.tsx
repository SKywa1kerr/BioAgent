import { useState, useCallback } from "react";
import type { Sample } from "./shared/types";
import { SequenceViewer } from "./features/analysis";
import { AgentPanel } from "./features/agent";
import { HistoryPanel } from "./features/history";
import { PrimerPanel } from "./features/primer";
import { SettingsPanel } from "./features/settings";
import { useSequencingStore, useSelectedRun, useAllRuns } from "./features/analysis/stores/sequencingStore";
import { keysToCamelCase } from "./shared/utils/caseConverter";
import "./App.css";

const { invoke } = window.electronAPI;

type NavFeature = "analysis" | "agent" | "history" | "settings" | "primer";

/**
 * BioAgent Desktop App
 *
 * ARCHITECTURE:
 * - Zustand store (sequencingStore) is the single source of truth
 * - importFromLegacy() normalizes API data into entities (runs/references/analyses)
 * - Components receive props, hooks derive data from store
 * - App.tsx is the container that coordinates data flow
 */

function App() {
  // Navigation state (UI only, not domain data)
  const [activeFeature, setActiveFeature] = useState<NavFeature>("analysis");

  // Directory selection
  const [ab1Dir, setAb1Dir] = useState<string | null>(null);
  const [genesDir, setGenesDir] = useState<string | null>(null);

  // Zustand store actions
  const importFromLegacy = useSequencingStore((s) => s.importFromLegacy);
  const selectRun = useSequencingStore((s) => s.selectRun);
  const setShowChromatogram = useSequencingStore((s) => s.setShowChromatogram);
  const isAnalyzing = useSequencingStore((s) => s.isAnalyzing);
  const showChromatogram = useSequencingStore((s) => s.showChromatogram);

  // Derived state from store
  const runs = useAllRuns();
  const selectedRun = useSelectedRun();
  const selectedId = selectedRun?.id || null;

  const runAnalysis = useCallback(async () => {
    if (!ab1Dir) {
      alert("Please select an AB1 folder first");
      return;
    }

    useSequencingStore.setState({ isAnalyzing: true });
    try {
      const result = (await invoke("run-analysis", ab1Dir, genesDir, {
        useLLM: false,
      })) as string;

      const data = JSON.parse(result);
      // Convert snake_case backend response to camelCase
      const camelCaseData = keysToCamelCase<{ samples: Sample[] }>(data);
      // Normalize into store entities
      importFromLegacy(camelCaseData.samples);
    } catch (error) {
      console.error("Analysis failed:", error);
      alert(`Analysis failed: ${error}`);
    } finally {
      useSequencingStore.setState({ isAnalyzing: false });
    }
  }, [ab1Dir, genesDir, importFromLegacy]);

  const handleSelectAb1Dir = async () => {
    const folder = (await invoke("open-folder-dialog")) as string | null;
    if (folder) setAb1Dir(folder);
  };

  const handleSelectGenesDir = async () => {
    const folder = (await invoke("open-folder-dialog")) as string | null;
    if (folder) setGenesDir(folder);
  };

  const handleSelectSample = (id: string) => {
    selectRun(id);
  };

  // Convert runs to Sample-like objects for child components
  const samples: Sample[] = runs.map((run) => ({
    id: run.id,
    name: run.name,
    clone: run.clone || "",
    status: run.analysis?.metrics?.identity === 1 ? "ok" : run.uiState.error ? "error" : "wrong",
    identity: run.analysis?.metrics?.identity || 0,
    coverage: run.analysis?.metrics?.coverage || 0,
    mutations: [],
    refSequence: run.raw?.baseCalls || "",
    querySequence: run.raw?.baseCalls || "",
    alignedRefG: run.analysis?.alignment?.refGapped,
    alignedQueryG: run.analysis?.alignment?.queryGapped,
    alignedQuery: run.analysis?.alignment?.queryGapped || "",
    matches: run.analysis?.alignment?.matches || [],
    cdsStart: 0,
    cdsEnd: 0,
    frameshift: run.analysis?.metrics?.frameshift || false,
    tracesA: run.raw?.traces?.A,
    tracesT: run.raw?.traces?.T,
    tracesG: run.raw?.traces?.G,
    tracesC: run.raw?.traces?.C,
    quality: run.raw?.quality,
    baseLocations: run.raw?.baseLocations,
    mixedPeaks: run.raw?.mixedPeaks,
  }));

  const selectedSample = samples.find((s) => s.id === selectedId);

  // Render feature content
  const renderFeatureContent = () => {
    switch (activeFeature) {
      case "analysis":
        return (
          <div className="app-body">
            {/* Sample List Sidebar */}
            <aside className="sidebar">
              <div className="sidebar-header">
                <h3>Samples ({samples.length})</h3>
              </div>
              <div className="sample-list">
                {samples.map((sample) => (
                  <div
                    key={sample.id}
                    className={`sample-item ${selectedId === sample.id ? "selected" : ""} ${sample.status}`}
                    onClick={() => handleSelectSample(sample.id)}
                    title={sample.name || sample.clone || sample.id || "Unnamed"}
                  >
                    <span className="sample-name">{sample.name || sample.clone || sample.id || "Unnamed"}</span>
                    <span className={`sample-status status-${sample.status || "unknown"}`}>
                      {sample.status || "unknown"}
                    </span>
                  </div>
                ))}
                {samples.length === 0 && (
                  <div className="sample-list-empty">No samples loaded</div>
                )}
              </div>
            </aside>

            {/* Analysis View */}
            <main className="main-content">
              {selectedSample ? (
                <div className="viewer">
                  <div className="sequence-section">
                    <SequenceViewer
                      sampleId={selectedSample.id}
                      refSequence={selectedSample.refSequence}
                      alignedRefG={selectedSample.alignedRefG}
                      alignedQueryG={selectedSample.alignedQueryG}
                      alignedQuery={selectedSample.alignedQuery}
                      matches={selectedSample.matches}
                      showChromatogram={showChromatogram}
                      cdsStart={selectedSample.cdsStart}
                      cdsEnd={selectedSample.cdsEnd}
                    />
                  </div>

                  {/* Mutation Summary */}
                  <div className="details-section">
                    <h4>Analysis Results</h4>
                    <div className="metrics">
                      <span>Identity: {((selectedSample.identity || 0) * 100).toFixed(1)}%</span>
                      <span>Coverage: {((selectedSample.coverage || 0) * 100).toFixed(1)}%</span>
                      <span>Mutations: {selectedSample.mutations?.length || 0}</span>
                    </div>
                    {selectedSample.frameshift && (
                      <div className="alert error">Frameshift detected!</div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="empty-state">
                  <div className="empty-state-content">
                    <p>Select AB1 files and run analysis to begin</p>
                  </div>
                </div>
              )}
            </main>
          </div>
        );

      case "agent":
        return (
          <div className="app-body feature-body">
            <AgentPanel samples={samples} selectedSampleId={selectedId} />
          </div>
        );

      case "history":
        return (
          <div className="app-body feature-body">
            <HistoryPanel />
          </div>
        );

      case "settings":
        return (
          <div className="app-body feature-body">
            <SettingsPanel />
          </div>
        );

      case "primer":
        return (
          <div className="app-body feature-body">
            <PrimerPanel />
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="logo">BioAgent</div>

        {/* Feature Navigation */}
        <nav className="feature-nav">
          <button
            className={`nav-btn ${activeFeature === "analysis" ? "active" : ""}`}
            onClick={() => setActiveFeature("analysis")}
          >
            Analysis
          </button>
          <button
            className={`nav-btn ${activeFeature === "agent" ? "active" : ""}`}
            onClick={() => setActiveFeature("agent")}
          >
            Agent
          </button>
          <button
            className={`nav-btn ${activeFeature === "primer" ? "active" : ""}`}
            onClick={() => setActiveFeature("primer")}
          >
            Primer
          </button>
          <button
            className={`nav-btn ${activeFeature === "history" ? "active" : ""}`}
            onClick={() => setActiveFeature("history")}
          >
            History
          </button>
          <button
            className={`nav-btn ${activeFeature === "settings" ? "active" : ""}`}
            onClick={() => setActiveFeature("settings")}
          >
            Settings
          </button>
        </nav>

        {/* Analysis Toolbar (only show in analysis mode) */}
        {activeFeature === "analysis" && (
          <div className="toolbar">
            <div className="path-selectors">
              <button onClick={handleSelectAb1Dir} title={ab1Dir || "Select AB1 Folder"}>
                {ab1Dir ? `AB1: ...${ab1Dir.slice(-20)}` : "Select AB1"}
              </button>
              <button onClick={handleSelectGenesDir} title={genesDir || "Select Genes"}>
                {genesDir ? `Genes: ...${genesDir.slice(-20)}` : "Select Genes"}
              </button>
            </div>
            <div className="action-buttons">
              <button
                className="btn-primary"
                onClick={runAnalysis}
                disabled={isAnalyzing || !ab1Dir}
              >
                {isAnalyzing ? "Analyzing..." : "Run Analysis"}
              </button>
              <button
                className="btn-secondary"
                onClick={() => setShowChromatogram(!showChromatogram)}
                disabled={!selectedSample}
              >
                {showChromatogram ? "Hide Trace" : "Show Trace"}
              </button>
            </div>
          </div>
        )}
      </header>

      {/* Feature Content */}
      {renderFeatureContent()}
    </div>
  );
}

export default App;
