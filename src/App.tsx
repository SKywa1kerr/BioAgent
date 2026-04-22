import { useState, useCallback, useMemo } from "react";
import type { Sample, ChromatogramData } from "./shared/types";
import { SequenceViewer } from "./features/analysis";
import { AgentPanel } from "./features/agent";
import { HistoryPanel } from "./features/history";
import { PrimerPanel } from "./features/primer";
import { SettingsPanel } from "./features/settings";
import { ReportPanel } from "./features/report";
import { keysToCamelCase } from "./shared/utils/caseConverter";
import "./App.css";

const { invoke } = window.electronAPI;

type NavFeature = "analysis" | "agent" | "history" | "settings" | "primer" | "report";

/**
 * BioAgent Desktop App
 *
 * ARCHITECTURE:
 * - Zustand store (sequencingStore) is available for child components
 * - App.tsx uses local state to avoid subscription complexity
 * - Components receive props (container/presenter pattern)
 */

function App() {
  // Navigation state
  const [activeFeature, setActiveFeature] = useState<NavFeature>("analysis");

  // Core analysis state
  const [samples, setSamples] = useState<Sample[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showChromatogram, setShowChromatogram] = useState(true);

  // Directory selection
  const [ab1Dir, setAb1Dir] = useState<string | null>(null);
  const [genesDir, setGenesDir] = useState<string | null>(null);

  const selectedSample = samples.find((s) => s.id === selectedId);

  // Build chromatogram data from sample
  const chromatogramData = useMemo<ChromatogramData | null>(() => {
    if (!selectedSample?.tracesA) return null;
    return {
      traces: {
        A: selectedSample.tracesA,
        T: selectedSample.tracesT || [],
        G: selectedSample.tracesG || [],
        C: selectedSample.tracesC || [],
      },
      quality: selectedSample.quality || [],
      baseCalls: selectedSample.querySequence,
      baseLocations: selectedSample.baseLocations || [],
      mixedPeaks: selectedSample.mixedPeaks || [],
    };
  }, [selectedSample]);

  const runAnalysis = useCallback(async () => {
    if (!ab1Dir) {
      alert("Please select an AB1 folder first");
      return;
    }

    setIsAnalyzing(true);
    try {
      const result = (await invoke("run-analysis", ab1Dir, genesDir, {
        useLLM: false,
      })) as string;

      const data = JSON.parse(result);
      // Convert snake_case backend response to camelCase
      const camelCaseData = keysToCamelCase<{ samples: Sample[] }>(data);
      setSamples(camelCaseData.samples);

      if (camelCaseData.samples.length > 0) {
        setSelectedId(camelCaseData.samples[0].id);
      }
    } catch (error) {
      console.error("Analysis failed:", error);
      alert(`Analysis failed: ${error}`);
    } finally {
      setIsAnalyzing(false);
    }
  }, [ab1Dir, genesDir]);

  const handleSelectAb1Dir = async () => {
    const folder = (await invoke("open-folder-dialog")) as string | null;
    if (folder) setAb1Dir(folder);
  };

  const handleSelectGenesDir = async () => {
    const folder = (await invoke("open-folder-dialog")) as string | null;
    if (folder) setGenesDir(folder);
  };

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
                    onClick={() => setSelectedId(sample.id)}
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
                      chromatogramData={chromatogramData}
                      showChromatogram={showChromatogram}
                      cdsStart={selectedSample.cdsStart}
                      cdsEnd={selectedSample.cdsEnd}
                    />
                  </div>

                  {/* Quick Stats - Link to full Report */}
                  <div className="details-section">
                    <h4>Quick Stats</h4>
                    <div className="metrics">
                      <span>Identity: {((selectedSample.identity || 0) * 100).toFixed(1)}%</span>
                      <span>Coverage: {((selectedSample.coverage || 0) * 100).toFixed(1)}%</span>
                      <span>Status: {selectedSample.status}</span>
                    </div>
                    <button
                      className="btn-link"
                      onClick={() => setActiveFeature("report")}
                    >
                      View Full Report →
                    </button>
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

      case "report":
        return (
          <div className="app-body feature-body">
            <ReportPanel samples={samples} selectedSampleId={selectedId} />
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
            className={`nav-btn ${activeFeature === "report" ? "active" : ""}`}
            onClick={() => setActiveFeature("report")}
          >
            Report
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
