import { useState, useCallback, useMemo } from "react";
import type { Sample, ChromatogramData } from "./shared/types";
import { SequenceViewer } from "./features/analysis";
import "./App.css";

// Handle both Electron and browser environments
const electronAPI = (window as unknown as { electronAPI?: { invoke: (cmd: string, ...args: unknown[]) => Promise<unknown> } }).electronAPI;
const isElectron = !!electronAPI;
const invoke = electronAPI?.invoke || (() => Promise.reject("Electron API not available"));

// Mock data for browser development
const mockSamples: Sample[] = [
  {
    id: "sample-1",
    name: "Test Sample 1",
    refSequence: "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
    querySequence: "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
    alignedRefG: "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
    alignedQueryG: "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
    alignedQuery: "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
    matches: Array(41).fill(true),
    cdsStart: 1,
    cdsEnd: 41,
    status: "ok",
    identity: 1.0,
    coverage: 1.0,
    mutations: [],
    tracesA: Array(500).fill(0).map(() => Math.random() * 100),
    tracesT: Array(500).fill(0).map(() => Math.random() * 100),
    tracesG: Array(500).fill(0).map(() => Math.random() * 100),
    tracesC: Array(500).fill(0).map(() => Math.random() * 100),
    quality: Array(41).fill(0).map(() => Math.floor(Math.random() * 40 + 10)),
    baseLocations: Array(41).fill(0).map((_, i) => i * 12),
    mixedPeaks: [],
  },
  {
    id: "sample-2",
    name: "Test Sample 2 (with mutation)",
    refSequence: "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
    querySequence: "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
    alignedRefG: "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
    alignedQueryG: "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
    alignedQuery: "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
    matches: Array(41).fill(true).map((_, i) => i !== 10),
    cdsStart: 1,
    cdsEnd: 41,
    status: "wrong",
    identity: 0.98,
    coverage: 1.0,
    mutations: [{ position: 10, type: "substitution", ref: "A", query: "G" }],
    tracesA: Array(500).fill(0).map(() => Math.random() * 100),
    tracesT: Array(500).fill(0).map(() => Math.random() * 100),
    tracesG: Array(500).fill(0).map(() => Math.random() * 100),
    tracesC: Array(500).fill(0).map(() => Math.random() * 100),
    quality: Array(41).fill(0).map(() => Math.floor(Math.random() * 40 + 10)),
    baseLocations: Array(41).fill(0).map((_, i) => i * 12),
    mixedPeaks: [],
  },
];

/**
 * Minimal App focused on Analysis feature
 *
 * EXTENSION POINTS:
 * - Agent panel: Add to right sidebar
 * - History: Add tab or drawer
 * - Settings: Add modal or tab
 * - Primer design: Add toolbar button
 */

function App() {
  // Core state
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
    if (!ab1Dir && isElectron) {
      alert("Please select an AB1 folder first");
      return;
    }

    setIsAnalyzing(true);
    try {
      if (!isElectron) {
        // Mock mode for browser development
        await new Promise(resolve => setTimeout(resolve, 500));
        setSamples(mockSamples);
        setSelectedId(mockSamples[0].id);
      } else {
        const result = (await invoke("run-analysis", ab1Dir, genesDir, {
          useLLM: false,
        })) as string;

        const data = JSON.parse(result);
        setSamples(data.samples);

        if (data.samples.length > 0) {
          setSelectedId(data.samples[0].id);
        }
      }
    } catch (error) {
      console.error("Analysis failed:", error);
      alert(`Analysis failed: ${error}`);
    } finally {
      setIsAnalyzing(false);
    }
  }, [ab1Dir, genesDir]);

  const handleSelectAb1Dir = async () => {
    if (!isElectron) {
      setAb1Dir("/mock/ab1/path");
      return;
    }
    const folder = (await invoke("open-folder-dialog")) as string | null;
    if (folder) setAb1Dir(folder);
  };

  const handleSelectGenesDir = async () => {
    if (!isElectron) {
      setGenesDir("/mock/genes/path");
      return;
    }
    const folder = (await invoke("open-folder-dialog")) as string | null;
    if (folder) setGenesDir(folder);
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="logo">BioAgent</div>
        <div className="toolbar">
          <div className="path-selectors">
            <button onClick={handleSelectAb1Dir} title={ab1Dir || "Select AB1 Folder"}>
              {ab1Dir ? `AB1: ...${ab1Dir.slice(-15)}` : "Select AB1"}
            </button>
            <button onClick={handleSelectGenesDir} title={genesDir || "Select Genes"}>
              {genesDir ? `Genes: ...${genesDir.slice(-15)}` : "Select Genes"}
            </button>
          </div>
          <div className="action-buttons">
            <button
              className="btn-primary"
              onClick={runAnalysis}
              disabled={isAnalyzing || (isElectron && !ab1Dir)}
            >
              {isAnalyzing ? "Analyzing..." : "Run Analysis"}
            </button>
            <button
              className="btn-secondary"
              onClick={() => setShowChromatogram(!showChromatogram)}
            >
              {showChromatogram ? "Hide Trace" : "Show Trace"}
            </button>
            {/* EXTENSION: Add Primer Design button here */}
            {/* EXTENSION: Add Agent toggle here */}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="app-body">
        {/* Sample List Sidebar */}
        <aside className="sidebar">
          <div className="sample-list">
            {samples.map((sample) => (
              <div
                key={sample.id}
                className={`sample-item ${selectedId === sample.id ? "selected" : ""} ${sample.status}`}
                onClick={() => setSelectedId(sample.id)}
              >
                <span className="sample-name">{sample.name}</span>
                <span className="sample-status">{sample.status}</span>
              </div>
            ))}
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
              <p>Select AB1 files and run analysis to begin</p>
            </div>
          )}
        </main>

        {/* EXTENSION: Agent Panel Sidebar */}
        {/* <aside className="agent-sidebar">...</aside> */}
      </div>

      {/* EXTENSION: Settings Modal */}
      {/* EXTENSION: History Drawer */}
    </div>
  );
}

export default App;
