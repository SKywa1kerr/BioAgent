import { useState, useCallback } from "react";
import { Sample, ChromatogramData } from "./types";
import { SampleList } from "./components/SampleList";
import { SequenceViewer } from "./components/SequenceViewer";
import { ChromatogramCanvas } from "./components/ChromatogramCanvas";
import "./App.css";

const { invoke } = window.electronAPI;

function App() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const [ab1Dir, setAb1Dir] = useState<string | null>(null);
  const [genesDir, setGenesDir] = useState<string | null>(null);
  const [plasmid, setPlasmid] = useState("pet22b");

  const runAnalysis = useCallback(
    async (options: { autoImport?: boolean } = {}) => {
      if (!options.autoImport && !ab1Dir) {
        alert("Please select an AB1 folder first");
        return;
      }

      setIsAnalyzing(true);
      try {
        const result = (await invoke(
          "run-analysis",
          ab1Dir,
          genesDir,
          {
            ...options,
            useLLM: false,
            plasmid,
          }
        )) as string;

        const data = JSON.parse(result);
        setSamples(data.samples);

        if (data.samples.length > 0) {
          setSelectedId(data.samples[0].id);
        }
      } catch (error) {
        console.error("Analysis failed:", error);
        alert(`Analysis failed: ${error}`);
      } finally {
        setIsAnalyzing(false);
      }
    },
    [ab1Dir, genesDir, plasmid]
  );

  const handleSelectAb1Dir = async () => {
    const folder = (await invoke("open-folder-dialog")) as string | null;
    if (folder) setAb1Dir(folder);
  };

  const handleSelectGenesDir = async () => {
    const folder = (await invoke("open-folder-dialog")) as string | null;
    if (folder) setGenesDir(folder);
  };

  const selectedSample = samples.find((s) => s.id === selectedId);

  const chromatogramData: ChromatogramData | null =
    selectedSample &&
    selectedSample.traces_a &&
    selectedSample.traces_t &&
    selectedSample.traces_g &&
    selectedSample.traces_c &&
    selectedSample.quality &&
    selectedSample.query_sequence
      ? {
          traces: {
            A: selectedSample.traces_a,
            T: selectedSample.traces_t,
            G: selectedSample.traces_g,
            C: selectedSample.traces_c,
          },
          quality: selectedSample.quality,
          baseCalls: selectedSample.query_sequence,
          base_locations: selectedSample.base_locations || [],
          mixed_peaks: selectedSample.mixed_peaks || [],
        }
      : null;

  return (
    <div className="app">
      <header className="app-header">
        <div className="logo">BioAgent</div>
        <div className="toolbar">
          <div className="path-selectors">
            <button onClick={handleSelectAb1Dir} title={ab1Dir || "Select AB1 Folder"}>
              {ab1Dir ? `AB1: ...${ab1Dir.slice(-15)}` : "Import AB1 Files"}
            </button>
            <button onClick={handleSelectGenesDir} title={genesDir || "Select Genes Directory"}>
              {genesDir ? `Genes: ...${genesDir.slice(-15)}` : "Import Reference Files"}
            </button>
            <select value={plasmid} onChange={(e) => setPlasmid(e.target.value)}>
              <option value="pet22b">pET-22b</option>
              <option value="pet15b">pET-15b</option>
              <option value="none">None</option>
            </select>
          </div>
          <div className="action-buttons">
            <button
              className="btn-primary"
              onClick={() => runAnalysis()}
              disabled={isAnalyzing || !ab1Dir}
            >
              Run Analysis
            </button>
            <button
              className="btn-secondary"
              onClick={() => runAnalysis({ autoImport: true })}
              disabled={isAnalyzing}
              title="Auto-import from Downloads"
            >
              Auto-Import
            </button>
          </div>
        </div>
        <div className="sample-info">
          {selectedSample && (
            <>
              <span className={`status-badge ${selectedSample.status}`}>
                {selectedSample.status.toUpperCase()}
              </span>
              <span>
                Identity: {((selectedSample.identity || 0) * 100).toFixed(1)}%
              </span>
              <span>
                Coverage: {((selectedSample.coverage || 0) * 100).toFixed(1)}%
              </span>
              {selectedSample.frameshift && (
                <span className="status-badge wrong">FRAMESHIFT</span>
              )}
            </>
          )}
        </div>
      </header>

      <div className="app-body">
        <aside className="sidebar">
          <SampleList
            samples={samples}
            selectedId={selectedId}
            onSelect={setSelectedId}
          />
        </aside>

        <main className="main-content">
          {selectedSample ? (
            selectedSample.status === "error" ? (
              <div className="error-state">
                <h3>Analysis Error</h3>
                <p>{selectedSample.error}</p>
              </div>
            ) : (
              <div className="viewer">
                <div className="sequence-section">
                <SequenceViewer
                  refSequence={selectedSample.ref_sequence || ""}
                  querySequence={selectedSample.query_sequence || ""}
                  alignedRefG={selectedSample.aligned_ref_g || ""}
                  alignedQueryG={selectedSample.aligned_query_g || ""}
                  alignedQuery={selectedSample.aligned_query || ""}
                  matches={selectedSample.matches || []}
                  mutations={selectedSample.mutations || []}
                  chromatogramData={chromatogramData}
                  cdsStart={selectedSample.cds_start || 0}
                  cdsEnd={selectedSample.cds_end || 0}
                  featureName={selectedSample.clone || "CDS"}
                />
                </div>

                <div className="details-section">
                  {chromatogramData && (
                    <div className="full-chromatogram">
                      <h4>Chromatogram Spectrum</h4>
                      <ChromatogramCanvas
                        data={chromatogramData}
                        startPosition={1}
                        endPosition={chromatogramData.baseCalls.length}
                        onHover={() => {}}
                      />
                    </div>
                  )}
                  <h4>Mutations</h4>
                  {selectedSample.llm_verdict && (
                    <div className="llm-verdict">
                      <strong>LLM Verdict:</strong>
                      <p>{selectedSample.llm_verdict}</p>
                    </div>
                  )}
                  {selectedSample.mutations && selectedSample.mutations.length > 0 ? (
                    <table className="mutation-table">
                      <thead>
                        <tr>
                          <th>Pos</th>
                          <th>Ref</th>
                          <th>Query</th>
                          <th>Type</th>
                          <th>Effect</th>
                        </tr>
                      </thead>
                      <tbody>
                        {selectedSample.mutations.map((m, i) => (
                          <tr key={i}>
                            <td>{m.position}</td>
                            <td className="base-cell">{m.refBase}</td>
                            <td className="base-cell">{m.queryBase}</td>
                            <td>{m.type}</td>
                            <td>{m.effect || "-"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <p className="no-mutations">No mutations detected</p>
                  )}
                </div>
              </div>
            )
          ) : (
            <div className="empty-state">
              <p>Open a folder containing AB1 and GenBank files to begin</p>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
