import { useState } from "react";
import "./PrimerPanel.css";

interface PrimerDesign {
  id: string;
  name: string;
  sequence: string;
  tm: number;
  gc: number;
  length: number;
}

export function PrimerPanel() {
  const [targetSequence, setTargetSequence] = useState("");
  const [primers, setPrimers] = useState<PrimerDesign[]>([]);
  const [isDesigning, setIsDesigning] = useState(false);

  const handleDesign = async () => {
    if (!targetSequence.trim()) return;

    setIsDesigning(true);
    try {
      // Placeholder for primer design logic
      // This would call the backend primer design algorithm
      await new Promise((resolve) => setTimeout(resolve, 1000));

      // Mock results for now
      const mockPrimers: PrimerDesign[] = [
        {
          id: "1",
          name: "Forward Primer",
          sequence: targetSequence.slice(0, 20),
          tm: 58.5,
          gc: 0.55,
          length: 20,
        },
        {
          id: "2",
          name: "Reverse Primer",
          sequence: targetSequence.slice(-20),
          tm: 60.2,
          gc: 0.5,
          length: 20,
        },
      ];
      setPrimers(mockPrimers);
    } finally {
      setIsDesigning(false);
    }
  };

  return (
    <div className="primer-panel">
      <header className="primer-panel-header">
        <h3>Primer Design</h3>
      </header>

      <div className="primer-content">
        <div className="primer-input-section">
          <label htmlFor="target-sequence">Target Sequence</label>
          <textarea
            id="target-sequence"
            value={targetSequence}
            onChange={(e) => setTargetSequence(e.target.value)}
            placeholder="Enter DNA sequence to design primers for..."
            rows={6}
          />
          <button
            onClick={handleDesign}
            disabled={isDesigning || !targetSequence.trim()}
            className="btn-primary"
          >
            {isDesigning ? "Designing..." : "Design Primers"}
          </button>
        </div>

        {primers.length > 0 && (
          <div className="primer-results">
            <h4>Designed Primers</h4>
            <div className="primer-list">
              {primers.map((primer) => (
                <div key={primer.id} className="primer-item">
                  <div className="primer-header">
                    <span className="primer-name">{primer.name}</span>
                    <span className="primer-length">{primer.length} bp</span>
                  </div>
                  <div className="primer-sequence">{primer.sequence}</div>
                  <div className="primer-stats">
                    <span>Tm: {primer.tm.toFixed(1)}°C</span>
                    <span>GC: {(primer.gc * 100).toFixed(0)}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
