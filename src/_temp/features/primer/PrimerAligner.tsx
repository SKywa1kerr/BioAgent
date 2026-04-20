import { useState } from "react";
import { PrimerAlignment, PrimerAlignmentDisplay, PrimerAlignInput } from "../types/primer";
import "./PrimerAligner.css";

interface PrimerAlignerProps {
  referenceSequence: string;
  onClose: () => void;
}

const VISIBLE_BASES = 100; // Number of bases to show per row

export function PrimerAligner({ referenceSequence, onClose }: PrimerAlignerProps) {
  const [primers, setPrimers] = useState<PrimerAlignInput[]>([
    { name: "", sequence: "" }
  ]);
  const [alignments, setAlignments] = useState<PrimerAlignmentDisplay[] | null>(null);
  const [isAligning, setIsAligning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refLength = referenceSequence.length;

  const addPrimer = () => {
    setPrimers([...primers, { name: "", sequence: "" }]);
  };

  const removePrimer = (index: number) => {
    setPrimers(primers.filter((_, i) => i !== index));
  };

  const updatePrimer = (index: number, field: keyof PrimerAlignInput, value: string) => {
    const newPrimers = [...primers];
    newPrimers[index] = { ...newPrimers[index], [field]: value };
    setPrimers(newPrimers);
  };

  const handleAlign = async () => {
    const validPrimers = primers.filter(p => p.sequence.trim().length >= 10);
    if (validPrimers.length === 0) {
      setError("Please enter at least one primer sequence (minimum 10 bases)");
      return;
    }

    setIsAligning(true);
    setError(null);

    try {
      const sequences = validPrimers.map(p => p.sequence.trim().toUpperCase());
      const names = validPrimers.map(p => p.name.trim() || `Primer ${validPrimers.indexOf(p) + 1}`);

      const result = await window.electronAPI.invoke(
        "align-primers",
        referenceSequence,
        sequences,
        names
      );

      const parsed: PrimerAlignment[] = JSON.parse(result as string);

      // Parse name from primer_sequence if it contains ":"
      const withDisplay: PrimerAlignmentDisplay[] = parsed.map((a, i) => {
        const name = names[i] || `Primer ${i + 1}`;
        const displaySequence = a.primer_sequence.includes(":")
          ? a.primer_sequence.split(":")[1]
          : a.primer_sequence;
        return { ...a, name, displaySequence };
      });

      setAlignments(withDisplay);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Alignment failed");
    } finally {
      setIsAligning(false);
    }
  };

  // Split alignment into chunks for display
  const getAlignmentChunks = (alignment: PrimerAlignmentDisplay) => {
    const chunks = [];
    const start = alignment.start;
    const end = alignment.end;

    for (let chunkStart = 0; chunkStart < refLength; chunkStart += VISIBLE_BASES) {
      const chunkEnd = Math.min(chunkStart + VISIBLE_BASES, refLength);

      // Check if this chunk overlaps with the primer
      if (chunkEnd < start || chunkStart > end) {
        continue; // Skip chunks that don't overlap with primer
      }

      // Build primer sequence for this chunk
      const primerBases: { char: string; isMismatch: boolean }[] = [];
      const geneBases: string[] = [];
      const positions: number[] = [];

      for (let i = chunkStart; i < chunkEnd; i++) {
        positions.push(i + 1); // 1-based position

        // Gene base
        geneBases.push(referenceSequence[i] || " ");

        // Primer base (if within primer range)
        if (i >= start && i < end) {
          const primerIdx = i - start;
          const primerBase = alignment.direction === "forward"
            ? alignment.displaySequence[primerIdx]
            : alignment.displaySequence[alignment.displaySequence.length - 1 - primerIdx];
          const isMismatch = alignment.mismatches.includes(i);
          primerBases.push({ char: primerBase || " ", isMismatch });
        } else {
          primerBases.push({ char: " ", isMismatch: false });
        }
      }

      chunks.push({
        chunkStart,
        chunkEnd,
        positions,
        primerBases,
        geneBases,
      });
    }

    return chunks;
  };

  // Generate position labels for a chunk
  const getPositionLabels = (chunkStart: number, chunkEnd: number) => {
    const labels: string[] = [];
    for (let i = chunkStart; i < chunkEnd; i++) {
      const pos = i + 1;
      if (pos % 10 === 0) {
        labels.push(String(pos));
      } else {
        labels.push("");
      }
    }
    return labels;
  };

  return (
    <div className="primer-aligner-overlay" onClick={onClose}>
      <div className="primer-aligner-modal" onClick={e => e.stopPropagation()}>
        <div className="primer-aligner-header">
          <h3>Primer Alignment Tool</h3>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>

        <div className="primer-aligner-content">
          <div className="primer-input-section">
            <label>Enter Primer Sequences:</label>
            {primers.map((primer, index) => (
              <div key={index} className="primer-input-row">
                <input
                  type="text"
                  className="primer-name-input"
                  placeholder={`Name`}
                  value={primer.name}
                  onChange={e => updatePrimer(index, "name", e.target.value)}
                />
                <input
                  type="text"
                  className="primer-seq-input"
                  placeholder={`Primer ${index + 1} sequence (e.g., CGGTACCGAC...)`}
                  value={primer.sequence}
                  onChange={e => updatePrimer(index, "sequence", e.target.value)}
                />
                {primers.length > 1 && (
                  <button
                    className="remove-primer-btn"
                    onClick={() => removePrimer(index)}
                  >
                    Remove
                  </button>
                )}
              </div>
            ))}
            <button className="add-primer-btn" onClick={addPrimer}>
              + Add Primer
            </button>

            <button
              className="align-btn"
              onClick={handleAlign}
              disabled={isAligning}
            >
              {isAligning ? "Aligning..." : "Align Primers"}
            </button>

            {error && <div className="error-message">{error}</div>}
          </div>

          {alignments && (
            <div className="primer-visualization">
              <h4>Alignment Results</h4>

              {alignments.length === 0 ? (
                <div className="no-alignments">
                  No primers aligned successfully. Check that your sequences are valid and match the reference.
                </div>
              ) : (
                <>
                  {alignments.map((alignment, idx) => (
                    <div key={idx} className="primer-alignment-block">
                      <div className="primer-alignment-header">
                        <h5>{alignment.name} (Tm: {alignment.tm.toFixed(1)}°C)</h5>
                        <div className="primer-stats">
                          <span>Position: {alignment.start + 1}-{alignment.end}</span>
                          <span>Direction: {alignment.direction}</span>
                          <span>Match: {alignment.alignment_score.toFixed(1)}%</span>
                          {alignment.mismatches.length > 0 && (
                            <span>Mismatches: {alignment.mismatches.length}</span>
                          )}
                        </div>
                      </div>

                      <div className="alignment-display">
                        {getAlignmentChunks(alignment).map((chunk, chunkIdx) => (
                          <div key={chunkIdx}>
                            {/* Primer sequence row */}
                            <div className={`alignment-row primer-row ${alignment.direction}`}>
                              <span className="alignment-label">
                                {alignment.name}
                              </span>
                              <span className="alignment-sequence">
                                {chunk.primerBases.map((base, i) => (
                                  <span
                                    key={i}
                                    className={`alignment-base ${base.isMismatch ? 'mismatch' : ''}`}
                                  >
                                    {base.char}
                                  </span>
                                ))}
                              </span>
                            </div>

                            {/* Gene sequence row */}
                            <div className="alignment-row gene-row">
                              <span className="alignment-label">Gene</span>
                              <span className="alignment-sequence">
                                {chunk.geneBases.map((base, i) => (
                                  <span key={i} className="alignment-base">
                                    {base}
                                  </span>
                                ))}
                              </span>
                            </div>

                            {/* Position row */}
                            <div className="alignment-row position-row">
                              <span className="alignment-label"></span>
                              <span className="alignment-sequence">
                                {getPositionLabels(chunk.chunkStart, chunk.chunkEnd).map((label, i) => (
                                  <span
                                    key={i}
                                    className={`alignment-base ${label ? 'tick-major' : ''}`}
                                  >
                                    {label ? label[0] : '·'}
                                  </span>
                                ))}
                              </span>
                            </div>

                            {chunkIdx < getAlignmentChunks(alignment).length - 1 && (
                              <div style={{ height: '16px' }} />
                            )}
                          </div>
                        ))}
                      </div>

                      {/* Primer info summary */}
                      <div className="primer-info-box">
                        <div className="primer-info-item">
                          <span className="primer-info-label">Tm:</span>
                          <span className="primer-info-value tm">{alignment.tm.toFixed(1)}°C</span>
                        </div>
                        <div className="primer-info-item">
                          <span className="primer-info-label">Direction:</span>
                          <span className={`primer-info-value direction-${alignment.direction}`}>
                            {alignment.direction === "forward" ? "Forward →" : "← Reverse"}
                          </span>
                        </div>
                        <div className="primer-info-item">
                          <span className="primer-info-label">Position:</span>
                          <span className="primer-info-value">{alignment.start + 1} - {alignment.end}</span>
                        </div>
                        <div className="primer-info-item">
                          <span className="primer-info-label">Match:</span>
                          <span className="primer-info-value">{alignment.match_count}/{alignment.end - alignment.start} ({alignment.alignment_score.toFixed(1)}%)</span>
                        </div>
                        {alignment.mismatches.length > 0 && (
                          <div className="primer-info-item">
                            <span className="primer-info-label">Mismatches at:</span>
                            <span className="primer-info-value">
                              {alignment.mismatches.map(m => m + 1).join(", ")}
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
