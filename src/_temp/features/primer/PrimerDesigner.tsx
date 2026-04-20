import { useState, useEffect } from "react";
import { Sample, Mutation } from "../types";
import { PrimerSkill, PrimerResult } from "../types/primer";
import "./PrimerDesigner.css";

interface PrimerDesignerProps {
  sample: Sample;
  mutations: Mutation[];
}

export function PrimerDesigner({ sample, mutations }: PrimerDesignerProps) {
  const [skills, setSkills] = useState<PrimerSkill[]>([]);
  const [selectedSkill, setSelectedSkill] = useState<string>("");
  const [isDesigning, setIsDesigning] = useState(false);
  const [result, setResult] = useState<PrimerResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showPrimerDesign, setShowPrimerDesign] = useState(true);

  // Load available skills on mount
  useEffect(() => {
    loadSkills();
  }, []);

  // Auto-select skill based on mutation context
  useEffect(() => {
    if (skills.length === 0) return;

    if (mutations.length === 1) {
      const sssm = skills.find(s => s.tags.includes("SSSM"));
      if (sssm) setSelectedSkill(sssm.name);
    } else if (mutations.length > 1 && mutations.length <= 3) {
      const msdm = skills.find(s => s.tags.includes("MSDM"));
      if (msdm) setSelectedSkill(msdm.name);
    } else if (mutations.length > 3) {
      const pas = skills.find(s => s.tags.includes("PAS"));
      if (pas) setSelectedSkill(pas.name);
    }
  }, [skills, mutations]);

  const loadSkills = async () => {
    try {
      const skillsJson = await window.electronAPI.invoke("list-primer-skills");
      const skillsList: PrimerSkill[] = JSON.parse(skillsJson as string);
      setSkills(skillsList);
    } catch (err) {
      console.error("Failed to load primer skills:", err);
      setError("Could not load primer design skills");
    }
  };

  const handleDesignPrimers = async () => {
    if (!selectedSkill) return;

    setIsDesigning(true);
    setError(null);

    try {
      // Convert mutations to targets
      const targets = mutations.map(mut => ({
        position: mut.position,
        original_codon: mut.refCodon || "XXX",
        target_codons: [mut.queryCodon || "XXX"],
        strategy: selectedSkill.includes("sssm") ? "SSSM" :
                 selectedSkill.includes("msdm") ? "MSDM" : "PAS"
      }));

      const params = {
        sequence: sample.refSequence,
        targets
      };

      const resultJson = await window.electronAPI.invoke(
        "design-primers",
        selectedSkill,
        params
      );

      const primerResult: PrimerResult = JSON.parse(resultJson as string);
      setResult(primerResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Primer design failed");
    } finally {
      setIsDesigning(false);
    }
  };

  const recommendedSkill = (() => {
    if (mutations.length === 1) return skills.find(s => s.tags.includes("SSSM"));
    if (mutations.length > 1 && mutations.length <= 3) return skills.find(s => s.tags.includes("MSDM"));
    if (mutations.length > 3) return skills.find(s => s.tags.includes("PAS"));
    return null;
  })();

  if (!showPrimerDesign) {
    return (
      <div className="primer-designer-collapsed">
        <button onClick={() => setShowPrimerDesign(true)} className="btn-link">
          ▶ Show Primer Design
        </button>
      </div>
    );
  }

  return (
    <div className="primer-designer">
      <div className="primer-header">
        <h4>Primer Design</h4>
        <button onClick={() => setShowPrimerDesign(false)} className="btn-link">
          ▼ Hide
        </button>
      </div>

      {skills.length === 0 ? (
        <p>Loading primer design skills...</p>
      ) : (
        <>
          <div className="skill-selection">
            <label>Design Strategy:</label>
            <div className="skill-options">
              {skills.map(skill => (
                <div key={skill.name} className="skill-option">
                  <input
                    type="radio"
                    id={skill.name}
                    name="primer-skill"
                    value={skill.name}
                    checked={selectedSkill === skill.name}
                    onChange={() => setSelectedSkill(skill.name)}
                  />
                  <label htmlFor={skill.name}>
                    <strong>{skill.name.replace("design_", "").replace("_primers", "")}</strong>
                    <br />
                    <small>{skill.description}</small>
                    {recommendedSkill?.name === skill.name && (
                      <span className="recommended-badge">Recommended</span>
                    )}
                  </label>
                </div>
              ))}
            </div>

            {recommendedSkill && selectedSkill !== recommendedSkill.name && (
              <div className="recommendation">
                <p>
                  Based on {mutations.length} mutation(s), we recommend using{" "}
                  <button
                    className="btn-link"
                    onClick={() => setSelectedSkill(recommendedSkill.name)}
                  >
                    {recommendedSkill.name}
                  </button>
                </p>
              </div>
            )}
          </div>

          <button
            onClick={handleDesignPrimers}
            disabled={isDesigning || !selectedSkill}
            className="btn-primary"
          >
            {isDesigning ? "Designing..." : "Design Primers"}
          </button>

          {error && <div className="error-message">{error}</div>}

          {result && (
            <div className="primer-results">
              <h5>Designed Primers ({result.primers.length})</h5>
              {result.primers.length === 0 ? (
                <p>No primers designed (stub implementation)</p>
              ) : (
                <table className="primer-table">
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Sequence</th>
                      <th>Tm (°C)</th>
                      <th>GC%</th>
                      <th>Position</th>
                      <th>Direction</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.primers.map((primer, idx) => (
                      <tr key={idx}>
                        <td>{primer.name}</td>
                        <td className="monospace">{primer.sequence}</td>
                        <td>{primer.tm.toFixed(1)}</td>
                        <td>{primer.gc_content.toFixed(1)}%</td>
                        <td>{primer.position}</td>
                        <td>{primer.direction}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              {result.warnings.length > 0 && (
                <div className="warnings">
                  <strong>Warnings:</strong>
                  <ul>
                    {result.warnings.map((w, i) => <li key={i}>{w}</li>)}
                  </ul>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}