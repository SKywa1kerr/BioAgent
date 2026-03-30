import React, { useRef, useEffect, useMemo } from "react";
import { Mutation, ChromatogramData } from "../types";
import {
  complementStrand,
  translateDNA,
  findRestrictionSites,
  groupRestrictionSites,
  baseColor,
  AminoAcid,
} from "../utils/sequence";
import "./SequenceViewer.css";

interface SequenceViewerProps {
  refSequence: string;
  querySequence: string;
  alignedQuery: string;
  matches: boolean[];
  mutations: Mutation[];
  chromatogramData: ChromatogramData | null;
  cdsStart: number;
  cdsEnd: number;
  featureName?: string;
}

const BASES_PER_LINE = 80;
const TRACE_HEIGHT = 100;

// Single-letter to 3-letter amino acid mapping
const AA_THREE: Record<string, string> = {
  A: "Ala", R: "Arg", N: "Asn", D: "Asp", C: "Cys",
  E: "Glu", Q: "Gln", G: "Gly", H: "His", I: "Ile",
  L: "Leu", K: "Lys", M: "Met", F: "Phe", P: "Pro",
  S: "Ser", T: "Thr", W: "Trp", Y: "Tyr", V: "Val",
  "*": "***", "?": "???",
};

const ChromatogramBlock: React.FC<{
  data: ChromatogramData;
  startBase: number;
  numBases: number;
  baseWidth: number;
}> = ({ data, startBase, numBases, baseWidth }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const width = numBases * baseWidth;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data.traces) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = TRACE_HEIGHT * dpr;
    ctx.scale(dpr, dpr);

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, TRACE_HEIGHT);

    const traces = data.traces;
    const totalTracePoints = traces.A.length;
    const totalBases = data.baseCalls.length;
    if (totalBases === 0 || totalTracePoints === 0) return;

    // Calculate trace points per base
    const pointsPerBase = totalTracePoints / totalBases;
    const traceStart = Math.floor(startBase * pointsPerBase);
    const traceEnd = Math.min(
      Math.floor((startBase + numBases) * pointsPerBase),
      totalTracePoints
    );
    const traceRange = traceEnd - traceStart;

    if (traceRange <= 0) return;

    // Find max value for scaling
    let maxVal = 0;
    const colors = { A: "#00aa00", T: "#aa0000", G: "#000000", C: "#0000aa" };
    for (const base of ["A", "T", "G", "C"] as const) {
      const trace = traces[base];
      for (let i = traceStart; i < traceEnd; i++) {
        if (trace[i] > maxVal) maxVal = trace[i];
      }
    }
    if (maxVal === 0) return;

    const padding = 4;
    const yScale = (TRACE_HEIGHT - 2 * padding) / (maxVal * 1.1);

    for (const base of ["A", "T", "G", "C"] as const) {
      const trace = traces[base];
      ctx.strokeStyle = colors[base];
      ctx.lineWidth = 1.2;
      ctx.beginPath();

      for (let i = traceStart; i < traceEnd; i++) {
        const x = ((i - traceStart) / traceRange) * width;
        const y = TRACE_HEIGHT - padding - trace[i] * yScale;
        if (i === traceStart) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // Draw base call letters at the bottom
    ctx.font = "9px Monaco, Consolas, monospace";
    ctx.textAlign = "center";
    for (let i = 0; i < numBases; i++) {
      const baseIdx = startBase + i;
      if (baseIdx < data.baseCalls.length) {
        const b = data.baseCalls[baseIdx];
        ctx.fillStyle = colors[b as keyof typeof colors] || "#888";
        ctx.fillText(b, (i + 0.5) * baseWidth, TRACE_HEIGHT - 2);
      }
    }
  }, [data, startBase, numBases, baseWidth, width]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: `${width}px`, height: `${TRACE_HEIGHT}px` }}
      className="trace-canvas"
    />
  );
};

export const SequenceViewer: React.FC<SequenceViewerProps> = ({
  refSequence,
  alignedQuery,
  matches,
  mutations,
  chromatogramData,
  cdsStart,
  cdsEnd,
  featureName = "CDS",
}) => {
  const complement = useMemo(() => complementStrand(refSequence), [refSequence]);

  const aminoAcids = useMemo(() => {
    if (cdsStart < 0 || cdsEnd < 0) return [];
    const cdsSeq = refSequence.slice(cdsStart, cdsEnd);
    return translateDNA(cdsSeq, 0).map((aa) => ({
      ...aa,
      position: aa.position + cdsStart,
    }));
  }, [refSequence, cdsStart, cdsEnd]);

  const restrictionSites = useMemo(
    () => findRestrictionSites(refSequence),
    [refSequence]
  );

  const siteGroups = useMemo(
    () => groupRestrictionSites(restrictionSites),
    [restrictionSites]
  );

  // Create mutation position set for quick lookup
  const mutationPositions = useMemo(() => {
    const set = new Set<number>();
    for (const m of mutations) {
      set.add(m.position);
    }
    return set;
  }, [mutations]);

  // Split into blocks
  const totalBases = refSequence.length;
  const numBlocks = Math.ceil(totalBases / BASES_PER_LINE);

  const renderBlock = (blockIdx: number) => {
    const blockStart = blockIdx * BASES_PER_LINE;
    const blockEnd = Math.min(blockStart + BASES_PER_LINE, totalBases);
    const blockLength = blockEnd - blockStart;
    const refSlice = refSequence.slice(blockStart, blockEnd);
    const compSlice = complement.slice(blockStart, blockEnd);
    const querySlice = alignedQuery
      ? alignedQuery.slice(blockStart, blockEnd)
      : "";
    const matchSlice = matches ? matches.slice(blockStart, blockEnd) : [];

    // Restriction sites in this block
    const blockSites: { position: number; names: string[] }[] = [];
    siteGroups.forEach((names, pos) => {
      if (pos >= blockStart && pos < blockEnd) {
        blockSites.push({ position: pos - blockStart, names });
      }
    });

    // Amino acids in this block
    const blockAAs: AminoAcid[] = aminoAcids.filter(
      (aa) => aa.position >= blockStart && aa.position < blockEnd
    );

    // Is CDS feature in this block?
    const featureInBlock =
      cdsStart < blockEnd && cdsEnd > blockStart && cdsStart >= 0;
    const featureStart = Math.max(0, cdsStart - blockStart);
    const featureEnd = Math.min(blockLength, cdsEnd - blockStart);

    // Right margin line number (like SnapGene shows bp count on right)
    const rightBp = blockEnd;

    return (
      <div key={blockIdx} className="sequence-block">
        {/* Restriction enzyme labels */}
        {blockSites.length > 0 && (
          <div className="enzyme-row">
            {blockSites.map((site, i) => (
              <span
                key={i}
                className="enzyme-label"
                style={{ left: `calc(60px + ${site.position} * var(--base-width))` }}
              >
                {site.names.join("\n")}
              </span>
            ))}
          </div>
        )}

        {/* Forward strand (reference) */}
        <div className="strand-row">
          <span className="row-gutter" />
          <span className="strand-bases">
            {refSlice.split("").map((base, i) => {
              const globalPos = blockStart + i;
              const isMutation = mutationPositions.has(globalPos);
              return (
                <span
                  key={i}
                  className={`base-char ${isMutation ? "mutation" : ""}`}
                  style={{ color: isMutation ? "#fff" : baseColor(base) }}
                >
                  {base}
                </span>
              );
            })}
          </span>
          <span className="row-bp-count">{rightBp}</span>
        </div>

        {/* Tick marks between strands */}
        <div className="strand-row tick-row">
          <span className="row-gutter" />
          <span className="strand-bases">
            {refSlice.split("").map((_, i) => {
              const pos = blockStart + i + 1;
              if (pos % 10 === 0) return <span key={i} className="tick-major">+</span>;
              return <span key={i} className="tick-minor">·</span>;
            })}
          </span>
        </div>

        {/* Reverse strand (complement) */}
        <div className="strand-row">
          <span className="row-gutter" />
          <span className="strand-bases">
            {compSlice.split("").map((base, i) => (
              <span key={i} className="base-char" style={{ color: baseColor(base) }}>
                {base}
              </span>
            ))}
          </span>
        </div>

        {/* Feature annotation bar */}
        {featureInBlock && (
          <div className="feature-row">
            <span className="row-gutter" />
            <div className="feature-track">
              <div
                className="feature-bar"
                style={{
                  left: `calc(${featureStart} * var(--base-width))`,
                  width: `calc(${featureEnd - featureStart} * var(--base-width))`,
                }}
              >
                <span className="feature-name">{featureName}</span>
                <span className="feature-arrow">{">"}</span>
              </div>
            </div>
          </div>
        )}

        {/* Amino acid translation */}
        {blockAAs.length > 0 && (
          <div className="aa-row">
            <span className="row-gutter" />
            <span className="aa-bases">
              {Array.from({ length: blockLength }, (_, i) => {
                const globalPos = blockStart + i;
                const aa = blockAAs.find((a) => a.position === globalPos);
                if (aa) {
                  const isMutation = mutationPositions.has(globalPos) ||
                    mutationPositions.has(globalPos + 1) ||
                    mutationPositions.has(globalPos + 2);
                  return (
                    <span
                      key={i}
                      className={`aa-char ${isMutation ? "aa-mutation" : ""}`}
                      title={`${AA_THREE[aa.aa] || aa.aa} (${aa.codon})`}
                    >
                      {aa.aa}
                    </span>
                  );
                }
                return <span key={i} className="aa-char aa-spacer" />;
              })}
            </span>
          </div>
        )}

        {/* Position numbers */}
        <div className="position-row">
          <span className="row-gutter" />
          <span className="position-bases">
            {Array.from({ length: blockLength }, (_, i) => {
              const pos = blockStart + i + 1;
              if (pos % 5 === 0) {
                return (
                  <span key={i} className="pos-mark">
                    {pos}
                  </span>
                );
              }
              return <span key={i} className="pos-spacer" />;
            })}
          </span>
        </div>

        {/* Chromatogram trace */}
        {chromatogramData && chromatogramData.traces && (
          <div className="trace-row">
            <span className="row-gutter" />
            <ChromatogramBlock
              data={chromatogramData}
              startBase={blockStart}
              numBases={blockLength}
              baseWidth={10.4}
            />
          </div>
        )}

        {/* Query alignment with mismatch highlighting */}
        {querySlice && (
          <div className="strand-row query-row">
            <span className="row-gutter query-label">Query</span>
            <span className="strand-bases">
              {querySlice.split("").map((base, i) => {
                const isMatch = matchSlice[i];
                const globalPos = blockStart + i;
                const isMutation = mutationPositions.has(globalPos);
                return (
                  <span
                    key={i}
                    className={`base-char ${!isMatch ? "mismatch" : ""} ${isMutation ? "mutation" : ""}`}
                    style={{
                      color: !isMatch || isMutation ? "#fff" : baseColor(base),
                    }}
                  >
                    {base}
                  </span>
                );
              })}
            </span>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="sequence-viewer">
      {/* Minimap ruler */}
      <div className="minimap">
        <div className="minimap-track">
          {/* Scale markers */}
          {Array.from({ length: Math.ceil(totalBases / 10) }, (_, i) => {
            const pos = i * 10;
            return (
              <span
                key={i}
                className="minimap-tick"
                style={{ left: `${(pos / totalBases) * 100}%` }}
              >
                {pos % 50 === 0 ? <span className="minimap-label">{pos}</span> : null}
              </span>
            );
          })}
          {/* Feature region on minimap */}
          {cdsStart >= 0 && cdsEnd > cdsStart && (
            <div
              className="minimap-feature"
              style={{
                left: `${(cdsStart / totalBases) * 100}%`,
                width: `${((cdsEnd - cdsStart) / totalBases) * 100}%`,
              }}
            />
          )}
          {/* Mutation markers on minimap */}
          {mutations.map((m, i) => (
            <div
              key={i}
              className="minimap-mutation"
              style={{ left: `${(m.position / totalBases) * 100}%` }}
            />
          ))}
        </div>
      </div>

      {/* Sequence blocks */}
      <div className="sequence-blocks">
        {Array.from({ length: numBlocks }, (_, i) => renderBlock(i))}
      </div>
    </div>
  );
};
