import React, { useRef, useEffect, useMemo, useState, memo } from "react";
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
  alignedRefG?: string;
  alignedQueryG?: string;
  alignedQuery: string;
  matches: boolean[];
  mutations: Mutation[];
  chromatogramData: ChromatogramData | null;
  cdsStart: number;
  cdsEnd: number;
  featureName?: string;
}

const BASE_WIDTH = 10.4;
const TRACE_HEIGHT = 140;

// Single-letter to 3-letter amino acid mapping
const AA_THREE: Record<string, string> = {
  A: "Ala", R: "Arg", N: "Asn", D: "Asp", C: "Cys",
  E: "Glu", Q: "Gln", G: "Gly", H: "His", I: "Ile",
  L: "Leu", K: "Lys", M: "Met", F: "Phe", P: "Pro",
  S: "Ser", T: "Thr", W: "Trp", Y: "Tyr", V: "Val",
  "*": "***", "?": "???",
};

const ChromatogramView = memo(({ 
  data, 
  totalBases, 
  alignedQueryG, 
  gappedToQueryIdx 
}: {
  data: ChromatogramData;
  totalBases: number;
  alignedQueryG: string;
  gappedToQueryIdx: (number | null)[];
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const width = totalBases * BASE_WIDTH;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data.traces) return;
    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = TRACE_HEIGHT * dpr;
    ctx.scale(dpr, dpr);

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, TRACE_HEIGHT);

    const traces = data.traces;
    const totalTracePoints = traces.A?.length || 0;
    if (totalTracePoints === 0) return;

    // Find max value for scaling
    let maxVal = 0;
    const colors = { A: "#00aa00", T: "#aa0000", G: "#000000", C: "#0000aa" };
    for (const base of ["A", "T", "G", "C"] as const) {
      const trace = traces[base];
      for (let i = 0; i < totalTracePoints; i++) {
        if (trace[i] > maxVal) maxVal = trace[i];
      }
    }
    if (maxVal === 0) maxVal = 100;

    const yScale = (TRACE_HEIGHT - 35) / (maxVal * 1.1);

    // 1. Draw Quality Bars (SnapGene style)
    if (data.quality) {
      for (let i = 0; i < totalBases; i++) {
        const qIdx = gappedToQueryIdx[i];
        if (qIdx !== null && qIdx < data.quality.length) {
          const q = data.quality[qIdx];
          if (q >= 40) ctx.fillStyle = "#e2e8f0";
          else if (q >= 20) ctx.fillStyle = "#fef3c7";
          else ctx.fillStyle = "#fee2e2";
          
          const barHeight = Math.min(1, q / 60) * (TRACE_HEIGHT - 45);
          const x = i * BASE_WIDTH + 1;
          ctx.fillRect(x, TRACE_HEIGHT - 25 - barHeight, BASE_WIDTH - 2, barHeight);
        }
      }
    }

    // 2. Draw Traces
    if (data.base_locations && data.base_locations.length > 0) {
      const queryToGapped: number[] = [];
      for (let i = 0; i < gappedToQueryIdx.length; i++) {
        const qIdx = gappedToQueryIdx[i];
        if (qIdx !== null) queryToGapped[qIdx] = i;
      }

      for (const base of ["A", "T", "G", "C"] as const) {
        const trace = traces[base];
        ctx.strokeStyle = colors[base];
        ctx.lineWidth = 1.2;
        ctx.beginPath();

        let b1 = 0;
        for (let i = 0; i < totalTracePoints; i++) {
          while (b1 < data.base_locations.length - 1 && data.base_locations[b1+1] < i) {
            b1++;
          }
          
          const t1 = data.base_locations[b1];
          const t2 = data.base_locations[b1+1] || totalTracePoints;
          const g1 = queryToGapped[b1] ?? 0;
          const g2 = queryToGapped[b1+1] ?? (g1 + 1);

          const ratio = (i - t1) / (t2 - t1);
          const gappedX = g1 + (g2 - g1) * ratio;
          const x = (gappedX + 0.5) * BASE_WIDTH;
          const y = TRACE_HEIGHT - 25 - trace[i] * yScale;

          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
    }

    // 3. Draw Base Call Letters
    ctx.font = "bold 10px Monaco, Consolas, monospace";
    ctx.textAlign = "center";
    for (let i = 0; i < totalBases; i++) {
      const b = alignedQueryG[i];
      if (b === "-" || !b) continue;

      const x = (i + 0.5) * BASE_WIDTH;
      ctx.fillStyle = colors[b as keyof typeof colors] || "#888";
      ctx.fillText(b, x, TRACE_HEIGHT - 8);

      const qIdx = gappedToQueryIdx[i];
      if (qIdx !== null && data.mixed_peaks && data.mixed_peaks.includes(qIdx)) {
        ctx.strokeStyle = "#eab308";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(x, TRACE_HEIGHT - 18, 4, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }, [data, totalBases, alignedQueryG, gappedToQueryIdx, width]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: `${width}px`, height: `${TRACE_HEIGHT}px` }}
      className="trace-canvas-full"
    />
  );
});

export const SequenceViewer: React.FC<SequenceViewerProps> = memo(({
  refSequence = "",
  querySequence = "",
  alignedRefG = "",
  alignedQueryG = "",
  matches = [],
  mutations = [],
  chromatogramData,
  cdsStart,
  cdsEnd,
  featureName = "CDS",
}) => {
  const displayRef = alignedRefG || refSequence;
  const displayQuery = alignedQueryG || querySequence;
  const totalBases = displayRef?.length || 0;

  const [hoverInfo, setHoverInfo] = useState<{ idx: number | null; quality: number | null; x: number; y: number }>({
    idx: null,
    quality: null,
    x: 0,
    y: 0
  });

  const refToGapped = useMemo(() => {
    const map = new Array(refSequence.length).fill(0);
    let refIdx = 0;
    for (let i = 0; i < displayRef.length; i++) {
      if (displayRef[i] !== "-") {
        map[refIdx] = i;
        refIdx++;
      }
    }
    return map;
  }, [refSequence, displayRef]);

  const gappedToQueryIdx = useMemo(() => {
    const map: (number | null)[] = [];
    let qIdx = 0;
    for (let i = 0; i < displayQuery.length; i++) {
      if (displayQuery[i] !== "-") {
        map[i] = qIdx;
        qIdx++;
      } else {
        map[i] = null;
      }
    }
    return map;
  }, [displayQuery]);

  const aminoAcids = useMemo(() => {
    if (cdsStart <= 0 || cdsEnd <= 0 || !refSequence) return [];
    const cdsSeq = refSequence.slice(cdsStart - 1, cdsEnd);
    return translateDNA(cdsSeq, 0).map((aa) => ({
      ...aa,
      position: refToGapped[aa.position + cdsStart - 1] ?? (aa.position + cdsStart - 1),
    }));
  }, [refSequence, cdsStart, cdsEnd, refToGapped]);

  const restrictionSites = useMemo(() => {
    const sites = findRestrictionSites(refSequence);
    return sites.map(s => ({
      ...s,
      position: refToGapped[s.position] ?? s.position
    }));
  }, [refSequence, refToGapped]);

  const siteGroups = useMemo(() => groupRestrictionSites(restrictionSites), [restrictionSites]);

  const gappedCdsStart = useMemo(() => refToGapped[cdsStart - 1] ?? (cdsStart - 1), [cdsStart, refToGapped]);
  const gappedCdsEnd = useMemo(() => {
    const lastIdx = refToGapped[cdsEnd - 1] ?? (cdsEnd - 1);
    return lastIdx + 1;
  }, [cdsEnd, refToGapped]);

  const mutationPositions = useMemo(() => {
    const set = new Set<number>();
    for (const m of mutations) {
      const gappedIdx = refToGapped[m.position - 1];
      if (gappedIdx !== undefined) set.add(gappedIdx);
    }
    return set;
  }, [mutations, refToGapped]);

  const blockSites: { position: number; names: string[] }[] = [];
  siteGroups.forEach((names, pos) => {
    blockSites.push({ position: pos, names });
  });

  const featureInView = gappedCdsStart >= 0;

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!chromatogramData) return;
    
    const container = e.currentTarget as HTMLElement;
    const rect = container.getBoundingClientRect();
    const x = e.clientX - rect.left - 100; // 100 is the gutter width
    
    if (x < 0) {
      if (hoverInfo.idx !== null) setHoverInfo(prev => ({ ...prev, idx: null }));
      return;
    }

    const idx = Math.floor(x / BASE_WIDTH);
    if (idx >= 0 && idx < totalBases) {
      const qIdx = gappedToQueryIdx[idx];
      const quality = (qIdx !== null && chromatogramData.quality) ? chromatogramData.quality[qIdx] : null;
      
      // Only update if index changed to avoid excessive re-renders
      if (idx !== hoverInfo.idx) {
        setHoverInfo({
          idx,
          quality,
          x: idx * BASE_WIDTH + 100 + BASE_WIDTH / 2,
          y: e.clientY - rect.top - 20 
        });
      }
    } else if (hoverInfo.idx !== null) {
      setHoverInfo(prev => ({ ...prev, idx: null }));
    }
  };

  const handleMouseLeave = () => {
    setHoverInfo(prev => ({ ...prev, idx: null }));
  };

  return (
    <div className="sequence-viewer horizontal">
      <div 
        className="sequence-container" 
        style={{ width: `${totalBases * BASE_WIDTH + 100}px` }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      >
        
        {/* 1. Restriction enzyme labels */}
        <div className="enzyme-row">
          <span className="row-gutter sticky-gutter">Enzymes</span>
          <div className="row-content">
            {blockSites.map((site, i) => (
              <span
                key={i}
                className="enzyme-label"
                style={{ left: `${site.position * BASE_WIDTH + BASE_WIDTH/2}px` }}
              >
                {site.names.join("\n")}
              </span>
            ))}
          </div>
        </div>

        {/* 2. Feature annotation bar */}
        {featureInView && (
          <div className="feature-row">
            <span className="row-gutter sticky-gutter">Features</span>
            <div className="row-content feature-track">
              <div
                className="feature-bar"
                style={{
                  left: `${gappedCdsStart * BASE_WIDTH}px`,
                  width: `${(gappedCdsEnd - gappedCdsStart) * BASE_WIDTH}px`,
                }}
              >
                <span className="feature-name">{featureName}</span>
                <span className="feature-arrow">{">"}</span>
              </div>
            </div>
          </div>
        )}

        {/* 3. SnapGene style alignment block (Ref / Matches / Query) */}
        <div className="alignment-block">
          {/* Reference line */}
          <div className="strand-row ref-line">
            <span className="row-gutter sticky-gutter">Ref</span>
            <div className="row-content">
              {displayRef.split("").map((base, i) => {
                const isMatch = matches[i];
                const isMutation = mutationPositions.has(i);
                const highlight = !isMatch || isMutation;
                return (
                  <span
                    key={i}
                    className={`base-char ${!isMatch ? "mismatch-ref" : ""} ${isMutation ? "mutation-ref" : ""}`}
                    style={{ color: highlight ? undefined : baseColor(base) }}
                    data-base={base}
                  >
                    {base}
                  </span>
                );
              })}
            </div>
          </div>

          {/* Match indicators */}
          <div className="strand-row match-indicators">
            <span className="row-gutter sticky-gutter" />
            <div className="row-content">
              {matches.map((match, i) => (
                <span key={i} className="match-char">
                  {match ? "|" : " "}
                </span>
              ))}
            </div>
          </div>

          {/* Query alignment (Sanger result) */}
          <div className="strand-row query-row">
            <span className="row-gutter sticky-gutter query-label">Sanger</span>
            <div className="row-content">
              {displayQuery.split("").map((base, i) => {
                const isMatch = matches[i];
                const isMutation = mutationPositions.has(i);
                const highlight = !isMatch || isMutation;
                return (
                  <span
                    key={i}
                    className={`base-char ${!isMatch ? "mismatch" : ""} ${isMutation ? "mutation" : ""}`}
                    style={{ color: highlight ? undefined : baseColor(base) }}
                    data-base={base}
                  >
                    {base}
                  </span>
                );
              })}
            </div>
          </div>
        </div>

        {/* 4. Amino acid translation */}
        <div className="aa-row">
          <span className="row-gutter sticky-gutter">AA</span>
          <div className="row-content aa-bases">
            {Array.from({ length: totalBases }, (_, i) => {
              const aa = aminoAcids.find((a) => a.position === i);
              if (aa) {
                const isMutation = mutationPositions.has(i) || mutationPositions.has(i + 1) || mutationPositions.has(i + 2);
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
          </div>
        </div>

        {/* 5. Chromatogram trace */}
        {chromatogramData && (
          <div className="trace-row">
            <span className="row-gutter sticky-gutter">Trace</span>
            <div className="row-content">
              <ChromatogramView
                data={chromatogramData}
                totalBases={totalBases}
                alignedQueryG={displayQuery}
                gappedToQueryIdx={gappedToQueryIdx}
              />
            </div>
          </div>
        )}

        {/* 6. Position numbers & Ticks */}
        <div className="position-row">
          <span className="row-gutter sticky-gutter" />
          <div className="row-content position-bases">
            {Array.from({ length: totalBases }, (_, i) => {
              const pos = i + 1;
              if (pos % 10 === 0) {
                return <span key={i} className="pos-mark">{pos}</span>;
              }
              return <span key={i} className="pos-spacer" />;
            })}
          </div>
        </div>
        <div className="strand-row tick-row">
          <span className="row-gutter sticky-gutter" />
          <div className="row-content">
            {displayRef.split("").map((_, i) => {
              const pos = i + 1;
              if (pos % 10 === 0) return <span key={i} className="tick-major">+</span>;
              return <span key={i} className="tick-minor">·</span>;
            })}
          </div>
        </div>

        {/* Hover Tooltip */}
        {hoverInfo.idx !== null && (
          <div 
            className="hover-tooltip"
            style={{ left: `${hoverInfo.x}px`, top: `${hoverInfo.y}px` }}
          >
            <div className="tooltip-quality">Q{hoverInfo.quality}</div>
            <div className="tooltip-base">{displayQuery[hoverInfo.idx]}</div>
          </div>
        )}
      </div>
    </div>
  );
});
