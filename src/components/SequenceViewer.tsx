import React, { useMemo, useState, memo, useRef, useCallback } from "react";
import { Mutation, ChromatogramData } from "../types";
import {
  translateCodon,
  baseColor,
} from "../utils/sequence";
import { ChromatogramWorkerView } from "./Chromatogram/ChromatogramWorkerView";
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
}) => {
  const displayRef = alignedRefG || refSequence;
  const displayQuery = alignedQueryG || querySequence;
  const totalBases = displayRef?.length || 0;

  const containerRef = useRef<HTMLDivElement>(null);

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

  const gappedCdsStart = useMemo(() => refToGapped[cdsStart - 1] ?? (cdsStart - 1), [cdsStart, refToGapped]);
  const gappedCdsEnd = useMemo(() => {
    const lastIdx = refToGapped[cdsEnd - 1] ?? (cdsEnd - 1);
    return lastIdx + 1;
  }, [cdsEnd, refToGapped]);

  const aminoAcids = useMemo(() => {
    if (cdsStart <= 0 || cdsEnd <= 0 || !refSequence) return { refAAs: [], queryAAs: [] };
    
    const refAAs = [];
    const queryAAs = [];

    // We iterate through the CDS region in the reference sequence
    // and find the corresponding codons in both aligned sequences.
    for (let i = cdsStart - 1; i + 2 < cdsEnd; i += 3) {
      // 1. Reference AA
      const refCodon = refSequence.slice(i, i + 3);
      const refGappedStart = refToGapped[i];
      const refGappedEnd = refToGapped[i + 2];
      
      if (refGappedStart !== undefined && refGappedEnd !== undefined) {
        refAAs.push({
          aa: translateCodon(refCodon),
          codon: refCodon,
          gappedStart: refGappedStart,
          gappedEnd: refGappedEnd
        });

        // 2. Query AA (Sanger)
        // We take the 3 bases from the aligned query sequence at the SAME gapped positions
        // This ensures we are comparing the same "slot" in the alignment.
        const qCodon = displayQuery.slice(refGappedStart, refGappedEnd + 1);
        queryAAs.push({
          aa: translateCodon(qCodon),
          codon: qCodon,
          gappedStart: refGappedStart,
          gappedEnd: refGappedEnd
        });
      }
    }
    return { refAAs, queryAAs };
  }, [refSequence, displayQuery, cdsStart, cdsEnd, refToGapped]);
  const mutationPositions = useMemo(() => {
    const set = new Set<number>();
    for (const m of mutations) {
      const gappedIdx = refToGapped[m.position - 1];
      if (gappedIdx !== undefined) set.add(gappedIdx);
    }
    return set;
  }, [mutations, refToGapped]);

  const mismatchPositions = useMemo(() => {
    const positions: number[] = [];
    for (let i = 0; i < matches.length; i++) {
      if (!matches[i] || mutationPositions.has(i)) {
        positions.push(i);
      }
    }
    return positions;
  }, [matches, mutationPositions]);

  const [currentErrorIndex, setCurrentErrorIndex] = useState<number>(-1);

  const navigateToNextError = useCallback(() => {
    if (mismatchPositions.length === 0 || !containerRef.current) return;

    const nextIndex = (currentErrorIndex + 1) % mismatchPositions.length;
    setCurrentErrorIndex(nextIndex);

    const targetPosition = mismatchPositions[nextIndex];
    const targetScrollLeft = targetPosition * BASE_WIDTH - containerRef.current.clientWidth / 2 + BASE_WIDTH / 2;

    containerRef.current.scrollTo({
      left: Math.max(0, targetScrollLeft),
      behavior: "smooth"
    });
  }, [mismatchPositions, currentErrorIndex]);

  const featureInView = gappedCdsStart >= 0;

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!chromatogramData) return;
    
    const container = e.currentTarget as HTMLElement;
    const rect = container.getBoundingClientRect();
    const x = e.clientX - rect.left - 100; // 100 is the gutter width
    const y = e.clientY - rect.top;

    // Find the trace row element to check its boundaries
    const traceRow = container.querySelector(".trace-row");
    if (!traceRow) return;
    const traceRect = traceRow.getBoundingClientRect();
    const relativeTraceTop = traceRect.top - rect.top;
    const relativeTraceBottom = traceRect.bottom - rect.top;

    // Only show tooltip if hovering within the trace row vertically
    if (x < 0 || y < relativeTraceTop || y > relativeTraceBottom) {
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
          y: y - 20 
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
    <div className="sequence-viewer horizontal" ref={containerRef}>
      {mismatchPositions.length > 0 && (
        <button
          className="navigate-errors-btn"
          onClick={navigateToNextError}
          title={`Next error (${currentErrorIndex + 1}/${mismatchPositions.length})`}
        >
          ▶ ({currentErrorIndex + 1}/{mismatchPositions.length})
        </button>
      )}
      <div
        className="sequence-container"
        style={{ width: `${totalBases * BASE_WIDTH + 100}px` }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      >
        {/* SnapGene style alignment block (Ref / Matches / Sanger) */}
        <div className="alignment-block">
          {/* Reference line */}
          <div className="strand-row ref-line">
            <span className="row-gutter sticky-gutter">Ref</span>
            <div className="row-content">
              {/* CDS Highlight Background */}
              {featureInView && (
                <div 
                  className="cds-highlight-bg"
                  style={{
                    left: `${gappedCdsStart * BASE_WIDTH}px`,
                    width: `${(gappedCdsEnd - gappedCdsStart) * BASE_WIDTH}px`,
                  }}
                />
              )}
              {displayRef.split("").map((base, i) => {
                const isMatch = matches[i];
                const isMutation = mutationPositions.has(i);
                const highlight = !isMatch || isMutation;
                const inCds = i >= gappedCdsStart && i < gappedCdsEnd;
                return (
                  <span
                    key={i}
                    className={`base-char ${!isMatch ? "mismatch-ref" : ""} ${isMutation ? "mutation-ref" : ""} ${inCds ? "in-cds" : ""}`}
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
              {/* CDS Highlight Background */}
              {featureInView && (
                <div 
                  className="cds-highlight-bg"
                  style={{
                    left: `${gappedCdsStart * BASE_WIDTH}px`,
                    width: `${(gappedCdsEnd - gappedCdsStart) * BASE_WIDTH}px`,
                  }}
                />
              )}
              {displayQuery.split("").map((base, i) => {
                const isMatch = matches[i];
                const isMutation = mutationPositions.has(i);
                const highlight = !isMatch || isMutation;
                const inCds = i >= gappedCdsStart && i < gappedCdsEnd;
                return (
                  <span
                    key={i}
                    className={`base-char ${!isMatch ? "mismatch" : ""} ${isMutation ? "mutation" : ""} ${inCds ? "in-cds" : ""}`}
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

        {/* Amino acid translation - Reference */}
        <div className="aa-row">
          <span className="row-gutter sticky-gutter">AA Ref</span>
          <div className="row-content aa-bases">
            {/* CDS Highlight Background */}
            {featureInView && (
              <div 
                className="cds-highlight-bg"
                style={{
                  left: `${gappedCdsStart * BASE_WIDTH}px`,
                  width: `${(gappedCdsEnd - gappedCdsStart) * BASE_WIDTH}px`,
                }}
              />
            )}
            {aminoAcids.refAAs?.map((aa, i) => {
              const startX = aa.gappedStart * BASE_WIDTH;
              const endX = (aa.gappedEnd + 1) * BASE_WIDTH;
              const width = endX - startX;
              
              return (
                <span
                  key={i}
                  className="aa-char"
                  style={{ 
                    position: "absolute", 
                    left: `${startX}px`, 
                    width: `${width}px`,
                    textAlign: "center"
                  }}
                  title={`${AA_THREE[aa.aa] || aa.aa} (${aa.codon})`}
                >
                  {aa.aa}
                </span>
              );
            })}
          </div>
        </div>

        {/* Amino acid translation - Sanger */}
        <div className="aa-row">
          <span className="row-gutter sticky-gutter">AA Sanger</span>
          <div className="row-content aa-bases">
            {/* CDS Highlight Background */}
            {featureInView && (
              <div
                className="cds-highlight-bg"
                style={{
                  left: `${gappedCdsStart * BASE_WIDTH}px`,
                  width: `${(gappedCdsEnd - gappedCdsStart) * BASE_WIDTH}px`,
                }}
              />
            )}
            {aminoAcids.queryAAs?.map((aa, i) => {
              const startX = aa.gappedStart * BASE_WIDTH;
              const endX = (aa.gappedEnd + 1) * BASE_WIDTH;
              const width = endX - startX;

              const refAA = aminoAcids.refAAs[i];
              const isMutation = refAA && refAA.aa !== aa.aa;

              return (
                <span
                  key={i}
                  className={`aa-char ${isMutation ? "aa-mutation" : ""}`}
                  style={{
                    position: "absolute",
                    left: `${startX}px`,
                    width: `${width}px`,
                    textAlign: "center"
                  }}
                  title={`${AA_THREE[aa.aa] || aa.aa} (${aa.codon})`}
                >
                  {aa.aa}
                </span>
              );
            })}
          </div>
        </div>

        {/* Chromatogram trace */}
        {chromatogramData && (
          <div className="trace-row">
            <span className="row-gutter sticky-gutter">Trace</span>
            <div className="row-content">
              <ChromatogramWorkerView
                data={chromatogramData}
                totalBases={totalBases}
                alignedQueryG={displayQuery}
                gappedToQueryIdx={gappedToQueryIdx}
                baseWidth={BASE_WIDTH}
                traceHeight={TRACE_HEIGHT}
              />
            </div>
          </div>
        )}

        {/* Position numbers & Ticks */}
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
