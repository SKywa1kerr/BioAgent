import React, { useMemo, useState, memo, useRef, useCallback } from "react";
import { Mutation, ChromatogramData } from "../types";
import {
  translateCodon,
} from "../utils/sequence";
import { ChromatogramWorkerView } from "./Chromatogram/ChromatogramWorkerView";
import "./SequenceViewer.css";

interface SequenceViewerProps {
  sampleId: string;
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
  showChromatogram?: boolean;
}

// Monaco/Consolas at 12px has ~7.2px char width (0.6em)
// This must match the actual rendered monospace font metrics
const BASE_WIDTH = 7.2;
const TRACE_HEIGHT = 140;

export const SequenceViewer: React.FC<SequenceViewerProps> = memo(({
  sampleId,
  refSequence = "",
  querySequence = "",
  alignedRefG = "",
  alignedQueryG = "",
  matches = [],
  mutations = [],
  chromatogramData,
  cdsStart,
  cdsEnd,
  showChromatogram = true,
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

  // Generate amino acid strings aligned with DNA sequences
  const { refAAString, queryAAString, aaMutationIndices } = useMemo(() => {
    if (cdsStart <= 0 || cdsEnd <= 0 || !refSequence || !displayRef) {
      return { refAAString: "", queryAAString: "", aaMutationIndices: new Set<number>() };
    }

    // Initialize arrays with spaces
    const refAAs: string[] = new Array(displayRef.length).fill(" ");
    const queryAAs: string[] = new Array(displayRef.length).fill(" ");
    const mutations = new Set<number>();

    // Iterate through the CDS region in the reference sequence
    for (let i = cdsStart - 1; i + 2 < cdsEnd; i += 3) {
      const refCodon = refSequence.slice(i, i + 3);
      const refGappedStart = refToGapped[i];
      const refGappedEnd = refToGapped[i + 2];

      if (refGappedStart !== undefined && refGappedEnd !== undefined) {
        const refAA = translateCodon(refCodon);
        const qCodon = displayQuery.slice(refGappedStart, refGappedEnd + 1);
        const queryAA = translateCodon(qCodon);

        // Center the AA in the 3-base codon (position 1 of the 3)
        refAAs[refGappedStart + 1] = refAA;
        queryAAs[refGappedStart + 1] = queryAA;

        if (refAA !== queryAA) {
          mutations.add(refGappedStart + 1);
        }
      }
    }

    return {
      refAAString: refAAs.join(""),
      queryAAString: queryAAs.join(""),
      aaMutationIndices: mutations
    };
  }, [refSequence, displayRef, displayQuery, cdsStart, cdsEnd, refToGapped]);
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
          {/* Reference line - using pre-formatted text for performance */}
          <div className="strand-row ref-line">
            <span className="row-gutter sticky-gutter">Ref</span>
            <div className="row-content sequence-text" style={{ position: 'relative' }}>
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
              <pre className="sequence-pre">{displayRef}</pre>
            </div>
          </div>

          {/* Match indicators - using pre-formatted text for performance */}
          <div className="strand-row match-indicators">
            <span className="row-gutter sticky-gutter" />
            <div className="row-content">
              <pre className="match-pre">{matches.map(m => m ? "|" : " ").join("")}</pre>
            </div>
          </div>

          {/* Query alignment (Sanger result) - using pre-formatted text for performance */}
          <div className="strand-row query-row">
            <span className="row-gutter sticky-gutter query-label">Sanger</span>
            <div className="row-content sequence-text" style={{ position: 'relative' }}>
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
              <pre className="sequence-pre">{displayQuery}</pre>
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
            <pre className="aa-pre">{refAAString}</pre>
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
            <pre className="aa-pre">
              {queryAAString.split("").map((char, idx) => (
                <span key={idx} className={aaMutationIndices.has(idx) ? "aa-mutation" : ""}>
                  {char}
                </span>
              ))}
            </pre>
          </div>
        </div>

        {/* Chromatogram trace - always mounted, visibility controlled by CSS */}
        {chromatogramData && (
          <div
            className="trace-row"
            style={{
              opacity: showChromatogram ? 1 : 0,
              height: showChromatogram ? "auto" : 0,
              overflow: "hidden",
              transition: "opacity 0.2s"
            }}
          >
            <span className="row-gutter sticky-gutter">Trace</span>
            <div className="row-content">
              <ChromatogramWorkerView
                data={chromatogramData}
                sampleId={sampleId}
                totalBases={totalBases}
                alignedQueryG={displayQuery}
                gappedToQueryIdx={gappedToQueryIdx}
                baseWidth={BASE_WIDTH}
                traceHeight={TRACE_HEIGHT}
                visible={showChromatogram}
              />
            </div>
          </div>
        )}

        {/* Position numbers & Ticks - using pre-formatted text for performance */}
        <div className="position-row">
          <span className="row-gutter sticky-gutter" />
          <div className="row-content">
            <pre className="position-pre">
              {Array.from({ length: totalBases }, (_, i) => {
                const pos = i + 1;
                if (pos % 10 === 0) return String(pos).padEnd(10, ' ');
                return ' ';
              }).join('').slice(0, totalBases)}
            </pre>
          </div>
        </div>
        <div className="strand-row tick-row">
          <span className="row-gutter sticky-gutter" />
          <div className="row-content">
            <pre className="tick-pre">
              {Array.from({ length: totalBases }, (_, i) => {
                const pos = i + 1;
                if (pos % 10 === 0) return '+';
                return '·';
              }).join('')}
            </pre>
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
