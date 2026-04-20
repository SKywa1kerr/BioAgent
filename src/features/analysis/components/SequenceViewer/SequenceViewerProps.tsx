import React, { useMemo, useRef, useCallback } from "react";
import { ChromatogramData } from "../../types";
import { ChromatogramWorkerView } from "../Chromatogram/ChromatogramWorkerView";
import { buildCoordinateMap, buildHighlights, findMismatchPositions } from "../../utils/coordinates";
import "./SequenceViewer.css";

const BASE_WIDTH = 7.2;
const TRACE_HEIGHT = 140;

interface SequenceViewerProps {
  sampleId: string;
  refSequence: string;
  alignedRefG?: string;
  alignedQueryG?: string;
  alignedQuery: string;
  matches: boolean[];
  chromatogramData: ChromatogramData | null;
  cdsStart: number;
  cdsEnd: number;
  showChromatogram?: boolean;
}

/**
 * SequenceViewer with props - no store dependencies.
 * This avoids the infinite loop issues with Zustand subscriptions.
 */
export const SequenceViewer: React.FC<SequenceViewerProps> = React.memo(({
  sampleId,
  refSequence = "",
  alignedRefG = "",
  alignedQueryG = "",
  alignedQuery,
  matches = [],
  chromatogramData,
  cdsStart,
  cdsEnd,
  showChromatogram = true,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  const displayRef = alignedRefG || refSequence;
  const displayQuery = alignedQueryG || alignedQuery;
  const totalBases = displayRef?.length || 0;

  // Build coordinate map (memoized)
  const coordMap = useMemo(() => {
    return buildCoordinateMap({
      refGapped: displayRef,
      queryGapped: displayQuery,
      matches,
    });
  }, [displayRef, displayQuery, matches]);

  // Build highlights (memoized)
  const highlights = useMemo(() => {
    const cdsFeature = cdsStart > 0 && cdsEnd > 0 ? {
      id: "cds",
      type: "CDS" as const,
      start: cdsStart - 1,
      end: cdsEnd,
      name: "CDS",
    } : null;

    return buildHighlights({
      alignment: { refGapped: displayRef, queryGapped: displayQuery, matches },
      coordMap,
      mutations: [], // We'll compute mutations from matches
      features: cdsFeature ? [cdsFeature] : [],
    });
  }, [displayRef, displayQuery, matches, coordMap, cdsStart, cdsEnd]);

  // Find CDS highlight
  const cdsHighlight = highlights.find(h => h.type === "cds");

  // Find mutation highlights (all mismatches)
  const mismatchPositions = useMemo(() => {
    return findMismatchPositions({ refGapped: displayRef, queryGapped: displayQuery, matches });
  }, [displayRef, displayQuery, matches]);

  // Navigation state
  const [currentErrorIndex, setCurrentErrorIndex] = React.useState(0);

  const navigateToNext = useCallback(() => {
    if (mismatchPositions.length === 0) return;
    const nextIndex = (currentErrorIndex + 1) % mismatchPositions.length;
    setCurrentErrorIndex(nextIndex);

    const targetPosition = mismatchPositions[nextIndex];
    if (containerRef.current) {
      const targetScrollLeft = targetPosition * BASE_WIDTH - containerRef.current.clientWidth / 2 + BASE_WIDTH / 2;
      containerRef.current.scrollTo({
        left: Math.max(0, targetScrollLeft),
        behavior: "smooth"
      });
    }
  }, [currentErrorIndex, mismatchPositions]);

  // Build match string
  const matchString = useMemo(() =>
    matches.map(m => m ? "|" : " ").join(""),
    [matches]
  );

  // Position ruler
  const positionString = useMemo(() => {
    return Array.from({ length: totalBases }, (_, i) => {
      const pos = i + 1;
      if (pos % 10 === 0) return String(pos).padEnd(10, ' ');
      return ' ';
    }).join('').slice(0, totalBases);
  }, [totalBases]);

  // Check if position is a mismatch
  const isMismatch = (pos: number) => !matches[pos];

  // Chromatogram mapping
  const gappedToQueryIdx = coordMap.gappedToQuery;

  return (
    <div className="sequence-viewer horizontal" ref={containerRef}>
      {mismatchPositions.length > 0 && (
        <button
          className="navigate-errors-btn"
          onClick={navigateToNext}
          title={`Next mutation (${currentErrorIndex + 1}/${mismatchPositions.length})`}
        >
          ▶ ({currentErrorIndex + 1}/{mismatchPositions.length})
        </button>
      )}

      <div
        className="sequence-container"
        style={{ width: `${totalBases * BASE_WIDTH + 100}px` }}
      >
        {/* DNA Alignment Block */}
        <div className="alignment-block">
          {/* Reference sequence */}
          <div className="strand-row ref-line">
            <span className="row-gutter sticky-gutter">Ref</span>
            <div className="row-content sequence-text" style={{ position: 'relative' }}>
              {cdsHighlight && (
                <div
                  className="cds-highlight-bg"
                  style={{
                    left: `${cdsHighlight.start * BASE_WIDTH}px`,
                    width: `${(cdsHighlight.end - cdsHighlight.start) * BASE_WIDTH}px`,
                  }}
                />
              )}
              <pre className="sequence-pre">{displayRef}</pre>
            </div>
          </div>

          {/* Match indicators */}
          <div className="strand-row match-indicators">
            <span className="row-gutter sticky-gutter" />
            <div className="row-content">
              <pre className="match-pre">{matchString}</pre>
            </div>
          </div>

          {/* Query sequence */}
          <div className="strand-row query-row">
            <span className="row-gutter sticky-gutter query-label">Sanger</span>
            <div className="row-content sequence-text" style={{ position: 'relative' }}>
              {cdsHighlight && (
                <div
                  className="cds-highlight-bg"
                  style={{
                    left: `${cdsHighlight.start * BASE_WIDTH}px`,
                    width: `${(cdsHighlight.end - cdsHighlight.start) * BASE_WIDTH}px`,
                  }}
                />
              )}
              <pre className="sequence-pre">
                {displayQuery.split("").map((char, idx) => (
                  <span
                    key={idx}
                    className={isMismatch(idx) ? "dna-mutation" : ""}
                  >
                    {char}
                  </span>
                ))}
              </pre>
            </div>
          </div>
        </div>

        {/* Chromatogram */}
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

        {/* Position ruler */}
        <div className="position-row">
          <span className="row-gutter sticky-gutter" />
          <div className="row-content">
            <pre className="position-pre">{positionString}</pre>
          </div>
        </div>
        <div className="strand-row tick-row">
          <span className="row-gutter sticky-gutter" />
          <div className="row-content">
            <pre className="tick-pre">
              {Array.from({ length: totalBases }, (_, i) =>
                (i + 1) % 10 === 0 ? '+' : '·'
              ).join('')}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
});

SequenceViewer.displayName = "SequenceViewer";
