import React, { useMemo, useRef, useCallback } from "react";
import { ChromatogramData } from "../../../../shared/types";
import { ChromatogramWorkerView } from "../Chromatogram/ChromatogramWorkerView";
import { buildCoordinateMap, buildHighlights, findMismatchPositions } from "../../utils/coordinates";
import { groupErrorsIntoRegions } from "../../../../utils/sequence";
import "./SequenceViewerNew.css";

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
 * SequenceViewer with codon-aware layout
 *
 * Visual hierarchy:
 * - AA translation row (above CDS region)
 * - Reference sequence row
 * - Match indicators row
 * - Query sequence row
 * - AA translation row (above CDS region)
 * - Chromatogram trace
 * - Position ruler
 */
export const SequenceViewer: React.FC<SequenceViewerProps> = React.memo(({
  sampleId,
  refSequence = "",
  alignedRefG = "",
  alignedQueryG = "",
  alignedQuery = "",
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

  // Build coordinate map
  const coordMap = useMemo(() => {
    return buildCoordinateMap({
      refGapped: displayRef,
      queryGapped: displayQuery,
      matches,
    });
  }, [displayRef, displayQuery, matches]);

  // Build highlights
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
      mutations: [],
      features: cdsFeature ? [cdsFeature] : [],
    });
  }, [displayRef, displayQuery, matches, coordMap, cdsStart, cdsEnd]);

  const cdsHighlight = highlights.find(h => h.type === "cds");
  const mismatchPositions = useMemo(() => {
    return findMismatchPositions({ refGapped: displayRef, queryGapped: displayQuery, matches });
  }, [displayRef, displayQuery, matches]);

  // Error regionalization
  const errorRegions = useMemo(() => {
    return groupErrorsIntoRegions(mismatchPositions, 5);
  }, [mismatchPositions]);

  // Navigation by regions
  const [currentRegionIndex, setCurrentRegionIndex] = React.useState(0);
  const navigateToNext = useCallback(() => {
    if (errorRegions.length === 0) return;
    const nextIndex = (currentRegionIndex + 1) % errorRegions.length;
    setCurrentRegionIndex(nextIndex);
    const targetRegion = errorRegions[nextIndex];
    if (containerRef.current) {
      const targetScrollLeft = targetRegion.start * BASE_WIDTH - containerRef.current.clientWidth / 2 + BASE_WIDTH / 2;
      containerRef.current.scrollTo({ left: Math.max(0, targetScrollLeft), behavior: "smooth" });
    }
  }, [currentRegionIndex, errorRegions]);

  const isMismatch = (pos: number) => !matches[pos];
  const gappedToQueryIdx = coordMap.gappedToQuery;

  // Build position ruler
  const positionString = useMemo(() => {
    return Array.from({ length: totalBases }, (_, i) => {
      const pos = i + 1;
      if (pos % 10 === 0) return String(pos).padEnd(10, ' ');
      return ' ';
    }).join('').slice(0, totalBases);
  }, [totalBases]);

  // Build match string
  const matchString = useMemo(() =>
    matches.map(m => m ? "|" : " ").join(""),
    [matches]
  );

  // Translate sequence to amino acids
  const translateSequence = (seq: string, cdsStartPos: number): { aa: string; pos: number }[] => {
    const result: { aa: string; pos: number }[] = [];

    // Build ungapped mapping
    const ungappedToGapped: number[] = [];
    let ungappedSeq = "";
    for (let i = 0; i < seq.length; i++) {
      if (seq[i] !== "-") {
        ungappedToGapped.push(i);
        ungappedSeq += seq[i];
      }
    }

    // Find CDS start in ungapped coordinates
    let cdsUngappedStart = 0;
    for (let i = 0; i < ungappedToGapped.length; i++) {
      if (ungappedToGapped[i] >= cdsStartPos) {
        cdsUngappedStart = i;
        break;
      }
    }

    // Translate codons
    const CODON_TABLE: Record<string, string> = {
      TTT: "F", TTC: "F", TTA: "L", TTG: "L",
      CTT: "L", CTC: "L", CTA: "L", CTG: "L",
      ATT: "I", ATC: "I", ATA: "I", ATG: "M",
      GTT: "V", GTC: "V", GTA: "V", GTG: "V",
      TCT: "S", TCC: "S", TCA: "S", TCG: "S",
      CCT: "P", CCC: "P", CCA: "P", CCG: "P",
      ACT: "T", ACC: "T", ACA: "T", ACG: "T",
      GCT: "A", GCC: "A", GCA: "A", GCG: "A",
      TAT: "Y", TAC: "Y", TAA: "*", TAG: "*",
      CAT: "H", CAC: "H", CAA: "Q", CAG: "Q",
      AAT: "N", AAC: "N", AAA: "K", AAG: "K",
      GAT: "D", GAC: "D", GAA: "E", GAG: "E",
      TGT: "C", TGC: "C", TGA: "*", TGG: "W",
      CGT: "R", CGC: "R", CGA: "R", CGG: "R",
      AGT: "S", AGC: "S", AGA: "R", AGG: "R",
      GGT: "G", GGC: "G", GGA: "G", GGG: "G",
    };

    for (let i = cdsUngappedStart; i + 2 < ungappedSeq.length; i += 3) {
      const codon = ungappedSeq.slice(i, i + 3).toUpperCase();
      const aa = CODON_TABLE[codon] || "?";
      const gappedPos = ungappedToGapped[i];
      result.push({ aa, pos: gappedPos });
    }

    return result;
  };

  const refAAs = cdsHighlight ? translateSequence(displayRef, cdsHighlight.start) : [];
  const queryAAs = cdsHighlight ? translateSequence(displayQuery, cdsHighlight.start) : [];

  if (!displayRef || !displayQuery || totalBases === 0) {
    return (
      <div className="sequence-viewer" ref={containerRef}>
        <div className="sequence-empty-state">
          <p>No sequence data available</p>
        </div>
      </div>
    );
  }

  return (
    <div className="sequence-viewer" ref={containerRef}>
      {errorRegions.length > 0 && (
        <button
          className="navigate-errors-btn"
          onClick={navigateToNext}
          title={`Next error region (${currentRegionIndex + 1}/${errorRegions.length})`}
        >
          ▶ Region ({currentRegionIndex + 1}/{errorRegions.length})
        </button>
      )}

      <div className="sequence-scroll-container" style={{ width: `${totalBases * BASE_WIDTH + 100}px` }}>

        {/* AA Translation - Reference */}
        {cdsHighlight && refAAs.length > 0 && (
          <div className="sequence-row aa-row">
            <span className="row-label">Ref AA</span>
            <div className="row-content">
              <div className="aa-track" style={{ paddingLeft: `${refAAs[0]?.pos * BASE_WIDTH}px` }}>
                {refAAs.map((item, idx) => (
                  <span
                    key={idx}
                    className="aa-marker"
                    style={{
                      left: `${(item.pos - (refAAs[0]?.pos || 0)) * BASE_WIDTH}px`,
                      width: `${BASE_WIDTH * 3}px`
                    }}
                  >
                    {item.aa}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Reference sequence */}
        <div className="sequence-row ref-row">
          <span className="row-label">Ref</span>
          <div className="row-content sequence-content">
            {cdsHighlight && (
              <div
                className="cds-highlight"
                style={{
                  left: `${cdsHighlight.start * BASE_WIDTH}px`,
                  width: `${(cdsHighlight.end - cdsHighlight.start) * BASE_WIDTH}px`,
                }}
              />
            )}
            <pre className="sequence-text">
              {displayRef.split("").map((char, idx) => (
                <span
                  key={idx}
                  className={`base ${isMismatch(idx) ? "mismatch" : ""} ${cdsHighlight && idx >= cdsHighlight.start && idx < cdsHighlight.end ? "in-cds" : ""}`}
                  style={{ width: `${BASE_WIDTH}px` }}
                >
                  {char}
                </span>
              ))}
            </pre>
          </div>
        </div>

        {/* Match indicators */}
        <div className="sequence-row match-row">
          <span className="row-label"></span>
          <div className="row-content">
            <pre className="match-text">
              {matchString.split("").map((char, idx) => (
                <span key={idx} style={{ width: `${BASE_WIDTH}px` }}>{char}</span>
              ))}
            </pre>
          </div>
        </div>

        {/* Query sequence */}
        <div className="sequence-row query-row">
          <span className="row-label">Query</span>
          <div className="row-content sequence-content">
            {cdsHighlight && (
              <div
                className="cds-highlight"
                style={{
                  left: `${cdsHighlight.start * BASE_WIDTH}px`,
                  width: `${(cdsHighlight.end - cdsHighlight.start) * BASE_WIDTH}px`,
                }}
              />
            )}
            <pre className="sequence-text">
              {displayQuery.split("").map((char, idx) => (
                <span
                  key={idx}
                  className={`base ${isMismatch(idx) ? "mismatch" : ""} ${cdsHighlight && idx >= cdsHighlight.start && idx < cdsHighlight.end ? "in-cds" : ""}`}
                  style={{ width: `${BASE_WIDTH}px` }}
                >
                  {char}
                </span>
              ))}
            </pre>
          </div>
        </div>

        {/* AA Translation - Query */}
        {cdsHighlight && queryAAs.length > 0 && (
          <div className="sequence-row aa-row">
            <span className="row-label">Query AA</span>
            <div className="row-content">
              <div className="aa-track" style={{ paddingLeft: `${queryAAs[0]?.pos * BASE_WIDTH}px` }}>
                {queryAAs.map((item, idx) => (
                  <span
                    key={idx}
                    className="aa-marker"
                    style={{
                      left: `${(item.pos - (queryAAs[0]?.pos || 0)) * BASE_WIDTH}px`,
                      width: `${BASE_WIDTH * 3}px`
                    }}
                  >
                    {item.aa}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Chromatogram */}
        {chromatogramData && (
          <div
            className="sequence-row trace-row"
            style={{
              opacity: showChromatogram ? 1 : 0,
              height: showChromatogram ? "auto" : 0,
              overflow: "hidden",
            }}
          >
            <span className="row-label">Trace</span>
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
        <div className="sequence-row ruler-row">
          <span className="row-label"></span>
          <div className="row-content">
            <pre className="ruler-text">{positionString}</pre>
          </div>
        </div>
      </div>
    </div>
  );
});

SequenceViewer.displayName = "SequenceViewer";
