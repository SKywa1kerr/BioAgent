import { useMemo } from "react";
import {
  useCurrentAnalysis,
  useSelectedReference,
} from "../stores/sequencingStore";
import { buildCoordinateMap, buildHighlights, findMismatchPositions } from "../utils/coordinates";
import type { Feature } from "../types/sequencing";

const BASE_WIDTH = 7.2; // px per base at 12px monospace

/**
 * Central hook for all alignment-related data.
 * This is the single source of truth for coordinates, highlights, and derived data.
 *
 * When the alignment changes, everything reactive updates automatically.
 */
export function useAlignment() {
  const analysis = useCurrentAnalysis();
  const reference = useSelectedReference();

  return useMemo(() => {
    if (!analysis) {
      return null;
    }

    // Build coordinate map inside useMemo so it's cached with the analysis
    const coordMap = buildCoordinateMap(analysis.alignment);

    const { alignment, mutations, metrics } = analysis;
    const features = reference?.features || [];

    // Build highlights (reactive to alignment changes)
    const highlights = buildHighlights({
      alignment,
      coordMap,
      mutations,
      features,
    });

    // CDS highlight for easy access
    const cdsHighlight = highlights.find((h) => h.type === "cds");

    // All mismatch positions (for navigation)
    const mismatchPositions = findMismatchPositions(alignment);

    // Mutation positions with metadata
    const mutationPositions = mutations
      .map((m) => {
        const gappedPos = coordMap.refToGapped[m.refPos];
        return {
          mutation: m,
          gappedPos,
          isProtein: !!m.proteinEffect,
        };
      })
      .filter((m) => m.gappedPos !== undefined);

    return {
      // Raw alignment data
      alignment,
      refGapped: alignment.refGapped,
      queryGapped: alignment.queryGapped,
      matches: alignment.matches,
      totalBases: alignment.refGapped.length,

      // Coordinate transformations
      coordMap,
      refPosToGapped: (refPos: number) => coordMap.refToGapped[refPos],
      gappedToRefPos: (gappedPos: number) => coordMap.gappedToRef[gappedPos],
      gappedToQueryPos: (gappedPos: number) => coordMap.gappedToQuery[gappedPos],

      // Highlights (reactive)
      highlights,
      cdsHighlight,
      mutationHighlights: highlights.filter(
        (h) => h.type === "mutation" || h.type === "protein-mutation"
      ),

      // Navigation data
      mismatchPositions,
      mutationPositions,

      // Metrics
      metrics,

      // Feature access
      features,
      getFeatureRange: (feature: Feature) => ({
        start: coordMap.refToGapped[feature.start],
        end: coordMap.refToGapped[feature.end - 1] + 1,
      }),

      // Helpers
      isInCds: (gappedPos: number) =>
        cdsHighlight
          ? gappedPos >= cdsHighlight.start && gappedPos < cdsHighlight.end
          : false,

      // Pixel calculations
      baseWidth: BASE_WIDTH,
      gappedToPixel: (gappedPos: number) => gappedPos * BASE_WIDTH,
      pixelToGapped: (pixel: number) => Math.floor(pixel / BASE_WIDTH),
    };
  }, [analysis, reference]);
}

/**
 * Hook for chromatogram data transformation.
 * Maps trace data to gapped coordinates.
 */
export function useChromatogramMapping() {
  const analysis = useCurrentAnalysis();

  return useMemo(() => {
    if (!analysis) {
      return null;
    }

    // Build coordinate map inside useMemo
    const coordMap = buildCoordinateMap(analysis.alignment);

    // Build array mapping gapped position -> query position (for trace indexing)
    const gappedToQueryIdx = coordMap.gappedToQuery;

    return {
      gappedToQueryIdx,
      // Get the query index for a gapped position (null if gap)
      getQueryIndex: (gappedPos: number) => gappedToQueryIdx[gappedPos],
      // Check if a gapped position has trace data
      hasTraceData: (gappedPos: number) => gappedToQueryIdx[gappedPos] !== null,
    };
  }, [analysis]);
}
