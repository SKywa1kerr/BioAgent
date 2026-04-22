import { Alignment, CoordinateMap, Feature, HighlightRegion, Mutation } from "../types/sequencing";

/**
 * Build coordinate mappings from an alignment.
 * This is the single source of truth for all coordinate transformations.
 */
export function buildCoordinateMap(alignment: Alignment): CoordinateMap {
  const { refGapped, queryGapped } = alignment;
  const length = refGapped.length;

  const refToGapped: number[] = [];
  const gappedToRef: (number | null)[] = new Array(length).fill(null);
  const gappedToQuery: (number | null)[] = new Array(length).fill(null);
  const queryToGapped: number[] = [];

  let refPos = 0;
  let queryPos = 0;

  for (let gappedPos = 0; gappedPos < length; gappedPos++) {
    const refBase = refGapped[gappedPos];
    const queryBase = queryGapped[gappedPos];

    if (refBase !== "-") {
      refToGapped[refPos] = gappedPos;
      gappedToRef[gappedPos] = refPos;
      refPos++;
    }

    if (queryBase !== "-") {
      queryToGapped[queryPos] = gappedPos;
      gappedToQuery[gappedPos] = queryPos;
      queryPos++;
    }
  }

  return {
    refToGapped,
    gappedToRef,
    gappedToQuery,
    queryToGapped,
  };
}

/**
 * Get the gapped range for a feature (e.g., CDS).
 * Returns null if feature is outside alignment bounds.
 */
export function getFeatureGappedRange(
  feature: Feature,
  coordMap: CoordinateMap
): { start: number; end: number } | null {
  const start = coordMap.refToGapped[feature.start];
  const end = coordMap.refToGapped[feature.end - 1];

  if (start === undefined || end === undefined) {
    return null;
  }

  return { start, end: end + 1 };
}

/**
 * Get the gapped range for a mutation.
 */
export function getMutationGappedRange(
  mutation: Mutation,
  coordMap: CoordinateMap
): { start: number; end: number } | null {
  const start = coordMap.refToGapped[mutation.refPos];

  if (start === undefined) {
    return null;
  }

  // For insertions, highlight the position after
  // For deletions/substitutions, highlight the base(s)
  const length = mutation.type === "insertion" ? 0 : mutation.refBase.length;

  return { start, end: start + Math.max(1, length) };
}

/**
 * Build all highlight regions from analysis data.
 * This is reactive: when alignment changes, re-run this to get updated highlights.
 */
export function buildHighlights(params: {
  alignment: Alignment;
  coordMap: CoordinateMap;
  mutations: Mutation[];
  features: Feature[];
}): HighlightRegion[] {
  const { coordMap, mutations, features } = params;
  const highlights: HighlightRegion[] = [];

  // Add CDS highlights
  for (const feature of features) {
    if (feature.type === "CDS") {
      const range = getFeatureGappedRange(feature, coordMap);
      if (range) {
        highlights.push({
          ...range,
          type: "cds",
          data: feature,
        });
      }
    }
  }

  // Add mutation highlights
  for (const mutation of mutations) {
    const range = getMutationGappedRange(mutation, coordMap);
    if (range) {
      highlights.push({
        ...range,
        type: mutation.proteinEffect ? "protein-mutation" : "mutation",
        data: mutation,
      });
    }
  }

  return highlights;
}

/**
 * Find all mismatch positions in gapped coordinates.
 */
export function findMismatchPositions(alignment: Alignment): number[] {
  const positions: number[] = [];
  for (let i = 0; i < alignment.matches.length; i++) {
    if (!alignment.matches[i]) {
      positions.push(i);
    }
  }
  return positions;
}

/**
 * Check if a gapped position is within a CDS region.
 */
export function isInCds(
  gappedPos: number,
  features: Feature[],
  coordMap: CoordinateMap
): boolean {
  for (const feature of features) {
    if (feature.type === "CDS") {
      const range = getFeatureGappedRange(feature, coordMap);
      if (range && gappedPos >= range.start && gappedPos < range.end) {
        return true;
      }
    }
  }
  return false;
}
