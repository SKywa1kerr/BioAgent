/** Normalized sequencing types - replaces the monolithic Sample type */

// Trace data from AB1 file
export interface TraceData {
  A: number[];
  T: number[];
  G: number[];
  C: number[];
}

// A single AB1 sequencing run
export interface SequencingRun {
  id: string;
  name: string;
  ab1Path: string;
  clone?: string;

  // Raw data (immutable once loaded)
  raw: {
    traces: TraceData;
    baseCalls: string;
    quality: number[];
    baseLocations: number[];
    mixedPeaks: number[];
  };

  // Analysis results (computed)
  analysis?: SequencingAnalysis;

  // UI state (ephemeral)
  uiState: {
    isSelected: boolean;
    isProcessing: boolean;
    error?: string;
  };
}

// Reference sequence (gene/plasmid)
export interface ReferenceSequence {
  id: string;
  name: string;
  sequence: string; // ungapped
  features: Feature[];
  gbPath?: string;
}

// Feature on reference (CDS, etc.)
export interface Feature {
  id: string;
  type: "CDS" | "promoter" | "terminator" | "other";
  start: number; // 0-based, ungapped
  end: number; // exclusive
  name: string;
}

// Alignment result - single source of truth for coordinates
export interface Alignment {
  refGapped: string;
  queryGapped: string;
  matches: boolean[];
}

// Mutation - always relative to reference, ungapped coords
export interface Mutation {
  id: string;
  refPos: number; // 0-based, ungapped reference position
  refBase: string;
  queryBase: string;
  type: "substitution" | "insertion" | "deletion";

  // Protein effect (if in CDS)
  proteinEffect?: {
    featureId: string;
    refCodon: string;
    queryCodon: string;
    refAA: string;
    queryAA: string;
  };
}

// Analysis links a Run to a Reference
export interface SequencingAnalysis {
  runId: string;
  refId: string;
  timestamp: number;

  alignment: Alignment;
  mutations: Mutation[];

  metrics: {
    identity: number;
    coverage: number;
    frameshift: boolean;
  };

  // Optional LLM verdict
  llmVerdict?: string;
}

// Coordinate mapping cache (computed from Alignment)
export interface CoordinateMap {
  refToGapped: number[]; // refPos -> gappedPos
  gappedToRef: (number | null)[]; // gappedPos -> refPos (null for gaps)
  gappedToQuery: (number | null)[]; // gappedPos -> queryPos (null for gaps)
  queryToGapped: number[]; // queryPos -> gappedPos
}

// Highlight region for UI
export interface HighlightRegion {
  start: number; // gapped position
  end: number; // exclusive
  type: "cds" | "mutation" | "protein-mutation" | "selection";
  data?: unknown; // associated data (mutation, feature, etc.)
}
