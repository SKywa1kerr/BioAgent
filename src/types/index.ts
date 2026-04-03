export interface Sample {
  id: string;
  name: string;
  clone: string;
  status: "ok" | "wrong" | "processing" | "warning" | "error";
  identity: number;
  coverage: number;
  mutations: Mutation[];
  ab1Path: string;
  gbPath: string;
  // Added fields from new Python backend
  refSequence: string;
  querySequence: string;
  alignedRefG?: string;
  alignedQueryG?: string;
  alignedQuery: string;
  matches: boolean[];
  cdsStart: number;
  cdsEnd: number;
  frameshift: boolean;
  llmVerdict?: string;
  error?: string;
  // Chromatogram data
  tracesA?: number[];
  tracesT?: number[];
  tracesG?: number[];
  tracesC?: number[];
  quality?: number[];
  baseLocations?: number[];
  mixedPeaks?: number[];
}

export interface Mutation {
  position: number;
  refBase: string;
  queryBase: string;
  refCodon?: string;
  queryCodon?: string;
  refAA?: string;
  queryAA?: string;
  type: "substitution" | "insertion" | "deletion";
  effect?: string;
}

export interface AlignmentResult {
  sampleId: string;
  refSequence: string;
  querySequence: string;
  alignedQuery: string;
  matches: boolean[];
  mutations: Mutation[];
  cdsStart: number;
  cdsEnd: number;
  frameshift: boolean;
}

export interface ChromatogramData {
  traces: {
    A: number[];
    T: number[];
    G: number[];
    C: number[];
  };
  quality: number[];
  baseCalls: string;
  baseLocations: number[];
  mixedPeaks: number[];
}

export interface AnalysisProgress {
  sampleId: string;
  status: "pending" | "processing" | "complete" | "error";
  progress: number;
  error?: string;
}
