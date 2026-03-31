export interface Sample {
  id: string;
  name: string;
  clone: string;
  status: "ok" | "wrong" | "uncertain" | "processing" | "error";
  identity: number;
  coverage: number;
  mutations: Mutation[];
  ab1Path: string;
  gbPath: string;
  // Added fields from new Python backend
  ref_sequence: string;
  query_sequence: string;
  aligned_ref_g?: string;
  aligned_query_g?: string;
  aligned_query: string;
  matches: boolean[];
  cds_start: number;
  cds_end: number;
  frameshift: boolean;
  llm_verdict?: string;
  error?: string;
  // Chromatogram data
  traces_a?: number[];
  traces_t?: number[];
  traces_g?: number[];
  traces_c?: number[];
  quality?: number[];
  base_locations?: number[];
  mixed_peaks?: number[];
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
  base_locations: number[];
  mixed_peaks: number[];
}

export interface AnalysisProgress {
  sampleId: string;
  status: "pending" | "processing" | "complete" | "error";
  progress: number;
  error?: string;
}

export interface AnalysisRecord {
  id: string;
  name: string;
  source_path: string;
  total: number;
  ok_count: number;
  wrong_count: number;
  uncertain_count: number;
  created_at: string;
}

export interface AppSettings {
  llmApiKey: string;
  llmBaseUrl: string;
  plasmid: string;
  qualityThreshold: number;
}
