/**
 * Shared types - NO FEATURE IMPORTS ALLOWED
 * These types are used by multiple features
 */

// ==================== Analysis Types ====================

export interface Sample {
  id: string;
  name: string;
  clone: string;
  status: "ok" | "wrong" | "uncertain" | "processing" | "error";
  reason?: string;
  identity: number;
  coverage: number;
  mutations: Mutation[];
  ab1Path?: string;
  gbPath?: string;
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
  ref_base: string;
  query_base: string;
  ref_codon?: string;
  query_codon?: string;
  ref_aa?: string;
  query_aa?: string;
  type: "substitution" | "insertion" | "deletion";
  effect?: string;
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

// ==================== Agent Types ====================

export interface AgentMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
  tools?: AgentToolCall[];
}

export interface AgentToolCall {
  name: string;
  arguments: Record<string, unknown>;
  result?: unknown;
}

export interface AgentSampleSummary {
  id: string;
  clone: string;
  status: Sample["status"];
  reason?: string;
  mutationCount: number;
  error?: string;
}

// ==================== Settings Types ====================

export type AppLanguage = "en" | "zh";

export type AppTheme = "light" | "dark" | "system";

export interface AppSettings {
  language: AppLanguage;
  theme: AppTheme;
  aiReviewEnabled: boolean;
  analysisDecisionMode: "auto" | "manual";
  llmApiKey?: string;
  llmBaseUrl?: string;
  llmModel?: string;
}

// ==================== Analysis Progress Types ====================

export type AnalysisProgressStage =
  | "preparing"
  | "scanning"
  | "aligning"
  | "aggregating"
  | "completed";

export interface AnalysisProgressState {
  stage: AnalysisProgressStage;
  progress: number;
  message?: string;
}

// ==================== Dataset Import Types ====================

export interface DatasetImportState {
  ab1Dir: string;
  genesDir?: string;
  isValid: boolean;
  errors: string[];
}
