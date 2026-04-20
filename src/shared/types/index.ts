/**
 * Shared types - NO FEATURE IMPORTS ALLOWED
 * These types are used by multiple features
 * ALL FIELDS USE CAMELCASE (JavaScript convention)
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
