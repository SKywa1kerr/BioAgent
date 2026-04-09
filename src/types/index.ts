export interface Sample {
  id: string;
  name: string;
  clone: string;
  status: "ok" | "wrong" | "uncertain" | "processing" | "error";
  reason?: string;
  rule_id?: number;
  identity: number;
  coverage: number;
  mutations: Mutation[];
  sub_count?: number;
  ins_count?: number;
  del_count?: number;
  sub?: number;
  ins?: number;
  dele?: number;
  aa_changes_n?: number;
  raw_aa_changes_n?: number;
  avg_quality?: number;
  avg_qry_quality?: number;
  orientation?: string;
  cds_coverage?: number;
  aa_changes?: string | string[];
  total_cds_coverage?: number;
  read_conflict?: boolean;
  other_reads?: string[];
  ab1?: string;
  gb?: string;
  ab1Path?: string;
  gbPath?: string;
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
  llm_status?: "ok" | "wrong";
  llm_reason?: string;
  decision_source?: "rules" | "llm" | "reviewed";
  error?: string;
  reviewed?: boolean;
  review_status?: "ok" | "wrong";
  review_reason?: string;
  review_source?: string;
  auto_status?: "ok" | "wrong" | "uncertain" | "processing" | "error";
  auto_reason?: string;
  // Chromatogram data
  traces_a?: number[];
  traces_t?: number[];
  traces_g?: number[];
  traces_c?: number[];
  quality?: number[];
  base_locations?: number[];
  mixed_peaks?: number[];
}

export interface AgentSampleSummary {
  id: string;
  clone: string;
  status: Sample["status"];
  reason?: string;
  mutationCount: number;
  error?: string;
}

export type ToolCategory = "query" | "action";

export type StopReason =
  | "final_reply"
  | "max_rounds_reached"
  | "tool_failed"
  | "invalid_model_output"
  | "permission_denied"
  | "aborted";

export interface TokenUsage {
  input: number;
  output: number;
  total?: number;
}

export interface ToolCall {
  tool:
    | "query_samples"
    | "query_history"
    | "get_sample_detail"
    | "run_analysis"
    | "export_report";
  args: Record<string, unknown>;
}

export interface ToolSpec {
  name: ToolCall["tool"];
  description: string;
  parameters: Record<string, unknown>;
  category: ToolCategory;
}

export interface AgentRuntimeConfig {
  maxRounds: number;
  maxToolCallsPerTurn: number;
  maxRecentMessages: number;
  allowActionTools: boolean;
  includeUsage: boolean;
}

export interface ToolResult {
  tool: ToolCall["tool"];
  ok: boolean;
  summary: string;
  data?: unknown;
}

export interface AgentFailure {
  kind: Exclude<StopReason, "final_reply" | "max_rounds_reached" | "aborted">;
  message: string;
  toolName?: string;
}

export type ChatMessage =
  | { id: string; type: "user"; content: string; timestamp: number }
  | {
      id: string;
      type: "agent";
      content: string;
      timestamp: number;
      usage?: TokenUsage;
      stopReason?: StopReason;
    }
  | { id: string; type: "plan"; content: string; timestamp: number }
  | {
      id: string;
      type: "tool_status";
      content: string;
      timestamp: number;
      toolName: ToolCall["tool"];
      status: "running" | "done" | "failed";
    };

export type AgentTurnResponse =
  | {
      action: "reply";
      content: string;
      usage?: TokenUsage;
      stopReason?: StopReason;
    }
  | {
      action: "tool_calls";
      message: string;
      calls: ToolCall[];
      usage?: TokenUsage;
    };

export interface AgentContext {
  currentAnalysis?: {
    sourcePath?: string;
    samples: AgentSampleSummary[];
    selectedSampleId?: string | null;
  };
  recentToolResults?: ToolResult[];
  history?: ChatMessage[];
  runtime?: AgentRuntimeConfig;
}

export interface AnalysisContextUpdate {
  sourcePath?: string;
  genesDir?: string;
  plasmid: string;
  samples: Sample[];
  selectedSampleId: string | null;
}

export interface DatasetImportState {
  datasetDir: string;
  datasetName: string;
  ab1Dir: string | null;
  gbDir: string | null;
  missing: Array<"ab1" | "gb">;
  valid: boolean;
}

export type ResultWorkbenchStatus = "ok" | "wrong" | "uncertain" | "untested";

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

export type AnalysisProgressStage =
  | "idle"
  | "preparing"
  | "scanning"
  | "aligning"
  | "aggregating"
  | "completed"
  | "failed";

export interface AnalysisProgressState {
  stage: AnalysisProgressStage;
  percent: number | null;
  processedSamples: number | null;
  totalSamples: number | null;
  sampleId?: string | null;
  message?: string;
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
  llmModel: string;
  plasmid: string;
  qualityThreshold: number;
  analysisDecisionMode?: "rules" | "hybrid";
  language?: AppLanguage;
  theme?: AppTheme;
}

export type AppLanguage = "zh" | "en";
export type AppTheme = "light" | "dark";
