import { z } from "zod";

/* ── Sample-level shapes ─────────────────────────────────────────────── */

export const MutationSchema = z
  .object({
    position: z.number().nullable().optional(),
    refBase: z.string().optional(),
    queryBase: z.string().optional(),
    type: z.string().optional(),
    effect: z.string().optional(),
  })
  .passthrough();

export const WorkbenchSampleSchema = z
  .object({
    id: z.string(),
    name: z.string().optional(),
    clone: z.string().optional(),
    status: z.string().optional(),
    sample: z.string().optional(),
    identity: z.number().nullable().optional(),
    coverage: z.number().nullable().optional(),
    cds_coverage: z.number().nullable().optional(),
    sub_count: z.number().optional(),
    ins_count: z.number().optional(),
    del_count: z.number().optional(),
    aa_changes: z.unknown().optional(),
    aa_changes_n: z.number().optional(),
    avg_qry_quality: z.number().optional(),
    avg_quality: z.number().optional(),
    orientation: z.string().optional(),
    frameshift: z.boolean().optional(),
    mutations: z.array(MutationSchema).optional(),
    error: z.string().optional(),
    reason: z.string().optional(),
  })
  .passthrough();

/* ── Analysis result (analyze_sequences / get_analysis_detail) ───────── */

export const AnalysisDetailSchema = z
  .object({
    samples: z.array(WorkbenchSampleSchema).optional(),
  })
  .passthrough();

export const AnalysisResultSchema = z
  .object({
    analysis_id: z.string().optional(),
    dataset: z.string().optional(),
    sample_count: z.number().optional(),
    totalSamples: z.number().optional(),
    detail: AnalysisDetailSchema.optional(),
    samples: z.array(WorkbenchSampleSchema).optional(),
    detail_error: z.string().optional(),
    __detailPending: z.boolean().optional(),
    __detailError: z.string().optional(),
    __summaryPending: z.boolean().optional(),
    __summaryError: z.string().optional(),
  })
  .passthrough();

/* ── Trend / lab-suggestion shapes (read by panels) ──────────────────── */

export const MutationHotspotSchema = z
  .object({
    position: z.union([z.number(), z.string()]).optional(),
    count: z.number().optional(),
  })
  .passthrough();

export const MutationTrendResultSchema = z
  .object({
    total_samples: z.number().optional(),
    total_mutations: z.number().optional(),
    mutation_hotspots: z.array(MutationHotspotSchema).optional(),
    insights: z.array(z.string()).optional(),
  })
  .passthrough();

export const LabDiagnosisSchema = z
  .object({
    clone: z.string().optional(),
    issue: z.string().optional(),
    suggestion: z.string().optional(),
    severity: z.string().optional(),
  })
  .passthrough();

export const LabSuggestionResultSchema = z
  .object({
    overall_health: z.string().optional(),
    summary: z.string().optional(),
    diagnoses: z.array(LabDiagnosisSchema).optional(),
    suggestions: z.array(z.string()).optional(),
  })
  .passthrough();

/* ── Agent event union ───────────────────────────────────────────────── */

export const AgentEventSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("lifecycle"), phase: z.string().optional(), message: z.string().optional() }),
  z.object({ type: z.literal("thinking") }),
  z.object({ type: z.literal("tool_calls_start") }),
  z.object({ type: z.literal("tool_call"), tool: z.string().optional(), chained: z.boolean().optional() }),
  z.object({
    type: z.literal("tool_result"),
    tool: z.string().optional(),
    result: AnalysisResultSchema.optional(),
    chained: z.boolean().optional(),
  }),
  z.object({
    type: z.literal("reply"),
    content: z.string().optional(),
    uiAction: z.string().optional(),
    result: AnalysisResultSchema.optional(),
  }),
  z.object({ type: z.literal("busy"), message: z.string().optional() }),
  z.object({ type: z.literal("error"), message: z.string().optional() }),
  z.object({ type: z.literal("confirm"), message: z.string().optional() }),
  z.object({
    type: z.literal("summary_pending"),
    analysis_id: z.string().optional(),
    dataset: z.string().optional(),
  }),
  z.object({
    type: z.literal("summary_chunk"),
    analysis_id: z.string().optional(),
    dataset: z.string().optional(),
    content: z.string().optional(),
  }),
  z.object({
    type: z.literal("summary_ready"),
    analysis_id: z.string().optional(),
    dataset: z.string().optional(),
    content: z.string().optional(),
    uiAction: z.string().optional(),
  }),
  z.object({
    type: z.literal("summary_failed"),
    analysis_id: z.string().optional(),
    dataset: z.string().optional(),
    message: z.string().optional(),
  }),
]);

/* ── Inferred TypeScript types ───────────────────────────────────────── */

export type Mutation = z.infer<typeof MutationSchema>;
export type WorkbenchSampleResult = z.infer<typeof WorkbenchSampleSchema>;
export type AnalysisDetail = z.infer<typeof AnalysisDetailSchema>;
export type AnalysisResult = z.infer<typeof AnalysisResultSchema>;
export type MutationHotspot = z.infer<typeof MutationHotspotSchema>;
export type MutationTrendResult = z.infer<typeof MutationTrendResultSchema>;
export type LabDiagnosis = z.infer<typeof LabDiagnosisSchema>;
export type LabSuggestionResult = z.infer<typeof LabSuggestionResultSchema>;
export type AgentEvent = z.infer<typeof AgentEventSchema>;
