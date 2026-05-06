import {
  AnalysisResultSchema,
  AgentEventSchema,
  type AgentEvent,
  type AnalysisResult,
  type Mutation,
} from "./agentResult";

/* Compile-only assertions. No runtime exports.
 * Run via `npm run typecheck` (tsc --noEmit). */

/* ── AnalysisResult type ─────────────────────────────────────────────── */

// Positive: minimal shape parses (everything optional)
const _ar_minimal: AnalysisResult = {};
void _ar_minimal;

// Positive: full shape parses
const _ar_full: AnalysisResult = {
  analysis_id: "abc",
  dataset: "base",
  sample_count: 5,
  detail: { samples: [] },
  samples: [{ id: "s1" }],
  detail_error: undefined,
  __detailPending: false,
  __detailError: undefined,
  __summaryPending: true,
  __summaryError: undefined,
};
void _ar_full;

// Negative: cannot assign mismatched shape (analysis_id must be string)
const _ar_bad: AnalysisResult = {
  // @ts-expect-error analysis_id must be string
  analysis_id: 123,
};
void _ar_bad;

// Negative: sample_count must be number, not string
const _ar_bad2: AnalysisResult = {
  // @ts-expect-error sample_count must be number
  sample_count: "5",
};
void _ar_bad2;

/* ── AgentEvent narrowing ────────────────────────────────────────────── */

function _narrow_reply(e: AgentEvent): string | undefined {
  if (e.type === "reply") {
    const _content: string | undefined = e.content;
    return _content;
  }
  return undefined;
}
void _narrow_reply;

function _narrow_tool_result(e: AgentEvent): AnalysisResult | undefined {
  if (e.type === "tool_result") {
    const _result: AnalysisResult | undefined = e.result;
    return _result;
  }
  return undefined;
}
void _narrow_tool_result;

// Negative: cannot read .content on lifecycle event
function _narrow_bad(e: AgentEvent): void {
  if (e.type === "lifecycle") {
    // @ts-expect-error lifecycle has no `content`
    const _bad: string | undefined = e.content;
    void _bad;
  }
}
void _narrow_bad;

// Negative: cannot construct AgentEvent with unknown type literal
const _ev_bad: AgentEvent = {
  // @ts-expect-error "bogus" is not a known event discriminator
  type: "bogus",
};
void _ev_bad;

/* ── Mutation type ───────────────────────────────────────────────────── */

const _mut: Mutation = { position: 42, refBase: "A", queryBase: "G" };
void _mut;

const _mut_null_pos: Mutation = { position: null };
void _mut_null_pos;

/* ── Schema runtime types match inferred TS types ────────────────────── */

const _schema_check_ar: AnalysisResult = AnalysisResultSchema.parse({});
void _schema_check_ar;

const _schema_check_ev: AgentEvent = AgentEventSchema.parse({ type: "thinking" });
void _schema_check_ev;
