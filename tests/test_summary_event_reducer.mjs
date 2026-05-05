import test from "node:test";
import assert from "node:assert/strict";
import { reduceSummaryEvent } from "../src/hooks/summaryEventReducer.js";

const SAMPLE_PENDING = { type: "summary_pending", analysis_id: "a-1", dataset: "pro" };
const SAMPLE_READY = { type: "summary_ready", analysis_id: "a-1", dataset: "pro", content: "整体稳定。" };
const SAMPLE_FAILED = { type: "summary_failed", analysis_id: "a-1", dataset: "pro", message: "boom" };

test("summary_pending sets pending flag and emits assistant placeholder", () => {
  const out = reduceSummaryEvent(SAMPLE_PENDING, { lang: "zh" });
  assert.equal(out.kind, "pending");
  assert.equal(out.analysisId, "a-1");
  assert.equal(out.assistantText, "助手正在生成本次分析解读…");
});

test("summary_pending in english falls back to en strings", () => {
  const out = reduceSummaryEvent(SAMPLE_PENDING, { lang: "en" });
  assert.equal(out.assistantText, "Assistant summary is being generated…");
});

test("summary_ready clears pending and pushes content as assistant message", () => {
  const out = reduceSummaryEvent(SAMPLE_READY, { lang: "zh" });
  assert.equal(out.kind, "ready");
  assert.equal(out.analysisId, "a-1");
  assert.equal(out.assistantText, "整体稳定。");
});

test("summary_ready ignores empty content (no extra assistant push)", () => {
  const out = reduceSummaryEvent({ ...SAMPLE_READY, content: "   " }, { lang: "zh" });
  assert.equal(out.kind, "ready");
  assert.equal(out.assistantText, null);
});

test("summary_failed clears pending and reports failure with i18n", () => {
  const out = reduceSummaryEvent(SAMPLE_FAILED, { lang: "zh" });
  assert.equal(out.kind, "failed");
  assert.equal(out.analysisId, "a-1");
  assert.equal(out.assistantText, "助手解读生成失败：boom。分析结果已就绪。");
});

test("summary_failed without message uses fallback", () => {
  const out = reduceSummaryEvent({ type: "summary_failed", analysis_id: "a-1" }, { lang: "en" });
  assert.equal(out.assistantText, "Assistant summary failed. The analysis result is ready.");
});

test("non-summary events return null (reducer is a no-op)", () => {
  assert.equal(reduceSummaryEvent({ type: "thinking" }, { lang: "zh" }), null);
  assert.equal(reduceSummaryEvent({ type: "reply", content: "x" }, { lang: "zh" }), null);
});

test("missing analysis_id is preserved as undefined for caller to ignore", () => {
  const out = reduceSummaryEvent({ type: "summary_pending" }, { lang: "zh" });
  assert.equal(out.kind, "pending");
  assert.equal(out.analysisId, undefined);
});
