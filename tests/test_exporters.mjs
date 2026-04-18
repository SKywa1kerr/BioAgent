import test from "node:test";
import assert from "node:assert/strict";
import {
  buildExportFilename,
  sanitizeSegment,
  formatStamp,
} from "../src/lib/exporters/filename.js";
import { samplesToCsv } from "../src/lib/exporters/csv.js";
import { samplesToJson } from "../src/lib/exporters/json.js";

test("formatStamp yields YYYYMMDD-HHmm", () => {
  const stamp = formatStamp(new Date(2026, 3, 18, 14, 23));
  assert.equal(stamp, "20260418-1423");
});

test("sanitizeSegment strips unsafe chars", () => {
  assert.equal(sanitizeSegment("pro/max:2"), "pro_max_2");
  assert.equal(sanitizeSegment(""), "");
});

test("buildExportFilename falls back to results when dataset missing", () => {
  const name = buildExportFilename({ dataset: "", ext: "csv", date: new Date(2026, 3, 18, 14, 23) });
  assert.equal(name, "bioagent-results-20260418-1423.csv");
});

test("buildExportFilename keeps valid dataset", () => {
  const name = buildExportFilename({ dataset: "promax", ext: "json", date: new Date(2026, 3, 18, 14, 23) });
  assert.equal(name, "bioagent-promax-20260418-1423.json");
});

test("samplesToCsv emits UTF-8 BOM and header", () => {
  const out = samplesToCsv([]);
  assert.ok(out.startsWith("\uFEFF"));
  assert.ok(out.includes("id,name,clone,status,reason"));
});

test("samplesToCsv escapes commas, quotes, newlines", () => {
  const rows = [{ id: "s1", reason: 'has "quote", and,comma\nline' }];
  const out = samplesToCsv(rows);
  assert.ok(out.includes('"has ""quote"", and,comma\nline"'));
});

test("samplesToCsv joins aa_changes arrays with semicolon-space", () => {
  const rows = [{ id: "s2", aa_changes: ["A1T", "G5C"] }];
  const out = samplesToCsv(rows);
  assert.ok(out.includes("A1T; G5C"));
});

test("samplesToCsv uses fallback mutation count fields", () => {
  const rows = [{ id: "s3", sub_count: 2, ins: 1, del_count: 0 }];
  const out = samplesToCsv(rows);
  const line = out.split("\n").find((l) => l.startsWith("s3"));
  assert.ok(line.includes(",2,1,0,"), `unexpected line: ${line}`);
});

test("samplesToJson wraps samples with metadata", () => {
  const fixedDate = new Date("2026-04-18T06:23:00Z");
  const out = samplesToJson([{ id: "a" }, { id: "b" }], {
    filters: { statusFilter: "wrong", searchQuery: "", sortKey: "status" },
    date: fixedDate,
  });
  const parsed = JSON.parse(out);
  assert.equal(parsed.count, 2);
  assert.equal(parsed.filters.statusFilter, "wrong");
  assert.equal(parsed.exportedAt, "2026-04-18T06:23:00.000Z");
  assert.equal(parsed.samples.length, 2);
});

test("samplesToJson omits filters when not provided", () => {
  const out = samplesToJson([{ id: "a" }], { date: new Date("2026-04-18T06:23:00Z") });
  const parsed = JSON.parse(out);
  assert.equal(parsed.filters, null);
  assert.equal(parsed.count, 1);
});

import { buildDocDefinition, PDF_LIMITS } from "../src/lib/exporters/pdfDoc.js";

test("buildDocDefinition detail mode produces sample sections", () => {
  const doc = buildDocDefinition({
    samples: [{ id: "s1", status: "ok", identity: 99.5, reason: "looks good" }],
    filters: { statusFilter: "all", searchQuery: "", sortKey: "status" },
    dataset: "pro",
    detailMode: true,
  });
  assert.ok(Array.isArray(doc.content));
  assert.ok(doc.content.some((c) => c.text === "s1" && c.style === "sampleHeader"));
});

test("buildDocDefinition summary mode skips per-sample sections", () => {
  const samples = Array.from({ length: PDF_LIMITS.MAX_DETAIL_SAMPLES + 1 }, (_, i) => ({ id: `s${i}` }));
  const doc = buildDocDefinition({
    samples,
    filters: { statusFilter: "all", searchQuery: "", sortKey: "status" },
    detailMode: false,
  });
  assert.equal(doc.content.filter((c) => c.style === "sampleHeader").length, 0);
});

test("buildDocDefinition truncates long reason text", () => {
  const longReason = "a".repeat(PDF_LIMITS.MAX_REASON_CHARS + 100);
  const doc = buildDocDefinition({
    samples: [{ id: "s1", reason: longReason }],
    filters: null,
    detailMode: true,
  });
  const reasonNode = doc.content.find((c) => c.style === "reason");
  assert.ok(reasonNode);
  assert.ok(reasonNode.text.length <= PDF_LIMITS.MAX_REASON_CHARS + 1);
  assert.ok(reasonNode.text.endsWith("…"));
});

test("buildDocDefinition uses NotoSansSC default font", () => {
  const doc = buildDocDefinition({ samples: [], filters: null, detailMode: false });
  assert.equal(doc.defaultStyle.font, "NotoSansSC");
});
