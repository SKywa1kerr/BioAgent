const MAX_DETAIL_SAMPLES = 200;
const MAX_REASON_CHARS = 400;

const STATUS_KEYS = ["ok", "wrong", "uncertain", "untested"];

function bucketStatus(sample) {
  if (sample.reason === "???") return "untested";
  if (sample.status === "ok" || sample.status === "wrong") return sample.status;
  return "uncertain";
}

function pickReason(sample) {
  return sample.reason ?? sample.review_reason ?? sample.llm_reason ?? sample.auto_reason ?? "";
}

function formatAa(value) {
  if (Array.isArray(value)) {
    const list = value.filter((x) => typeof x === "string" && x.trim());
    return list.length ? list.join("; ") : "-";
  }
  if (typeof value === "string" && value.trim()) return value.trim();
  return "-";
}

function truncate(s, max) {
  if (typeof s !== "string") return "";
  return s.length > max ? `${s.slice(0, max)}…` : s;
}

function formatPercentLike(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  const v = value <= 1.5 ? value * 100 : value;
  return `${v.toFixed(1)}%`;
}

export function buildDocDefinition({
  samples,
  filters,
  dataset,
  detailMode,
  stringsFn,
  date = new Date(),
}) {
  const T = stringsFn || ((k) => k);
  const counts = { ok: 0, wrong: 0, uncertain: 0, untested: 0 };
  for (const s of samples) counts[bucketStatus(s)] += 1;

  const filterSummary = filters
    ? `status=${filters.statusFilter}, search="${filters.searchQuery}", sort=${filters.sortKey}`
    : "-";

  const content = [
    { text: T("export.pdf.title"), style: "title" },
    { text: `${T("export.pdf.dataset")}: ${dataset || "-"}`, margin: [0, 4, 0, 2] },
    { text: `${T("export.pdf.exportedAt")}: ${date.toLocaleString()}`, margin: [0, 0, 0, 4] },
    { text: `${T("export.pdf.filters")}: ${filterSummary}`, style: "meta", margin: [0, 0, 0, 8] },
    {
      style: "counts",
      table: {
        widths: ["*", "*", "*", "*", "*"],
        body: [
          ["total", ...STATUS_KEYS],
          [String(samples.length), ...STATUS_KEYS.map((k) => String(counts[k]))],
        ],
      },
      layout: "lightHorizontalLines",
    },
  ];

  if (!detailMode) {
    content.push({
      text: T("export.pdf.summaryOnly"),
      style: "meta",
      margin: [0, 12, 0, 0],
    });
    return wrap(content);
  }

  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    content.push({
      text: s.id || s.name || `(sample ${i + 1})`,
      style: "sampleHeader",
      pageBreak: i === 0 ? undefined : "before",
    });
    content.push({
      table: {
        widths: ["auto", "*"],
        body: [
          ["status", s.status || bucketStatus(s)],
          ["identity", formatPercentLike(s.identity)],
          ["coverage", formatPercentLike(s.cds_coverage ?? s.coverage)],
          ["sub/ins/del", `${pickCount(s.sub_count, s.sub)} / ${pickCount(s.ins_count, s.ins)} / ${pickCount(s.del_count, s.dele)}`],
          ["aa_changes", formatAa(s.aa_changes)],
        ],
      },
      layout: "lightHorizontalLines",
      margin: [0, 4, 0, 4],
    });
    const reason = truncate(pickReason(s), MAX_REASON_CHARS);
    if (reason) content.push({ text: reason, style: "reason" });
  }

  return wrap(content);
}

function pickCount(a, b) {
  if (typeof a === "number") return a;
  if (typeof b === "number") return b;
  return 0;
}

function wrap(content) {
  return {
    content,
    defaultStyle: { font: "NotoSansSC", fontSize: 10 },
    styles: {
      title: { fontSize: 20, bold: true, margin: [0, 0, 0, 8] },
      meta: { fontSize: 9, color: "#666" },
      counts: { margin: [0, 8, 0, 8] },
      sampleHeader: { fontSize: 13, bold: true, margin: [0, 12, 0, 4] },
      reason: { fontSize: 9, italics: true, color: "#444" },
    },
    pageMargins: [40, 40, 40, 40],
  };
}

export const PDF_LIMITS = Object.freeze({
  MAX_DETAIL_SAMPLES,
  MAX_REASON_CHARS,
});
