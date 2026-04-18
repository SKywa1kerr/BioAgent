const BOM = "\uFEFF";

function pickCount(a, b) {
  if (typeof a === "number") return a;
  if (typeof b === "number") return b;
  return 0;
}

function formatAaChanges(value) {
  if (Array.isArray(value)) {
    return value.filter((x) => typeof x === "string" && x.trim()).join("; ");
  }
  if (typeof value === "string" && value.trim()) return value.trim();
  return "";
}

const COLUMNS = [
  ["id", (s) => s.id ?? ""],
  ["name", (s) => s.name ?? ""],
  ["clone", (s) => s.clone ?? ""],
  ["status", (s) => s.status ?? ""],
  ["reason", (s) => s.reason ?? s.review_reason ?? s.llm_reason ?? s.auto_reason ?? ""],
  ["identity", (s) => (typeof s.identity === "number" ? s.identity : "")],
  [
    "cds_coverage",
    (s) => {
      const v = s.cds_coverage ?? s.coverage;
      return typeof v === "number" ? v : "";
    },
  ],
  ["sub", (s) => pickCount(s.sub_count, s.sub)],
  ["ins", (s) => pickCount(s.ins_count, s.ins)],
  ["del", (s) => pickCount(s.del_count, s.dele)],
  ["aa_changes", (s) => formatAaChanges(s.aa_changes)],
  [
    "avg_quality",
    (s) => {
      const v = s.avg_qry_quality ?? s.avg_quality;
      return typeof v === "number" ? v : "";
    },
  ],
];

function escapeCell(value) {
  const s = value == null ? "" : String(value);
  if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

export function samplesToCsv(samples) {
  const header = COLUMNS.map(([name]) => name).join(",");
  const rows = samples.map((s) => COLUMNS.map(([, get]) => escapeCell(get(s))).join(","));
  return BOM + [header, ...rows].join("\n");
}
