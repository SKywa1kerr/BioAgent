function pad(n) {
  return n < 10 ? `0${n}` : `${n}`;
}

export function formatStamp(date = new Date()) {
  const y = date.getFullYear();
  const m = pad(date.getMonth() + 1);
  const d = pad(date.getDate());
  const hh = pad(date.getHours());
  const mm = pad(date.getMinutes());
  return `${y}${m}${d}-${hh}${mm}`;
}

export function sanitizeSegment(s) {
  if (!s || typeof s !== "string") return "";
  return s.replace(/[^a-zA-Z0-9_\-]+/g, "_").slice(0, 40);
}

export function buildExportFilename({ dataset, ext, date } = {}) {
  const stamp = formatStamp(date);
  const ds = sanitizeSegment(dataset) || "results";
  return `bioagent-${ds}-${stamp}.${ext}`;
}
