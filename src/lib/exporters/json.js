export function samplesToJson(samples, { filters, date = new Date() } = {}) {
  const payload = {
    exportedAt: date.toISOString(),
    filters: filters ?? null,
    count: samples.length,
    samples,
  };
  return JSON.stringify(payload, null, 2);
}
