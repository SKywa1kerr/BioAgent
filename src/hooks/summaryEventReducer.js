const STRINGS = {
  zh: {
    pending: "助手正在生成本次分析解读…",
    failedWithReason: "助手解读生成失败：{message}。分析结果已就绪。",
    failedFallback: "助手解读生成失败。分析结果已就绪。",
  },
  en: {
    pending: "Assistant summary is being generated…",
    failedWithReason: "Assistant summary failed: {message}. The analysis result is ready.",
    failedFallback: "Assistant summary failed. The analysis result is ready.",
  },
};

function pickLang(input) {
  return input === "en" ? "en" : "zh";
}

function format(template, params) {
  return template.replace(/\{(\w+)\}/g, (_, key) => (params && params[key] != null ? String(params[key]) : ""));
}

export function reduceSummaryEvent(event, opts) {
  if (!event || typeof event !== "object") return null;
  const lang = pickLang(opts && opts.lang);
  const strings = STRINGS[lang];
  const analysisId = event.analysis_id;

  if (event.type === "summary_pending") {
    return { kind: "pending", analysisId, assistantText: strings.pending };
  }

  if (event.type === "summary_ready") {
    const raw = typeof event.content === "string" ? event.content.trim() : "";
    return { kind: "ready", analysisId, assistantText: raw.length > 0 ? raw : null };
  }

  if (event.type === "summary_failed") {
    const reason = typeof event.message === "string" ? event.message.trim() : "";
    const text = reason.length > 0 ? format(strings.failedWithReason, { message: reason }) : strings.failedFallback;
    return { kind: "failed", analysisId, assistantText: text };
  }

  return null;
}
