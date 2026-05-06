import { useCallback, useEffect, useRef, useState } from "react";

export interface HistoryItem {
  analysis_id: string;
  dataset: string;
  sample_count: number;
  used_llm?: boolean;
  created_at?: string;
  output_dir?: string;
}

export interface UseAnalysisHistoryResult {
  items: HistoryItem[];
  total: number;
  isLoading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

interface UseAnalysisHistoryOptions {
  enabled?: boolean;
  limit?: number;
}

export function useAnalysisHistory(options: UseAnalysisHistoryOptions = {}): UseAnalysisHistoryResult {
  const { enabled = true, limit = 20 } = options;

  const [items, setItems] = useState<HistoryItem[]>([]);
  const [total, setTotal] = useState<number>(0);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const mountedRef = useRef<boolean>(true);
  const inFlightRef = useRef<boolean>(false);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const refresh = useCallback(async (): Promise<void> => {
    if (inFlightRef.current) return;
    if (typeof window === "undefined" || !window.electronAPI?.invoke) return;
    inFlightRef.current = true;
    setIsLoading(true);
    try {
      const resp = await window.electronAPI.invoke("agent-query-history", { limit });
      if (!mountedRef.current) return;
      if (resp?.ok && resp?.result) {
        const nextItems = Array.isArray(resp.result.items) ? (resp.result.items as HistoryItem[]) : [];
        const nextTotal = typeof resp.result.total === "number" ? resp.result.total : nextItems.length;
        setItems(nextItems);
        setTotal(nextTotal);
        setError(null);
      } else {
        setError(resp?.error || "Failed to load history");
      }
    } catch (err) {
      if (!mountedRef.current) return;
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      inFlightRef.current = false;
      if (mountedRef.current) setIsLoading(false);
    }
  }, [limit]);

  useEffect(() => {
    if (!enabled) return;
    void refresh();
  }, [enabled, refresh]);

  return { items, total, isLoading, error, refresh };
}
