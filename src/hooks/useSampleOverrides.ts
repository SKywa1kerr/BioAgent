import { useCallback, useState } from "react";
import {
  getOverrideKey,
  loadSampleOverrides,
  saveSampleOverrides,
  type SampleOverride,
  type SampleOverrideMap,
  type SampleStatusOverride,
} from "../lib/ui/sampleOverrides.js";

function getSafeStorage(): Storage | null {
  try {
    return typeof window !== "undefined" ? window.localStorage : null;
  } catch {
    return null;
  }
}

export interface SampleOverridesApi {
  overrides: SampleOverrideMap;
  getOverride(analysisId: string, sampleId: string): SampleOverride | undefined;
  setOverride(
    analysisId: string,
    sampleId: string,
    status: SampleStatusOverride,
    reason: string,
  ): void;
  clearOverride(analysisId: string, sampleId: string): void;
}

export function useSampleOverrides(): SampleOverridesApi {
  const [overrides, setOverrides] = useState<SampleOverrideMap>(() => {
    const store = getSafeStorage();
    return store ? loadSampleOverrides(store) : {};
  });

  const getOverride = useCallback(
    (analysisId: string, sampleId: string): SampleOverride | undefined => {
      if (!analysisId || !sampleId) return undefined;
      return overrides[getOverrideKey(analysisId, sampleId)];
    },
    [overrides],
  );

  const setOverride = useCallback(
    (
      analysisId: string,
      sampleId: string,
      status: SampleStatusOverride,
      reason: string,
    ) => {
      if (!analysisId || !sampleId) return;
      const key = getOverrideKey(analysisId, sampleId);
      const next: SampleOverride = {
        status,
        reason: reason || "",
        updatedAt: new Date().toISOString(),
      };
      setOverrides((prev) => {
        const updated = { ...prev, [key]: next };
        const store = getSafeStorage();
        if (store) saveSampleOverrides(updated, store);
        return updated;
      });
    },
    [],
  );

  const clearOverride = useCallback(
    (analysisId: string, sampleId: string) => {
      if (!analysisId || !sampleId) return;
      const key = getOverrideKey(analysisId, sampleId);
      setOverrides((prev) => {
        if (!(key in prev)) return prev;
        const { [key]: _gone, ...rest } = prev;
        const store = getSafeStorage();
        if (store) saveSampleOverrides(rest, store);
        return rest;
      });
    },
    [],
  );

  return { overrides, getOverride, setOverride, clearOverride };
}
