export type SampleStatusOverride = "ok" | "wrong" | "uncertain";

export interface SampleOverride {
  status: SampleStatusOverride;
  reason: string;
  updatedAt: string;
}

export type SampleOverrideMap = Record<string, SampleOverride>;

interface Storage {
  getItem(k: string): string | null;
  setItem(k: string, v: string): void;
}

export function getOverrideKey(analysisId: string, sampleId: string): string;
export function loadSampleOverrides(store?: Storage): SampleOverrideMap;
export function saveSampleOverrides(map: SampleOverrideMap, store?: Storage): void;
