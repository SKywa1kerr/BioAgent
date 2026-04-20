import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";
import {
  SequencingRun,
  ReferenceSequence,
  SequencingAnalysis,
  Alignment,
  Mutation,
} from "../types/sequencing";
import { buildCoordinateMap } from "../utils/coordinates";
import type { CoordinateMap } from "../types/sequencing";

// Legacy Sample type for migration
import { Sample } from "../types";

interface SequencingState {
  // Entities (normalized)
  runs: Map<string, SequencingRun>;
  references: Map<string, ReferenceSequence>;
  analyses: Map<string, SequencingAnalysis>; // key: `${runId}:${refId}`

  // Selection
  selectedRunId: string | null;
  selectedRefId: string | null;

  // UI state
  showChromatogram: boolean;
  isAnalyzing: boolean;

  // Getters (computed)
  getSelectedRun: () => SequencingRun | undefined;
  getSelectedReference: () => ReferenceSequence | undefined;
  getCurrentAnalysis: () => SequencingAnalysis | undefined;
  getCoordinateMap: () => CoordinateMap | undefined;

  // Actions
  importFromLegacy: (samples: Sample[]) => void;
  selectRun: (runId: string | null) => void;
  setReference: (refId: string) => void;
  setShowChromatogram: (show: boolean) => void;
  updateAnalysis: (runId: string, analysis: SequencingAnalysis) => void;
}

export const useSequencingStore = create<SequencingState>()(
  subscribeWithSelector((set, get) => ({
    runs: new Map(),
    references: new Map(),
    analyses: new Map(),
    selectedRunId: null,
    selectedRefId: null,
    showChromatogram: true,
    isAnalyzing: false,

    getSelectedRun: () => {
      const { runs, selectedRunId } = get();
      return selectedRunId ? runs.get(selectedRunId) : undefined;
    },

    getSelectedReference: () => {
      const { references, selectedRefId } = get();
      return selectedRefId ? references.get(selectedRefId) : undefined;
    },

    getCurrentAnalysis: () => {
      const { analyses, selectedRunId, selectedRefId } = get();
      if (!selectedRunId || !selectedRefId) return undefined;
      return analyses.get(`${selectedRunId}:${selectedRefId}`);
    },

    getCoordinateMap: () => {
      const analysis = get().getCurrentAnalysis();
      if (!analysis) return undefined;
      return buildCoordinateMap(analysis.alignment);
    },

    importFromLegacy: (samples: Sample[]) => {
      const runs = new Map<string, SequencingRun>();
      const references = new Map<string, ReferenceSequence>();
      const analyses = new Map<string, SequencingAnalysis>();

      for (const sample of samples) {
        // Create run
        const run: SequencingRun = {
          id: sample.id,
          name: sample.name,
          ab1Path: sample.ab1Path,
          clone: sample.clone,
          raw: {
            traces: {
              A: sample.tracesA || [],
              T: sample.tracesT || [],
              G: sample.tracesG || [],
              C: sample.tracesC || [],
            },
            baseCalls: sample.querySequence,
            quality: sample.quality || [],
            baseLocations: sample.baseLocations || [],
            mixedPeaks: sample.mixedPeaks || [],
          },
          uiState: {
            isSelected: false,
            isProcessing: false,
            error: sample.error,
          },
        };
        runs.set(run.id, run);

        // Create reference (if not exists)
        if (!references.has(sample.id)) {
          const ref: ReferenceSequence = {
            id: sample.id,
            name: sample.name,
            sequence: sample.refSequence,
            gbPath: sample.gbPath,
            features: [],
          };

          // Add CDS feature if defined
          if (sample.cdsStart > 0 && sample.cdsEnd > 0) {
            ref.features.push({
              id: `${sample.id}-cds`,
              type: "CDS",
              start: sample.cdsStart - 1,
              end: sample.cdsEnd,
              name: "CDS",
            });
          }

          references.set(ref.id, ref);
        }

        // Create analysis if alignment data exists
        if (sample.alignedQuery) {
          const alignment: Alignment = {
            refGapped: sample.alignedRefG || sample.refSequence,
            queryGapped: sample.alignedQueryG || sample.alignedQuery,
            matches: sample.matches,
          };

          const mutations: Mutation[] = sample.mutations.map((m, idx) => ({
            id: `${sample.id}-mut-${idx}`,
            refPos: m.position - 1,
            refBase: m.refBase,
            queryBase: m.queryBase,
            type: m.type,
            proteinEffect:
              m.refCodon && m.queryCodon
                ? {
                    featureId: `${sample.id}-cds`,
                    refCodon: m.refCodon,
                    queryCodon: m.queryCodon,
                    refAA: m.refAA || "?",
                    queryAA: m.queryAA || "?",
                  }
                : undefined,
          }));

          const analysis: SequencingAnalysis = {
            runId: sample.id,
            refId: sample.id,
            timestamp: Date.now(),
            alignment,
            mutations,
            metrics: {
              identity: sample.identity,
              coverage: sample.coverage,
              frameshift: sample.frameshift,
            },
            llmVerdict: sample.llmVerdict,
          };

          analyses.set(`${sample.id}:${sample.id}`, analysis);
          run.analysis = analysis;
        }
      }

      set({
        runs,
        references,
        analyses,
        selectedRunId: samples[0]?.id || null,
        selectedRefId: samples[0]?.id || null,
      });
    },

    selectRun: (runId: string | null) => {
      set({ selectedRunId: runId });
    },

    setReference: (refId: string) => {
      set({ selectedRefId: refId });
    },

    setShowChromatogram: (show: boolean) => {
      set({ showChromatogram: show });
    },

    updateAnalysis: (runId: string, analysis: SequencingAnalysis) => {
      const { analyses, runs } = get();
      const key = `${runId}:${analysis.refId}`;

      analyses.set(key, analysis);

      const run = runs.get(runId);
      if (run) {
        run.analysis = analysis;
      }

      set({ analyses: new Map(analyses) });
    },
  }))
);

// Selector hooks for common access patterns
export const useSelectedRun = () =>
  useSequencingStore((s) => s.getSelectedRun());
export const useSelectedReference = () =>
  useSequencingStore((s) => s.getSelectedReference());
export const useCurrentAnalysis = () =>
  useSequencingStore((s) => s.getCurrentAnalysis());
export const useCoordinateMap = () =>
  useSequencingStore((s) => s.getCoordinateMap());
export const useAllRuns = () =>
  useSequencingStore((s) => Array.from(s.runs.values()));
