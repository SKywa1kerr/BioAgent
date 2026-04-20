// Analysis Feature - Core sequencing analysis
// Exports the public API for the analysis feature

export { SequenceViewer } from "./components/SequenceViewer";
export { ChromatogramWorkerView } from "./components/Chromatogram/ChromatogramWorkerView";

export { useSequencingStore } from "./stores/sequencingStore";
export type {
  SequencingRun,
  ReferenceSequence,
  SequencingAnalysis,
  Alignment,
  Mutation,
  Feature,
  CoordinateMap,
  HighlightRegion,
} from "./types/sequencing";

export { useAlignment, useChromatogramMapping } from "./hooks/useAlignment";
export { useHighlights, useHoverHighlight } from "./hooks/useHighlights";

export {
  buildCoordinateMap,
  getFeatureGappedRange,
  getMutationGappedRange,
  buildHighlights,
  findMismatchPositions,
  isInCds,
} from "./utils/coordinates";
