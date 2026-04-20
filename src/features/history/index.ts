// History Feature - Analysis history with SQLite persistence
// Lazy-loaded feature for viewing past analyses

export { HistoryPage } from "./components/HistoryPage";

export interface HistoryEntry {
  id: string;
  timestamp: number;
  sampleName: string;
  result: AnalysisResult;
}

export interface AnalysisResult {
  identity: number;
  coverage: number;
  mutations: number;
  status: "ok" | "wrong" | "error";
}
