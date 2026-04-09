interface ElectronAPI {
  invoke: (channel: string, ...args: unknown[]) => Promise<unknown>;
  onAnalysisProgress: (
    callback: (payload: {
      stage: string;
      percent: number;
      processedSamples: number;
      totalSamples: number;
      sampleId?: string | null;
      message?: string;
    }) => void
  ) => () => void;
}

declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}

export {};
