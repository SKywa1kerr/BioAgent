type CommandIntentAction = {
  id: string;
  args: Record<string, unknown>;
};

type CommandIntentPlan = {
  summary: string;
  actions: CommandIntentAction[];
  needsConfirmation: boolean;
};

interface ElectronAPI {
  invoke: (channel: string, ...args: unknown[]) => Promise<unknown>;
  interpretCommand: (text: string) => Promise<CommandIntentPlan>;
  openExportFolder: (exportedPath: string) => Promise<boolean>;
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
