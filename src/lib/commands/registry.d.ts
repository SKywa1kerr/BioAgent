export type CommandGroup = "nav" | "workbench" | "appearance" | "examples" | "log";

export interface Command {
  id: string;
  title: string;
  group: CommandGroup;
  keywords?: string[];
  shortcut?: string;
  when?: () => boolean;
  run: () => void | Promise<void>;
}

export function registerCommand(cmd: Command): () => void;
export function getCommands(): Command[];
export function filterCommands(query: string): Command[];
export function clearCommands(): void;
