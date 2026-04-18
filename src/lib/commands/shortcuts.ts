export interface ShortcutEntry {
  key: string;
  actionKey: string;
}

export const SHORTCUTS: ShortcutEntry[] = [
  { key: "Ctrl+K", actionKey: "shortcuts.action.palette" },
  { key: "Ctrl+L", actionKey: "shortcuts.action.focusChat" },
  { key: "Ctrl+,", actionKey: "shortcuts.action.settings" },
  { key: "Enter", actionKey: "shortcuts.action.send" },
  { key: "Shift+Enter", actionKey: "shortcuts.action.newline" },
  { key: "?", actionKey: "shortcuts.action.overlay" },
  { key: "Esc", actionKey: "shortcuts.action.close" },
];
