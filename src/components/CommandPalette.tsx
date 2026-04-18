import { useEffect, useMemo, useRef, useState } from "react";
import { filterCommands } from "../lib/commands/registry";
import type { Command, CommandGroup } from "../lib/commands/registry";
import type { AppLanguage } from "../i18n";
import { t } from "../i18n";
import "./CommandPalette.css";

const GROUP_ORDER: CommandGroup[] = ["nav", "workbench", "appearance", "examples", "log"];
const GROUP_LABEL_KEY: Record<CommandGroup, string> = {
  nav: "palette.groupNav",
  workbench: "palette.groupWorkbench",
  appearance: "palette.groupAppearance",
  examples: "palette.groupExamples",
  log: "palette.groupLog",
};

interface Props {
  open: boolean;
  onClose: () => void;
  language: AppLanguage;
}

export function CommandPalette({ open, onClose, language }: Props) {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const previouslyFocusedRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!open) return;
    previouslyFocusedRef.current = document.activeElement as HTMLElement | null;
    setQuery("");
    setSelectedIndex(0);
    requestAnimationFrame(() => inputRef.current?.focus());
    return () => {
      previouslyFocusedRef.current?.focus?.();
    };
  }, [open]);

  const items = useMemo(() => (open ? filterCommands(query) : []), [open, query]);

  useEffect(() => {
    if (selectedIndex >= items.length) setSelectedIndex(Math.max(0, items.length - 1));
  }, [items, selectedIndex]);

  if (!open) return null;

  async function runAndClose(cmd: Command) {
    onClose();
    try {
      await cmd.run();
    } catch (err) {
      console.error(`Command "${cmd.id}" failed:`, err);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLDivElement>) {
    if (e.key === "Escape") {
      e.preventDefault();
      onClose();
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex((i) => Math.min(items.length - 1, i + 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex((i) => Math.max(0, i - 1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      const cmd = items[selectedIndex];
      if (cmd) runAndClose(cmd);
    } else if (e.key === "Tab") {
      e.preventDefault();
    }
  }

  const grouped = groupItems(items);

  return (
    <div className="command-palette-scrim" onMouseDown={onClose} role="presentation">
      <div
        className="command-palette"
        role="dialog"
        aria-label={t(language, "palette.title")}
        aria-modal="true"
        onMouseDown={(e) => e.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
        <input
          ref={inputRef}
          type="text"
          className="command-palette-input"
          placeholder={t(language, "palette.placeholder")}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label={t(language, "palette.placeholder")}
        />
        <div className="command-palette-list" role="listbox">
          {items.length === 0 ? (
            <div className="command-palette-empty">{t(language, "palette.empty")}</div>
          ) : (
            GROUP_ORDER.map((g) => {
              const list = grouped.get(g);
              if (!list || list.length === 0) return null;
              return (
                <div key={g} className="command-palette-group">
                  <div className="command-palette-group-label">{t(language, GROUP_LABEL_KEY[g])}</div>
                  {list.map((cmd) => {
                    const flatIndex = items.indexOf(cmd);
                    const active = flatIndex === selectedIndex;
                    return (
                      <button
                        key={cmd.id}
                        type="button"
                        role="option"
                        aria-selected={active}
                        className={`command-palette-item${active ? " is-active" : ""}`}
                        onMouseEnter={() => setSelectedIndex(flatIndex)}
                        onClick={() => runAndClose(cmd)}
                      >
                        <span className="command-palette-item-title">{cmd.title}</span>
                        {cmd.shortcut ? <span className="command-palette-item-shortcut">{cmd.shortcut}</span> : null}
                      </button>
                    );
                  })}
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}

function groupItems(items: Command[]): Map<CommandGroup, Command[]> {
  const m = new Map<CommandGroup, Command[]>();
  for (const cmd of items) {
    const list = m.get(cmd.group) ?? [];
    list.push(cmd);
    m.set(cmd.group, list);
  }
  return m;
}
