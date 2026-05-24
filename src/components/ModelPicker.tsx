import { useEffect, useRef, useState } from "react";
import { ChevronDown, Check } from "lucide-react";

interface ModelPickerProps {
  currentModel: string;
  availableModels: readonly string[];
  providerLabel: string;
  onChange: (model: string) => void;
  onOpenSettings?: () => void;
  /** Localized strings injected by the host so this stays i18n-pure. */
  labels: {
    current: string;
    settingsCTA: string;
    chooseLabel: string;
  };
}

/**
 * OpenWebUI-style top-of-chat model selector. Shows the current model as a
 * pill button; clicking it opens a popover with the configured provider's
 * suggested models so the user can switch without diving into the settings
 * modal. Custom models the user has typed manually appear at the top with
 * a "(custom)" tag.
 */
export function ModelPicker({
  currentModel,
  availableModels,
  providerLabel,
  onChange,
  onOpenSettings,
  labels,
}: ModelPickerProps): JSX.Element {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (event: MouseEvent) => {
      if (!rootRef.current) return;
      if (!rootRef.current.contains(event.target as Node)) setOpen(false);
    };
    const onEsc = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("mousedown", handler);
    window.addEventListener("keydown", onEsc);
    return () => {
      window.removeEventListener("mousedown", handler);
      window.removeEventListener("keydown", onEsc);
    };
  }, [open]);

  const labelText = currentModel || labels.current;
  const isCustom = currentModel && !availableModels.includes(currentModel);
  const items: { id: string; label: string; tag?: string }[] = [];
  if (isCustom) items.push({ id: currentModel, label: currentModel, tag: "custom" });
  for (const m of availableModels) {
    if (m && m !== currentModel) items.push({ id: m, label: m });
    else if (m === currentModel && !isCustom) items.push({ id: m, label: m });
  }

  return (
    <div className="model-picker" ref={rootRef}>
      <button
        type="button"
        className="model-picker-trigger"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="listbox"
        aria-expanded={open}
        title={labels.chooseLabel}
      >
        <span className="model-picker-provider" aria-hidden>{providerLabel}</span>
        <span className="model-picker-name">{labelText}</span>
        <ChevronDown size={14} strokeWidth={1.7} aria-hidden />
      </button>
      {open ? (
        <div className="model-picker-menu" role="listbox" aria-label={labels.chooseLabel}>
          {items.length === 0 ? (
            <div className="model-picker-empty">{labels.chooseLabel}</div>
          ) : (
            items.map((it) => {
              const active = it.id === currentModel;
              return (
                <button
                  key={it.id}
                  type="button"
                  role="option"
                  aria-selected={active}
                  className={`model-picker-item${active ? " is-active" : ""}`}
                  onClick={() => {
                    onChange(it.id);
                    setOpen(false);
                  }}
                >
                  <span className="model-picker-item-label">{it.label}</span>
                  {it.tag ? <span className="model-picker-item-tag">{it.tag}</span> : null}
                  {active ? <Check size={14} strokeWidth={1.7} aria-hidden /> : null}
                </button>
              );
            })
          )}
          {onOpenSettings ? (
            <button
              type="button"
              className="model-picker-settings"
              onClick={() => {
                setOpen(false);
                onOpenSettings();
              }}
            >
              {labels.settingsCTA}
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
