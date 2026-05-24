import { useEffect, useRef, useState, type ReactNode } from "react";
import { t, type AppLanguage } from "../i18n";
import "./InitDialog.css";

interface BaseProps {
  open: boolean;
  language: AppLanguage;
  title?: string;
  body?: ReactNode;
  onCancel: () => void;
}

interface ConfirmProps extends BaseProps {
  kind: "confirm";
  confirmLabel?: string;
  cancelLabel?: string;
  danger?: boolean;
  onConfirm: () => void;
}

interface PromptProps extends BaseProps {
  kind: "prompt";
  defaultValue?: string;
  placeholder?: string;
  confirmLabel?: string;
  cancelLabel?: string;
  onConfirm: (value: string) => void;
}

type Props = ConfirmProps | PromptProps;

/**
 * Themed replacement for window.confirm() / window.prompt(). Electron's
 * native dialogs are unstyled OS chrome that broke the OpenWebUI-style
 * polish we've been adding everywhere else (and window.prompt is
 * actually a no-op in modern Electron renderers, so rename was silently
 * non-functional). Same .settings-modal-overlay class as the rest of
 * the app's modals so the visual stays consistent.
 */
export function AppDialog(props: Props): JSX.Element | null {
  const { open, language, title, body, onCancel } = props;
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [value, setValue] = useState("");

  useEffect(() => {
    if (!open) return;
    if (props.kind === "prompt") {
      setValue(props.defaultValue ?? "");
      // Focus the input after the dialog mounts.
      const id = window.setTimeout(() => {
        inputRef.current?.focus();
        inputRef.current?.select();
      }, 30);
      return () => window.clearTimeout(id);
    }
    return undefined;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
      if (e.key === "Enter" && props.kind === "prompt") {
        e.preventDefault();
        props.onConfirm(value);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onCancel, props, value]);

  if (!open) return null;

  const fallbackConfirm =
    props.kind === "confirm" ? props.confirmLabel ?? t(language, "common.confirm") : props.confirmLabel ?? t(language, "common.ok");
  const fallbackCancel = props.cancelLabel ?? t(language, "common.cancel");

  return (
    <div
      className="settings-modal-overlay"
      onClick={onCancel}
      role="dialog"
      aria-modal="true"
      aria-label={title}
    >
      <div className="settings-modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 420 }}>
        {title ? <h3 style={{ marginTop: 0 }}>{title}</h3> : null}
        {body ? (
          <div style={{ color: "var(--text-muted)", fontSize: 13, lineHeight: 1.55 }}>{body}</div>
        ) : null}

        {props.kind === "prompt" ? (
          <input
            ref={inputRef}
            type="text"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder={props.placeholder}
            style={{
              width: "100%",
              marginTop: 14,
              padding: "8px 10px",
              border: "1px solid var(--border-default)",
              borderRadius: 8,
              background: "var(--bg-app)",
              color: "var(--text-main)",
              font: "inherit",
              fontSize: 14,
            }}
          />
        ) : null}

        <div className="init-dialog-actions">
          <button type="button" className="init-dialog-secondary" onClick={onCancel}>
            {fallbackCancel}
          </button>
          <button
            type="button"
            className="init-dialog-primary"
            onClick={() => {
              if (props.kind === "confirm") props.onConfirm();
              else props.onConfirm(value);
            }}
            style={
              props.kind === "confirm" && props.danger
                ? { background: "#c64f4f" }
                : undefined
            }
          >
            {fallbackConfirm}
          </button>
        </div>
      </div>
    </div>
  );
}
