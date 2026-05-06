import { motion } from "framer-motion";
import type { Toast as ToastType, ToastKind } from "../../lib/ui/toastReducer";
import { Icon, type IconName } from "./Icon";
import { useToasts, useToastActionDispatcher } from "./ToastProvider";

interface Props {
  toast: ToastType;
  reduceMotion: boolean;
}

const KIND_ICON: Record<ToastKind, IconName> = {
  info: "info",
  success: "success",
  warning: "alert",
  error: "alert",
};

const EASING = [0.2, 0.7, 0.2, 1] as const;
const DURATION_S = 0.14;

export function Toast({ toast, reduceMotion }: Props): JSX.Element {
  const { dismissToast } = useToasts();
  const dispatchAction = useToastActionDispatcher();

  const initial = reduceMotion ? { opacity: 0 } : { opacity: 0, x: 20, y: 8 };
  const animate = reduceMotion ? { opacity: 1 } : { opacity: 1, x: 0, y: 0 };
  const exit = reduceMotion ? { opacity: 0 } : { opacity: 0, x: 20 };

  return (
    <motion.div
      className={`toast toast--${toast.kind}`}
      role="status"
      layout
      initial={initial}
      animate={animate}
      exit={exit}
      transition={{ duration: DURATION_S, ease: EASING }}
    >
      <div className="toast-icon" aria-hidden="true">
        <Icon name={KIND_ICON[toast.kind]} size={18} />
      </div>
      <div className="toast-body">
        <div className="toast-title">{toast.title}</div>
        {toast.description !== undefined && (
          <div className="toast-desc">{toast.description}</div>
        )}
        {toast.action !== undefined && (
          <button
            type="button"
            className="toast-action"
            onClick={() => {
              if (toast.action) dispatchAction(toast.action.actionId, toast.id);
            }}
          >
            {toast.action.label}
          </button>
        )}
      </div>
      <button
        type="button"
        className="toast-close"
        aria-label="Close"
        onClick={() => dismissToast(toast.id)}
      >
        <Icon name="close" size={14} />
      </button>
    </motion.div>
  );
}
