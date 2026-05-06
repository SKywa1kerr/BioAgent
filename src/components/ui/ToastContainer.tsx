import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { AnimatePresence, useReducedMotion } from "framer-motion";
import { useToastsState } from "./ToastProvider";
import { Toast } from "./Toast";

export function ToastContainer(): JSX.Element | null {
  const { toasts } = useToastsState();
  const [mounted, setMounted] = useState(false);
  const reduceMotion = useReducedMotion() ?? false;

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted || typeof document === "undefined" || !document.body) return null;

  return createPortal(
    <div
      className="toast-container"
      role="status"
      aria-live="polite"
      aria-atomic="false"
    >
      <AnimatePresence initial={false}>
        {toasts.map((toast) => (
          <Toast key={toast.id} toast={toast} reduceMotion={reduceMotion} />
        ))}
      </AnimatePresence>
    </div>,
    document.body,
  );
}
