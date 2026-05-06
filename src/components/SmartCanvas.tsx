import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ReactNode } from "react";

export type PanelType = "text" | "analysis" | "trends" | "suggestions" | "confirmation";

interface SmartCanvasProps {
  title: string;
  panelType: PanelType;
  children: ReactNode;
}

export function SmartCanvas({ title, panelType, children }: SmartCanvasProps) {
  const reduceMotion = useReducedMotion() ?? false;
  const initial = reduceMotion ? { opacity: 0 } : { opacity: 0, y: 10 };
  const animate = reduceMotion ? { opacity: 1 } : { opacity: 1, y: 0 };
  const exit = reduceMotion ? { opacity: 0 } : { opacity: 0, y: -10 };

  return (
    <div className="canvas-shell">
      <div className="panel-title">{title}</div>
      <div className="canvas-body">
        <AnimatePresence mode="wait">
          <motion.div
            key={panelType}
            initial={initial}
            animate={animate}
            exit={exit}
            transition={{ duration: 0.14, ease: [0.2, 0.7, 0.2, 1] }}
            className="canvas-motion"
          >
            {children}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
