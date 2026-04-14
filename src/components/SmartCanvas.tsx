import { AnimatePresence, motion } from "framer-motion";
import { ReactNode } from "react";

export type PanelType = "text" | "analysis" | "trends" | "suggestions" | "confirmation";

interface SmartCanvasProps {
  title: string;
  panelType: PanelType;
  children: ReactNode;
}

export function SmartCanvas({ title, panelType, children }: SmartCanvasProps) {
  return (
    <div className="canvas-shell">
      <div className="panel-title">{title}</div>
      <div className="canvas-body">
        <AnimatePresence mode="wait">
          <motion.div
            key={panelType}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.22 }}
            className="canvas-motion"
          >
            {children}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
