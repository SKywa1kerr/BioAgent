import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronRight } from "lucide-react";
import { t, type AppLanguage } from "../../i18n";
import styles from "./Splitter.module.css";

interface SplitterProps {
  onResize: (dx: number) => void;
  onCollapse?: () => void;
  ariaLabel?: string;
}

export function Splitter({ onResize, onCollapse, ariaLabel }: SplitterProps): JSX.Element {
  const [dragging, setDragging] = useState(false);
  const lastXRef = useRef<number | null>(null);
  const onResizeRef = useRef(onResize);
  useEffect(() => {
    onResizeRef.current = onResize;
  }, [onResize]);

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(true);
    lastXRef.current = e.clientX;
  }, []);

  useEffect(() => {
    if (!dragging) return;
    function onMove(ev: MouseEvent) {
      const last = lastXRef.current;
      if (last == null) return;
      const dx = ev.clientX - last;
      lastXRef.current = ev.clientX;
      if (dx !== 0) onResizeRef.current(dx);
    }
    function onUp() {
      setDragging(false);
      lastXRef.current = null;
    }
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    return () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [dragging]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        onResizeRef.current(-20);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        onResizeRef.current(20);
      } else if ((e.key === "Enter" || e.key === " ") && onCollapse) {
        e.preventDefault();
        onCollapse();
      }
    },
    [onCollapse],
  );

  return (
    <div
      className={styles.splitter}
      data-dragging={dragging ? "true" : "false"}
      role="separator"
      aria-orientation="vertical"
      aria-label={ariaLabel ?? "Resize chat column"}
      tabIndex={0}
      onMouseDown={handleMouseDown}
      onKeyDown={handleKeyDown}
    />
  );
}

interface CollapsedRailProps {
  onExpand: () => void;
  language: AppLanguage;
}

export function CollapsedRail({ onExpand, language }: CollapsedRailProps): JSX.Element {
  return (
    <button
      type="button"
      className={styles.collapsedRail}
      onClick={onExpand}
      aria-label={t(language, "splitter.expandChat")}
      title={t(language, "splitter.expandChat")}
    >
      <ChevronRight size={14} aria-hidden="true" />
      <span className={styles.collapsedRailLabel}>{t(language, "splitter.expandChat")}</span>
    </button>
  );
}
