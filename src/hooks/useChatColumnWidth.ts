import { useCallback, useEffect, useRef, useState } from "react";
import {
  computeNextWidth,
  loadChatWidthState,
  saveChatWidthState,
  SPLITTER_CONSTANTS,
  type ChatWidthState,
} from "../lib/ui/chatColumnWidth.js";

interface UseChatColumnWidthApi {
  width: number;
  collapsed: boolean;
  lastExpandedWidth: number;
  applyDelta: (dx: number, containerWidth: number, canvasMin: number) => void;
  setWidth: (next: number, containerWidth: number, canvasMin: number) => void;
  collapse: () => void;
  expand: () => void;
}

export function useChatColumnWidth(): UseChatColumnWidthApi {
  const [state, setState] = useState<ChatWidthState>(() => loadChatWidthState());
  const stateRef = useRef(state);
  useEffect(() => {
    stateRef.current = state;
    saveChatWidthState(state);
  }, [state]);

  const setWidth = useCallback((next: number, containerWidth: number, canvasMin: number) => {
    setState((prev) => {
      const r = computeNextWidth(next, containerWidth, canvasMin);
      const lastExpandedWidth = r.collapsed ? prev.lastExpandedWidth : r.width;
      return { width: r.width, collapsed: r.collapsed, lastExpandedWidth };
    });
  }, []);

  const applyDelta = useCallback(
    (dx: number, containerWidth: number, canvasMin: number) => {
      const prev = stateRef.current;
      const desired = (prev.collapsed ? prev.lastExpandedWidth : prev.width) + dx;
      setWidth(desired, containerWidth, canvasMin);
    },
    [setWidth],
  );

  const collapse = useCallback(() => {
    setState((prev) => ({
      width: SPLITTER_CONSTANTS.RAIL_WIDTH,
      collapsed: true,
      lastExpandedWidth: prev.collapsed ? prev.lastExpandedWidth : prev.width,
    }));
  }, []);

  const expand = useCallback(() => {
    setState((prev) => ({
      width: prev.lastExpandedWidth || SPLITTER_CONSTANTS.DEFAULT_WIDTH,
      collapsed: false,
      lastExpandedWidth: prev.lastExpandedWidth || SPLITTER_CONSTANTS.DEFAULT_WIDTH,
    }));
  }, []);

  return {
    width: state.width,
    collapsed: state.collapsed,
    lastExpandedWidth: state.lastExpandedWidth,
    applyDelta,
    setWidth,
    collapse,
    expand,
  };
}
