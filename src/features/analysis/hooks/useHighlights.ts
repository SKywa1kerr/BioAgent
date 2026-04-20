import { useMemo, useState, useCallback } from "react";
import { useAlignment } from "./useAlignment";
import type { HighlightRegion } from "../types/sequencing";

export interface HighlightState {
  // All highlight regions
  regions: HighlightRegion[];

  // Filtered by type
  cds: HighlightRegion | null;
  mutations: HighlightRegion[];
  proteinMutations: HighlightRegion[];

  // Navigation
  currentIndex: number;
  navigateToNext: () => void;
  navigateToPrev: () => void;
  navigateTo: (index: number) => void;

  // Current highlight for navigation
  currentHighlight: HighlightRegion | null;

  // Check if position is highlighted
  isHighlighted: (gappedPos: number) => boolean;
  getHighlightAt: (gappedPos: number) => HighlightRegion | undefined;
}

/**
 * Hook for reactive highlighting with navigation.
 *
 * Automatically updates when:
 * - Alignment changes (coordinates shift)
 * - Mutations are recalculated
 * - CDS boundaries change
 */
export function useHighlights(): HighlightState | null {
  const alignment = useAlignment();
  const [currentIndex, setCurrentIndex] = useState(0);

  return useMemo(() => {
    if (!alignment) {
      return null;
    }

    const { highlights, cdsHighlight, mutationHighlights } = alignment;

    // Combine all navigable highlights (mutations only for now)
    const navigableHighlights = mutationHighlights;

    const navigateToNext = () => {
      if (navigableHighlights.length === 0) return;
      setCurrentIndex((prev) => (prev + 1) % navigableHighlights.length);
    };

    const navigateToPrev = () => {
      if (navigableHighlights.length === 0) return;
      setCurrentIndex(
        (prev) =>
          (prev - 1 + navigableHighlights.length) % navigableHighlights.length
      );
    };

    const navigateTo = (index: number) => {
      if (index >= 0 && index < navigableHighlights.length) {
        setCurrentIndex(index);
      }
    };

    const isHighlighted = (gappedPos: number) => {
      return highlights.some(
        (h) => gappedPos >= h.start && gappedPos < h.end
      );
    };

    const getHighlightAt = (gappedPos: number) => {
      return highlights.find((h) => gappedPos >= h.start && gappedPos < h.end);
    };

    return {
      regions: highlights,
      cds: cdsHighlight || null,
      mutations: mutationHighlights,
      proteinMutations: mutationHighlights.filter(
        (h) => h.type === "protein-mutation"
      ),
      currentIndex,
      navigateToNext,
      navigateToPrev,
      navigateTo,
      currentHighlight:
        navigableHighlights[currentIndex] ||
        navigableHighlights[0] ||
        null,
      isHighlighted,
      getHighlightAt,
    };
  }, [alignment, currentIndex]);
}

/**
 * Hook for hover-based highlighting with tooltip data.
 */
export function useHoverHighlight() {
  const [hoverPos, setHoverPos] = useState<number | null>(null);
  const alignment = useAlignment();

  const handleMouseMove = useCallback(
    (e: React.MouseEvent, containerRef: React.RefObject<HTMLElement>) => {
      if (!containerRef.current || !alignment) return;

      const rect = containerRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left - 100; // 100px gutter
      const gappedPos = Math.floor(x / alignment.baseWidth);

      if (gappedPos >= 0 && gappedPos < alignment.totalBases) {
        setHoverPos(gappedPos);
        return gappedPos;
      }
      return null;
    },
    [alignment]
  );

  const handleMouseLeave = useCallback(() => {
    setHoverPos(null);
  }, []);

  return {
    hoverPos,
    setHoverPos,
    handleMouseMove,
    handleMouseLeave,
  };
}
