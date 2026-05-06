export interface ViewportState {
  start: number;
  end: number;
}

export function clampViewport(
  viewport: { start?: number; end?: number } | null | undefined,
  total: number,
  minWindow?: number,
): ViewportState;

export function zoomViewport(
  viewport: { start?: number; end?: number } | null | undefined,
  factor: number,
  anchor: number | null | undefined,
  total: number,
  minWindow?: number,
): ViewportState;

export function panViewport(
  viewport: { start?: number; end?: number } | null | undefined,
  delta: number,
  total: number,
  minWindow?: number,
): ViewportState;

export function centerViewport(
  viewport: { start?: number; end?: number } | null | undefined,
  targetBase: number,
  total: number,
  minWindow?: number,
): ViewportState;

export function viewportZoomLevel(
  viewport: { start?: number; end?: number } | null | undefined,
  total: number,
): number;

export const CHROMATOGRAM_VIEWPORT_DEFAULT_MIN_WINDOW: number;
