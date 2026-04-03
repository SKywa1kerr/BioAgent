import React, { useRef, useEffect, useState, memo } from "react";
import { ChromatogramData } from "../../types";

interface ChromatogramWorkerViewProps {
  data: ChromatogramData;
  sampleId: string;
  totalBases: number;
  alignedQueryG: string;
  gappedToQueryIdx: (number | null)[];
  baseWidth: number;
  traceHeight: number;
  visible?: boolean;
}

// Cache for rendered chromatogram ImageBitmaps
// ImageBitmap can be drawn multiple times but not transferred after first use
const renderCache = new Map<string, ImageBitmap>();

export const ChromatogramWorkerView: React.FC<ChromatogramWorkerViewProps> = memo(({
  data,
  sampleId,
  totalBases,
  alignedQueryG,
  gappedToQueryIdx,
  baseWidth,
  traceHeight,
  visible = true
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const renderedRef = useRef<boolean>(false);
  const [isReady, setIsReady] = useState(false);
  const width = totalBases * baseWidth;

  // Stable cache key based on sample and dimensions
  const cacheKey = React.useMemo(() => {
    return `chrom-${sampleId}-${totalBases}-${baseWidth}-${traceHeight}`;
  }, [sampleId, totalBases, baseWidth, traceHeight]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const dpr = window.devicePixelRatio || 1;
    const targetWidth = Math.floor(width * dpr);
    const targetHeight = Math.floor(traceHeight * dpr);

    // Only set canvas size once to avoid clearing
    if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
      canvas.width = targetWidth;
      canvas.height = targetHeight;
      renderedRef.current = false;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Check if we already rendered this cacheKey to this canvas
    if (renderedRef.current) {
      setIsReady(true);
      return;
    }

    // Check global cache
    const cached = renderCache.get(cacheKey);
    if (cached) {
      ctx.drawImage(cached, 0, 0);
      renderedRef.current = true;
      setIsReady(true);
      console.log('[Chromatogram] Cache hit:', cacheKey.slice(0, 50));
      return;
    }

    console.log('[Chromatogram] Rendering:', cacheKey.slice(0, 50));
    setIsReady(false);

    // Create offscreen canvas for worker rendering
    const offscreen = new OffscreenCanvas(targetWidth, targetHeight);

    const worker = new Worker(
      new URL("../../workers/chromatogram.worker.ts", import.meta.url),
      { type: "module" }
    );

    // Encode gappedToQueryIdx
    const gqiBuffer = new Int32Array(gappedToQueryIdx.length);
    for (let i = 0; i < gappedToQueryIdx.length; i++) {
      gqiBuffer[i] = gappedToQueryIdx[i] ?? -1;
    }

    // Downsample trace data
    const pixelsWide = Math.ceil(width);
    const originalLength = data.traces.A.length;
    const downsampleStep = Math.max(1, Math.floor(originalLength / (pixelsWide * 3)));

    const downsampleArray = (arr: number[]): Float32Array => {
      if (downsampleStep === 1) return new Float32Array(arr);
      const result = new Float32Array(Math.ceil(arr.length / downsampleStep));
      for (let i = 0, j = 0; i < arr.length; i += downsampleStep, j++) {
        let maxVal = arr[i];
        for (let k = i + 1; k < Math.min(i + downsampleStep, arr.length); k++) {
          if (arr[k] > maxVal) maxVal = arr[k];
        }
        result[j] = maxVal;
      }
      return result;
    };

    const traceA = downsampleArray(data.traces.A);
    const traceT = downsampleArray(data.traces.T);
    const traceG = downsampleArray(data.traces.G);
    const traceC = downsampleArray(data.traces.C);

    const baseLocBuf = new Int32Array(data.baseLocations.length);
    for (let i = 0; i < data.baseLocations.length; i++) {
      baseLocBuf[i] = Math.floor(data.baseLocations[i] / downsampleStep);
    }

    const mixedBuf = new Int32Array(data.mixedPeaks.length);
    for (let i = 0; i < data.mixedPeaks.length; i++) {
      mixedBuf[i] = Math.floor(data.mixedPeaks[i] / downsampleStep);
    }

    const qualityBuf = new Uint8Array(data.quality);

    const payload = {
      canvas: offscreen,
      traces: { A: traceA, T: traceT, G: traceG, C: traceC },
      quality: qualityBuf,
      baseLocations: baseLocBuf,
      mixedPeaks: mixedBuf,
      gappedToQueryIdx: gqiBuffer,
      totalBases,
      alignedQueryG,
      baseWidth,
      traceHeight,
      dpr,
    };

    const transferables: Transferable[] = [
      offscreen,
      traceA.buffer,
      traceT.buffer,
      traceG.buffer,
      traceC.buffer,
      qualityBuf.buffer,
      baseLocBuf.buffer,
      mixedBuf.buffer,
      gqiBuffer.buffer,
    ];

    worker.onmessage = (e) => {
      if (e.data.type === "rendered") {
        const bitmap = e.data.bitmap as ImageBitmap;
        if (bitmap && ctx) {
          ctx.drawImage(bitmap, 0, 0);
          renderedRef.current = true;
          // Clone the bitmap for caching by drawing to a new canvas
          const cacheCanvas = new OffscreenCanvas(targetWidth, targetHeight);
          const cacheCtx = cacheCanvas.getContext("2d");
          if (cacheCtx) {
            cacheCtx.drawImage(bitmap, 0, 0);
            createImageBitmap(cacheCanvas).then(cachedBitmap => {
              renderCache.set(cacheKey, cachedBitmap);
            });
          }
          setIsReady(true);
        }
        worker.terminate();
      }
    };

    worker.postMessage(payload, transferables);

    return () => {
      worker.terminate();
    };
  }, [cacheKey, data, gappedToQueryIdx, alignedQueryG, width, traceHeight]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: `${width}px`,
        height: `${traceHeight}px`,
        opacity: visible ? (isReady ? 1 : 0.5) : 0,
        transition: "opacity 0.2s",
        pointerEvents: visible ? "auto" : "none"
      }}
      className="trace-canvas-worker"
    />
  );
});
