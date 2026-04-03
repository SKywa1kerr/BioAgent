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
  const renderedKeyRef = useRef<string | null>(null);
  const [isReady, setIsReady] = useState(false);
  const width = totalBases * baseWidth;

  // Stable cache key based on sample and dimensions
  const cacheKey = React.useMemo(() => {
    return `chrom-${sampleId}-${totalBases}-${baseWidth}-${traceHeight}`;
  }, [sampleId, totalBases, baseWidth, traceHeight]);

  // Store latest props in refs for worker access without triggering re-renders
  const propsRef = useRef({ data, alignedQueryG, gappedToQueryIdx, width, traceHeight });
  useEffect(() => {
    propsRef.current = { data, alignedQueryG, gappedToQueryIdx, width, traceHeight };
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const dpr = window.devicePixelRatio || 1;
    const targetWidth = Math.floor(width * dpr);
    const targetHeight = Math.floor(traceHeight * dpr);

    // Only set canvas size when it changes
    if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
      canvas.width = targetWidth;
      canvas.height = targetHeight;
      renderedKeyRef.current = null; // Canvas was cleared
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Check if we already rendered this exact cacheKey to this canvas
    if (renderedKeyRef.current === cacheKey) {
      setIsReady(true);
      return;
    }

    // Check global cache
    const cached = renderCache.get(cacheKey);
    if (cached) {
      ctx.drawImage(cached, 0, 0);
      renderedKeyRef.current = cacheKey;
      setIsReady(true);
      console.log('[Chromatogram] Cache hit:', cacheKey);
      return;
    }

    console.log('[Chromatogram] Rendering:', cacheKey);
    setIsReady(false);

    // Get latest props from refs
    const { data: currentData, alignedQueryG: currentAlignedQueryG, gappedToQueryIdx: currentGappedToQueryIdx } = propsRef.current;

    // Create offscreen canvas for worker rendering
    const offscreen = new OffscreenCanvas(targetWidth, targetHeight);

    const worker = new Worker(
      new URL("../../workers/chromatogram.worker.ts", import.meta.url),
      { type: "module" }
    );

    // Encode gappedToQueryIdx
    const gqiBuffer = new Int32Array(currentGappedToQueryIdx.length);
    for (let i = 0; i < currentGappedToQueryIdx.length; i++) {
      gqiBuffer[i] = currentGappedToQueryIdx[i] ?? -1;
    }

    // Downsample trace data
    const pixelsWide = Math.ceil(width);
    const originalLength = currentData.traces.A.length;
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

    const traceA = downsampleArray(currentData.traces.A);
    const traceT = downsampleArray(currentData.traces.T);
    const traceG = downsampleArray(currentData.traces.G);
    const traceC = downsampleArray(currentData.traces.C);

    const baseLocBuf = new Int32Array(currentData.baseLocations.length);
    for (let i = 0; i < currentData.baseLocations.length; i++) {
      baseLocBuf[i] = Math.floor(currentData.baseLocations[i] / downsampleStep);
    }

    const mixedBuf = new Int32Array(currentData.mixedPeaks.length);
    for (let i = 0; i < currentData.mixedPeaks.length; i++) {
      mixedBuf[i] = Math.floor(currentData.mixedPeaks[i] / downsampleStep);
    }

    const qualityBuf = new Uint8Array(currentData.quality);

    const payload = {
      canvas: offscreen,
      traces: { A: traceA, T: traceT, G: traceG, C: traceC },
      quality: qualityBuf,
      baseLocations: baseLocBuf,
      mixedPeaks: mixedBuf,
      gappedToQueryIdx: gqiBuffer,
      totalBases,
      alignedQueryG: currentAlignedQueryG,
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
          renderedKeyRef.current = cacheKey;
          // Clone the bitmap for caching
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
  }, [cacheKey, width, traceHeight, totalBases, baseWidth]);

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
