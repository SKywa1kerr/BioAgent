import React, { useRef, useEffect, useState, memo } from "react";
import { ChromatogramData } from "../../types";

interface ChromatogramWorkerViewProps {
  data: ChromatogramData;
  totalBases: number;
  alignedQueryG: string;
  gappedToQueryIdx: (number | null)[];
  baseWidth: number;
  traceHeight: number;
  visible?: boolean;
}

// Cache for rendered chromatogram bitmaps
const renderCache = new Map<string, ImageBitmap>();

export const ChromatogramWorkerView: React.FC<ChromatogramWorkerViewProps> = memo(({
  data,
  totalBases,
  alignedQueryG,
  gappedToQueryIdx,
  baseWidth,
  traceHeight,
  visible = true
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isReady, setIsReady] = useState(false);
  const width = totalBases * baseWidth;

  // Create a stable cache key based on actual data content (not object reference)
  // Use the array references themselves as they are stable from the sample data
  const cacheKey = React.useMemo(() => {
    // Use array lengths and sample points for a stable fingerprint
    const tracesLen = data.traces.A.length;
    const locLen = data.baseLocations.length;
    // Sample a few points from the middle of the trace for uniqueness
    const midIdx = Math.floor(tracesLen / 2);
    const sampleA = data.traces.A[midIdx] || 0;
    const sampleT = data.traces.T[midIdx] || 0;
    return `chrom-${tracesLen}-${locLen}-${sampleA}-${sampleT}-${totalBases}-${baseWidth}`;
  }, [
    // Only depend on the actual array references and primitive values
    data.traces.A,
    data.traces.T,
    data.traces.G,
    data.traces.C,
    data.baseLocations,
    totalBases,
    baseWidth
  ]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const dpr = window.devicePixelRatio || 1;
    const targetWidth = Math.floor(width * dpr);
    const targetHeight = Math.floor(traceHeight * dpr);

    // Set canvas size
    canvas.width = targetWidth;
    canvas.height = targetHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Check cache first
    const cached = renderCache.get(cacheKey);
    if (cached) {
      ctx.drawImage(cached, 0, 0);
      setIsReady(true);
      console.log('[Chromatogram] Cache hit:', cacheKey.slice(0, 50) + '...');
      return;
    }
    console.log('[Chromatogram] Cache miss - rendering:', cacheKey.slice(0, 50) + '...');

    setIsReady(false);

    // Create offscreen canvas for worker rendering
    const offscreen = new OffscreenCanvas(targetWidth, targetHeight);

    const worker = new Worker(
      new URL("../../workers/chromatogram.worker.ts", import.meta.url),
      { type: "module" }
    );

    // Encode gappedToQueryIdx as Int32Array (-1 for null) for efficient transfer
    const gqiBuffer = new Int32Array(gappedToQueryIdx.length);
    for (let i = 0; i < gappedToQueryIdx.length; i++) {
      gqiBuffer[i] = gappedToQueryIdx[i] ?? -1;
    }

    const traceA = new Float32Array(data.traces.A);
    const traceT = new Float32Array(data.traces.T);
    const traceG = new Float32Array(data.traces.G);
    const traceC = new Float32Array(data.traces.C);
    const qualityBuf = new Uint8Array(data.quality);
    const baseLocBuf = new Int32Array(data.baseLocations);
    const mixedBuf = new Int32Array(data.mixedPeaks);

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
        // Copy from offscreen to visible canvas
        const bitmap = e.data.bitmap as ImageBitmap;
        if (bitmap && ctx) {
          ctx.drawImage(bitmap, 0, 0);
          // Cache the bitmap for future use
          renderCache.set(cacheKey, bitmap);
          setIsReady(true);
        }
        worker.terminate();
      }
    };

    worker.postMessage(payload, transferables);

    return () => {
      worker.terminate();
    };
  }, [cacheKey, width, traceHeight]); // Only re-render when data actually changes

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
