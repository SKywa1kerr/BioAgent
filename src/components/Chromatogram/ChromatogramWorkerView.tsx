import React, { useRef, useEffect, useState, memo } from "react";
import { ChromatogramData } from "../../types";

interface ChromatogramWorkerViewProps {
  data: ChromatogramData;
  totalBases: number;
  alignedQueryG: string;
  gappedToQueryIdx: (number | null)[];
  baseWidth: number;
  traceHeight: number;
}

// Cache for rendered chromatogram bitmaps
const renderCache = new Map<string, ImageBitmap>();

export const ChromatogramWorkerView: React.FC<ChromatogramWorkerViewProps> = memo(({
  data,
  totalBases,
  alignedQueryG,
  gappedToQueryIdx,
  baseWidth,
  traceHeight
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isReady, setIsReady] = useState(false);
  const width = totalBases * baseWidth;

  // Create a cache key based on data content
  const cacheKey = React.useMemo(() => {
    // Use data length and first/last few values as a fingerprint
    const tracesLen = data.traces.A.length;
    const firstA = data.traces.A[0] || 0;
    const lastA = data.traces.A[tracesLen - 1] || 0;
    const locLen = data.baseLocations.length;
    return `chrom-${tracesLen}-${firstA}-${lastA}-${locLen}-${totalBases}-${baseWidth}`;
  }, [data, totalBases, baseWidth]);

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
      return;
    }

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
        opacity: isReady ? 1 : 0.5,
        transition: "opacity 0.2s"
      }}
      className="trace-canvas-worker"
    />
  );
});
