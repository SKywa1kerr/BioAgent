import React, { useRef, useEffect, memo } from "react";
import { ChromatogramData } from "../../types";

interface ChromatogramWorkerViewProps {
  data: ChromatogramData;
  totalBases: number;
  alignedQueryG: string;
  gappedToQueryIdx: (number | null)[];
  baseWidth: number;
  traceHeight: number;
}

export const ChromatogramWorkerView: React.FC<ChromatogramWorkerViewProps> = memo(({
  data,
  totalBases,
  alignedQueryG,
  gappedToQueryIdx,
  baseWidth,
  traceHeight
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const workerRef = useRef<Worker | null>(null);
  const width = totalBases * baseWidth;

  // Single effect: create worker + transfer canvas + send data.
  // Re-runs when data changes — tears down old worker and creates fresh one.
  // This avoids the bug where OffscreenCanvas can only be transferred once
  // but a stale worker tries to reuse a neutered context.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const worker = new Worker(
      new URL("../../workers/chromatogram.worker.ts", import.meta.url),
      { type: "module" }
    );
    workerRef.current = worker;

    const dpr = window.devicePixelRatio || 1;

    // Encode gappedToQueryIdx as Int32Array (-1 for null) for efficient transfer (Bug 5)
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

    let offscreen: OffscreenCanvas;
    try {
      offscreen = canvas.transferControlToOffscreen();
    } catch {
      // If transfer fails (e.g. already transferred), abort — next effect cycle
      // with a fresh canvas element will work.
      worker.terminate();
      return;
    }

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

    worker.postMessage(payload, transferables);

    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, [data, totalBases, alignedQueryG, gappedToQueryIdx, baseWidth, traceHeight]);

  // Force React to create a new <canvas> element each render cycle so
  // transferControlToOffscreen always gets a fresh canvas.
  // We use the data reference as a key to remount when sample changes.
  const canvasKey = `trace-${totalBases}-${data.baseLocations.length}`;

  return (
    <canvas
      key={canvasKey}
      ref={canvasRef}
      style={{ width: `${width}px`, height: `${traceHeight}px` }}
      className="trace-canvas-worker"
    />
  );
});
