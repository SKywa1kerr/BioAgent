import test from "node:test";
import assert from "node:assert/strict";
import {
  buildChromatogramRenderModel,
  drawChromatogram,
  findNearestBaseIndex,
  percentile,
} from "../src/components/workbench/chromatogramRender.js";

const data = {
  traces: {
    A: [0, 10, 20, 500, 20, 10, 0],
    T: [0, 8, 16, 24, 16, 8, 0],
    G: [0, 5, 10, 15, 10, 5, 0],
    C: [0, 2, 4, 6, 4, 2, 0],
  },
  quality: [40, 35, 20],
  baseCalls: "ATG",
  base_locations: [1, 3, 5],
  mixed_peaks: [1],
};

test("percentile handles empty input and robust rank", () => {
  assert.equal(percentile([], 0.985), 0);
  assert.equal(percentile([1, 2, 100], 0.5), 2);
});

test("buildChromatogramRenderModel clamps visible base range and preserves base labels", () => {
  const model = buildChromatogramRenderModel(data, {
    startPosition: 1,
    endPosition: 3,
    width: 300,
    height: 120,
    maxPointsPerPixel: 1,
  });

  assert.equal(model.visibleStartTrace, 0);
  assert.equal(model.visibleEndTrace, 7);
  assert.equal(model.baseLabels.length, 3);
  assert.deepEqual(model.baseLabels.map((b) => b.base), ["A", "T", "G"]);
  assert.equal(model.maxVal > 0, true);
});

test("buildChromatogramRenderModel downsamples traces when trace range exceeds canvas budget", () => {
  const longTrace = Array.from({ length: 1000 }, (_, i) => i % 50);
  const model = buildChromatogramRenderModel({
    traces: { A: longTrace, T: longTrace, G: longTrace, C: longTrace },
    quality: [],
    baseCalls: "AT",
    base_locations: [10, 990],
    mixed_peaks: [],
  }, {
    startPosition: 1,
    endPosition: 2,
    width: 100,
    height: 120,
    maxPointsPerPixel: 1,
  });

  assert.equal(model.step > 1, true);
  assert.equal(model.tracePoints.A.length <= 101, true);
});

test("buildChromatogramRenderModel preserves narrow off-stride peaks when downsampling", () => {
  const aTrace = Array.from({ length: 1000 }, () => 1);
  const flatTrace = Array.from({ length: 1000 }, () => 0);
  aTrace[501] = 10000;

  const model = buildChromatogramRenderModel({
    traces: { A: aTrace, T: flatTrace, G: flatTrace, C: flatTrace },
    quality: [],
    baseCalls: "AT",
    base_locations: [0, 999],
    mixed_peaks: [],
  }, {
    startPosition: 1,
    endPosition: 2,
    width: 100,
    height: 120,
    maxPointsPerPixel: 1,
  });

  assert.equal(model.step > 1, true);
  assert.equal(Math.max(...model.tracePoints.A.map((point) => point.value)), 10000);
  assert.equal(model.maxVal > 1, true);
});

test("buildChromatogramRenderModel returns an empty model for non-finite ranges", () => {
  for (const options of [
    { startPosition: NaN, endPosition: 2 },
    { startPosition: 1, endPosition: NaN },
  ]) {
    const model = buildChromatogramRenderModel(data, options);

    assert.deepEqual(model.baseLabels, []);
    assert.deepEqual(model.tracePoints, { A: [], T: [], G: [], C: [] });
    assert.equal(model.visibleStartTrace, 0);
    assert.equal(model.visibleEndTrace, 0);
    assert.equal(model.step, 1);
    assert.equal(model.maxVal, 1);
  }
});

test("findNearestBaseIndex returns nearest base within visible labels", () => {
  const model = buildChromatogramRenderModel(data, {
    startPosition: 1,
    endPosition: 3,
    width: 300,
    height: 120,
  });
  assert.equal(findNearestBaseIndex(model, model.baseLabels[1].x + 1), 1);
});

test("drawChromatogram renders traces, mixed peaks, and mutation markers", () => {
  const model = buildChromatogramRenderModel(data, {
    startPosition: 1,
    endPosition: 3,
    width: 300,
    height: 120,
  });
  const calls = [];
  const ctx = {
    set fillStyle(value) { calls.push(["fillStyle", value]); },
    set strokeStyle(value) { calls.push(["strokeStyle", value]); },
    set lineWidth(value) { calls.push(["lineWidth", value]); },
    set font(value) { calls.push(["font", value]); },
    set textAlign(value) { calls.push(["textAlign", value]); },
    fillRect: (...args) => calls.push(["fillRect", ...args]),
    beginPath: () => calls.push(["beginPath"]),
    moveTo: (...args) => calls.push(["moveTo", ...args]),
    lineTo: (...args) => calls.push(["lineTo", ...args]),
    stroke: () => calls.push(["stroke"]),
    fillText: (...args) => calls.push(["fillText", ...args]),
    arc: (...args) => calls.push(["arc", ...args]),
    closePath: () => calls.push(["closePath"]),
    fill: () => calls.push(["fill"]),
  };

  drawChromatogram(ctx, model, { dark: true, mutations: [{ position: 2 }] });

  assert.deepEqual(calls[0], ["fillStyle", "#0f172a"]);
  assert.deepEqual(calls[1], ["fillRect", 0, 0, 300, 120]);
  assert.equal(calls.some((call) => call[0] === "fillText" && call[1] === "A"), true);
  assert.equal(calls.some((call) => call[0] === "arc"), true);
  assert.equal(calls.some((call) => call[0] === "fillStyle" && call[1] === "#f87171"), true);
});
