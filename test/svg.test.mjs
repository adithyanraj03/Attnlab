import { test } from "node:test";
import assert from "node:assert";
import { heatColor, heatmapSVG, barsSVG, esc } from "../dist/svg.js";
import { fromRows } from "../dist/mat.js";

test("heatColor is an rgb() string across the ramp", () => {
  assert.ok(heatColor(0).startsWith("rgb("));
  assert.ok(heatColor(1).startsWith("rgb("));
  assert.notStrictEqual(heatColor(0), heatColor(1));
  // out-of-range values clamp
  assert.strictEqual(heatColor(-3), heatColor(0));
  assert.strictEqual(heatColor(3), heatColor(1));
});

test("heatmapSVG emits one rect per cell plus the frame", () => {
  const w = fromRows([[0.9, 0, 0], [0, 0.5, 0], [0.1, 0.2, 0.7]]);
  const svg = heatmapSVG(w, ["a", "b", "c"], { cell: 20, causal: true, labels: true });
  const rects = (svg.match(/<rect /g) || []).length;
  // 9 cells + 1 background
  assert.strictEqual(rects, 10);
  assert.ok(svg.includes("<svg"));
  assert.ok(svg.includes("</svg>"));
  assert.ok(svg.includes(">a<")); // labels present
});

test("heatmapSVG grays out the causal future and highlights the selected row", () => {
  const w = fromRows([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]);
  const plain = heatmapSVG(w, ["a", "b", "c"], { cell: 20, causal: true, labels: false });
  assert.ok(plain.includes("fill=\"#fbfaf7\""), "masked cells should use the mask fill");
  const sel = heatmapSVG(w, ["a", "b", "c"], { cell: 20, causal: false, labels: false, selectedRow: 1 });
  assert.ok(sel.includes("stroke=\"#252524\""), "selected row stroke missing");
  assert.ok(!plain.includes("stroke=\"#252524\""));
});

test("heatmapSVG emits row click handlers when requested", () => {
  const w = fromRows([[0.5, 0.5], [0.5, 0.5]]);
  const svg = heatmapSVG(w, ["a", "b"], {
    cell: 20, causal: false, labels: false, rowClick: (i) => `pick(${i})`,
  });
  assert.ok(svg.includes("onclick=\"pick(0)\""));
  assert.ok(svg.includes("onclick=\"pick(1)\""));
  const none = heatmapSVG(w, ["a", "b"], { cell: 20, causal: false, labels: false });
  assert.ok(!none.includes("onclick"));
});

test("barsSVG emits one row per value with labels and values", () => {
  const svg = barsSVG([0.5, 0.25, 1], ["a", "b", "c"], "#5b51c7");
  assert.ok(svg.includes(">a<"));
  assert.ok(svg.includes("0.500"));
  assert.ok(svg.includes("1.000"));
  assert.ok(svg.includes("#5b51c7"));
});

test("esc escapes markup", () => {
  assert.strictEqual(esc("<b>&</b>"), "&lt;b&gt;&amp;&lt;/b&gt;");
});
