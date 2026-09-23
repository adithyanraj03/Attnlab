import { test } from "node:test";
import assert from "node:assert";
import { runDemo } from "../dist/demo.js";

test("demo is deterministic", () => {
  assert.strictEqual(runDemo(), runDemo());
});

test("demo shows the standard state header", () => {
  const out = runDemo();
  assert.ok(out.includes("attnlab v1.0.0"));
  assert.ok(out.includes("seed 7"));
  assert.ok(out.includes("12 tokens · 8 types · 3 heads × d 8"));
  assert.ok(out.includes("causal on · T 1.00"));
});

test("demo prints the full token sequence", () => {
  const out = runDemo();
  assert.ok(out.includes("tokens: the cat sat on the mat the dog bit the cat tail"));
});

test("demo prints a 12x12 type-head matrix with row-stochastic rows", () => {
  const out = runDemo();
  const lines = out.split("\n");
  // the first matrix row looks like "  0  1.000 0.000 ..."; find it by shape
  const first = lines.findIndex((l) => /^\s*\d{1,2}\s+\d\.\d{3}/.test(l));
  assert.ok(first >= 0, "matrix not found in demo output");
  const matrixLines = lines.slice(first, first + 12);
  assert.strictEqual(matrixLines.length, 12);
  for (const line of matrixLines) {
    const cells = line.trim().split(/\s+/).slice(1);
    assert.strictEqual(cells.length, 12);
    const vals = cells.map(Number);
    const sum = vals.reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sum - 1) < 0.01, `row sums to ${sum}`); // 3-decimal rounding
  }
});

test("demo shows summary lines for the other heads", () => {
  const out = runDemo();
  assert.ok(out.includes("recency"));
  assert.ok(out.includes("entropy"));
  assert.ok(out.includes("deterministic: same seed"));
});

test("demo reflects config overrides", () => {
  const out = runDemo({ scenario: "blocks", heads: 1, headDim: 8, temperature: 2, causal: false, preset: "random", seed: 3 });
  assert.ok(out.includes("blocks"));
  assert.ok(out.includes("1 heads × d 8"));
  assert.ok(out.includes("causal off · T 2.00"));
  assert.ok(out.includes("seed 3"));
});
