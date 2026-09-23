import { test } from "node:test";
import assert from "node:assert";
import { Attnlab } from "../dist/app.js";

test("Attnlab is exported with the public surface", () => {
  assert.strictEqual(Attnlab.version, "1.0.0");
  assert.strictEqual(typeof Attnlab.init, "function");
  assert.strictEqual(typeof Attnlab.selectRow, "function");
  assert.strictEqual(typeof Attnlab.refresh, "function");
});

test("default state mirrors DEFAULT_CONFIG", () => {
  assert.deepStrictEqual(Attnlab.state.cfg, {
    scenario: "narrative",
    seed: 7,
    heads: 3,
    headDim: 8,
    temperature: 1,
    causal: true,
    preset: "structured",
  });
  assert.strictEqual(Attnlab.state.row, 0);
});

test("the reseed step is a deterministic LCG (KAT)", () => {
  assert.strictEqual(Attnlab._lcgStep(7), 1282168116);
  assert.strictEqual(Attnlab._lcgStep(1282168116), 642666333);
  assert.strictEqual(Attnlab._lcgStep(0), 12345);
});

test("init/refresh/selectRow are no-ops without a DOM and never throw", () => {
  // no #al-root in node: init returns quietly
  assert.doesNotThrow(() => Attnlab.init());
  assert.doesNotThrow(() => Attnlab.refresh());
  assert.doesNotThrow(() => Attnlab.selectRow(3));
});
