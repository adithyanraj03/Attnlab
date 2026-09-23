import { test } from "node:test";
import assert from "node:assert";
import {
  SCENARIOS, DEFAULT_CONFIG, scenarioTokens, headPlan, forward,
} from "../dist/model.js";
import { get } from "../dist/mat.js";

test("scenarios expose the expected sequences", () => {
  assert.strictEqual(SCENARIOS.length, 3);
  assert.deepStrictEqual(scenarioTokens("narrative"),
    ["the", "cat", "sat", "on", "the", "mat", "the", "dog", "bit", "the", "cat", "tail"]);
  assert.strictEqual(scenarioTokens("alternating").length, 12);
  assert.strictEqual(scenarioTokens("blocks").length, 12);
  assert.throws(() => scenarioTokens("nope"));
});

test("headPlan: structured preset plans recency + type + randoms", () => {
  assert.deepStrictEqual(
    headPlan({ ...DEFAULT_CONFIG, heads: 1 }).map((p) => p.kind),
    ["type"]);
  assert.deepStrictEqual(
    headPlan({ ...DEFAULT_CONFIG, heads: 2 }).map((p) => p.kind),
    ["recency", "type"]);
  assert.deepStrictEqual(
    headPlan({ ...DEFAULT_CONFIG, heads: 4 }).map((p) => p.kind),
    ["recency", "type", "random", "random"]);
  assert.deepStrictEqual(
    headPlan({ ...DEFAULT_CONFIG, heads: 3, preset: "random" }).map((p) => p.kind),
    ["random", "random", "random"]);
});

test("forward is deterministic at the weight level", () => {
  const a = forward(DEFAULT_CONFIG);
  const b = forward(DEFAULT_CONFIG);
  assert.strictEqual(a.tokens.join(), b.tokens.join());
  for (let h = 0; h < a.heads; h++) {
    assert.deepStrictEqual(
      Array.from(a.multi.perHead[h].weights.a),
      Array.from(b.multi.perHead[h].weights.a),
      `head ${h} weights differ`);
  }
  assert.deepStrictEqual(a.entropies, b.entropies);
});

test("different seeds change the random head but not the designed heads", () => {
  const a = forward(DEFAULT_CONFIG);
  const b = forward({ ...DEFAULT_CONFIG, seed: 8 });
  const ra = a.plans.findIndex((p) => p.kind === "random");
  const rb = b.plans.findIndex((p) => p.kind === "random");
  // random heads must differ across seeds
  assert.notDeepStrictEqual(
    Array.from(a.multi.perHead[ra].weights.a),
    Array.from(b.multi.perHead[rb].weights.a));
  // type head is pure design: identical across seeds
  const ta = a.plans.findIndex((p) => p.kind === "type");
  const tb = b.plans.findIndex((p) => p.kind === "type");
  assert.deepStrictEqual(
    Array.from(a.multi.perHead[ta].weights.a),
    Array.from(b.multi.perHead[tb].weights.a));
});

test("x layout: one-hot type block + 3-dim position features", () => {
  const r = forward(DEFAULT_CONFIG);
  const vocabSize = r.vocab.length;
  for (let i = 0; i < r.n; i++) {
    // type block: exactly one 1
    let ones = 0;
    for (let t = 0; t < vocabSize; t++) if (get(r.x, i, t) === 1) ones++;
    assert.strictEqual(ones, 1, `token ${i} type block`);
    // the one is at the right type
    assert.strictEqual(get(r.x, i, r.typeOf[i]), 1);
    // position block
    const p = i / r.n;
    assert.ok(Math.abs(get(r.x, i, vocabSize) - p) < 1e-12);
    assert.ok(Math.abs(get(r.x, i, vocabSize + 1) - p * p) < 1e-12);
    assert.strictEqual(get(r.x, i, vocabSize + 2), 1);
  }
});

test("type head: same-type keys weigh strictly more than cross-type keys", () => {
  const r = forward(DEFAULT_CONFIG);
  const h = r.plans.findIndex((p) => p.kind === "type");
  const w = r.multi.perHead[h].weights;
  for (let i = 0; i < r.n; i++) {
    for (let j = 0; j <= i; j++) {
      const same = r.typeOf[j] === r.typeOf[i];
      for (let k = 0; k < j; k++) {
        const same2 = r.typeOf[k] === r.typeOf[i];
        if (same && !same2) {
          assert.ok(get(w, i, j) > get(w, i, k) + 1e-9,
            `row ${i}: same-type key ${j} not above cross-type key ${k}`);
        }
      }
    }
  }
});

test("type head: all visible same-type keys share equal weight", () => {
  const r = forward(DEFAULT_CONFIG);
  const h = r.plans.findIndex((p) => p.kind === "type");
  const w = r.multi.perHead[h].weights;
  for (let i = 0; i < r.n; i++) {
    const sameIdx = [];
    for (let j = 0; j <= i; j++) if (r.typeOf[j] === r.typeOf[i]) sameIdx.push(j);
    for (const j of sameIdx) {
      assert.ok(Math.abs(get(w, i, j) - get(w, i, sameIdx[0])) < 1e-12,
        `row ${i}: same-type weights differ (j=${j})`);
    }
  }
});

test("recency head (causal): every row peaks at the look-back centre j = round(i/√2)", () => {
  const r = forward({ ...DEFAULT_CONFIG, causal: true });
  const h = r.plans.findIndex((p) => p.kind === "recency");
  const w = r.multi.perHead[h].weights;
  for (let i = 0; i < r.n; i++) {
    let bi = -1;
    for (let j = 0; j <= i; j++) {
      if (bi === -1 || get(w, i, j) > get(w, i, bi)) bi = j;
    }
    // the score s(i,j) = (S/n²)·j·(√2·i − j) is a parabola in j with its
    // maximum at j = i/√2; row 0 has a single key so it peaks on itself
    const want = i === 0 ? 0 : Math.min(i, Math.round(i / Math.SQRT2));
    assert.strictEqual(bi, want, `row ${i} peaks at key ${bi}, want ${want}`);
  }
});

test("recency head decays parabolically away from its look-back centre", () => {
  const r = forward(DEFAULT_CONFIG);
  const h = r.plans.findIndex((p) => p.kind === "recency");
  const w = r.multi.perHead[h].weights;
  for (const i of [4, 7, 11]) {
    let p = -1;
    for (let j = 0; j <= i; j++) {
      if (p === -1 || get(w, i, j) > get(w, i, p)) p = j;
    }
    // strictly decays moving away from the peak in both directions
    for (let d = 1; d + p <= i; d++) {
      assert.ok(get(w, i, p + d) < get(w, i, p + d - 1), `row ${i} right side distance ${d}`);
    }
    for (let d = 1; d > 0 && p - d >= 0; d++) {
      assert.ok(get(w, i, p - d) < get(w, i, p - d + 1), `row ${i} left side distance ${d}`);
    }
  }
});

test("causal: no head attends to the future", () => {
  const r = forward(DEFAULT_CONFIG);
  for (let h = 0; h < r.heads; h++) {
    const w = r.multi.perHead[h].weights;
    for (let i = 0; i < r.n; i++) {
      for (let j = i + 1; j < r.n; j++) {
        assert.strictEqual(get(w, i, j), 0, `head ${h} row ${i} key ${j} not masked`);
      }
    }
  }
});

test("non-causal: weights are the same run without masking (differ where the mask acted)", () => {
  const a = forward({ ...DEFAULT_CONFIG, causal: true });
  const b = forward({ ...DEFAULT_CONFIG, causal: false });
  // some cells must differ (the mask changed something)
  let differ = 0;
  for (let h = 0; h < a.heads; h++) {
    for (let i = 0; i < a.n; i++) {
      for (let j = i + 1; j < a.n; j++) {
        if (Math.abs(get(a.multi.perHead[h].weights, i, j) - get(b.multi.perHead[h].weights, i, j)) > 1e-9) differ++;
      }
    }
  }
  assert.ok(differ > 0, "causal mask had no effect");
});

test("forward validates its config", () => {
  assert.throws(() => forward({ ...DEFAULT_CONFIG, heads: 0 }));
  assert.throws(() => forward({ ...DEFAULT_CONFIG, heads: 5 }));
  assert.throws(() => forward({ ...DEFAULT_CONFIG, headDim: 3 }));
  assert.throws(() => forward({ ...DEFAULT_CONFIG, headDim: 17 }));
  assert.throws(() => forward({ ...DEFAULT_CONFIG, temperature: 0 }));
});

test("output dim is heads * headDim", () => {
  const r = forward(DEFAULT_CONFIG);
  assert.strictEqual(r.multi.out.c, r.heads * r.headDim);
  assert.strictEqual(r.multi.out.r, r.n);
});

test("entropy and peak summaries have one entry per head", () => {
  const r = forward(DEFAULT_CONFIG);
  assert.strictEqual(r.entropies.length, r.heads);
  assert.strictEqual(r.peaks.length, r.heads);
  for (const e of r.entropies) assert.ok(e >= 0 && Number.isFinite(e));
  for (const p of r.peaks) assert.ok(p.value > 0 && p.value <= 1 + 1e-9);
});

test("headDim below vocab degrades gracefully (still a valid forward)", () => {
  const r = forward({ ...DEFAULT_CONFIG, headDim: 4 });
  for (let h = 0; h < r.heads; h++) {
    const w = r.multi.perHead[h].weights;
    for (let i = 0; i < r.n; i++) {
      let s = 0;
      for (let j = 0; j < r.n; j++) s += get(w, i, j);
      assert.ok(Math.abs(s - 1) < 1e-9, `head ${h} row ${i} sums to ${s}`);
    }
  }
});
