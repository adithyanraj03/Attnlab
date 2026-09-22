import { test } from "node:test";
import assert from "node:assert";
import { fromRows, get } from "../dist/mat.js";
import { attention, multiHead } from "../dist/attn.js";

// Hand-computed single-head example (softmax worked out by hand here, not via the library).
// q = k = identity(2), v=[[10,0],[0,20]]  d=2  T=1
// scores = identity; scale = sqrt(2); scaled rows = [1/√2, 0] and [0, 1/√2]
// weights rows = [e^{1/√2}, 1] / (e^{1/√2} + 1) and its transpose
test("single-head attention matches the hand computation", () => {
  const q = fromRows([[1, 0], [0, 1]]);
  const k = fromRows([[1, 0], [0, 1]]);
  const v = fromRows([[10, 0], [0, 20]]);
  const e = Math.exp(1 / Math.SQRT2);
  const w0 = e / (e + 1);
  const w1 = 1 / (e + 1);
  const res = attention(q, k, v, { temperature: 1, causal: false });
  assert.ok(Math.abs(res.weights.a[0] - w0) < 1e-12, `w00=${res.weights.a[0]} want ${w0}`);
  assert.ok(Math.abs(res.weights.a[1] - w1) < 1e-12, `w01=${res.weights.a[1]} want ${w1}`);
  assert.ok(Math.abs(res.weights.a[2] - w1) < 1e-12, `w10=${res.weights.a[2]} want ${w1}`);
  assert.ok(Math.abs(res.weights.a[3] - w0) < 1e-12, `w11=${res.weights.a[3]} want ${w0}`);
  assert.ok(Math.abs(res.out.a[0] - w0 * 10) < 1e-12, `out0=${res.out.a[0]}`);
  assert.ok(Math.abs(res.out.a[1] - w1 * 20) < 1e-12, `out1=${res.out.a[1]}`);
  assert.ok(Math.abs(res.out.a[2] - w1 * 10) < 1e-12, `out2=${res.out.a[2]}`);
  assert.ok(Math.abs(res.out.a[3] - w0 * 20) < 1e-12, `out3=${res.out.a[3]}`);
});

test("attention weights are row-stochastic", () => {
  const q = fromRows([[1, 0], [0, 1], [1, 1]]);
  const k = fromRows([[1, 0], [0, 1], [1, 1]]);
  const v = fromRows([[1, 0], [0, 1], [1, 1]]);
  const res = attention(q, k, v, { temperature: 1, causal: false });
  for (let i = 0; i < 3; i++) {
    const s = get(res.weights, i, 0) + get(res.weights, i, 1) + get(res.weights, i, 2);
    assert.ok(Math.abs(s - 1) < 1e-9, `row ${i} sums to ${s}`);
  }
});

test("causal mask zeroes the future and leaves row 0 at its own key", () => {
  const q = fromRows([[1, 0], [0, 1], [1, 1]]);
  const k = fromRows([[1, 0], [0, 1], [1, 1]]);
  const v = fromRows([[1, 0], [0, 1], [0, 0]]);
  const res = attention(q, k, v, { temperature: 1, causal: true });
  // row 0 can only attend to key 0
  assert.strictEqual(res.weights.a[0 * 3 + 1], 0);
  assert.strictEqual(res.weights.a[0 * 3 + 2], 0);
  assert.ok(Math.abs(res.weights.a[0 * 3 + 0] - 1) < 1e-9);
  // row 2 may attend to keys 0,1,2 (all past incl. self)
  assert.ok(res.weights.a[2 * 3 + 0] >= 0);
  // every row still sums to 1
  for (let i = 0; i < 3; i++) {
    const s = res.weights.a[i * 3] + res.weights.a[i * 3 + 1] + res.weights.a[i * 3 + 2];
    assert.ok(Math.abs(s - 1) < 1e-9);
  }
});

test("larger temperature increases row entropy (flattens)", () => {
  const q = fromRows([[3, 0], [0, 3], [1, 1]]);
  const k = fromRows([[1, 0], [0.9, 0], [0.1, 0]]);
  const v = fromRows([[1, 0], [1, 0], [1, 0]]);
  const ent = (t) => {
    const res = attention(q, k, v, { temperature: t, causal: false });
    const w = res.weights.a;
    let h = 0;
    for (let j = 0; j < 3; j++) if (w[j] > 0) h -= w[j] * Math.log(w[j]);
    return h;
  };
  assert.ok(ent(4) > ent(1), "T=4 must be flatter than T=1");
  assert.ok(ent(1) > ent(0.25), "T=1 must be flatter than T=0.25");
});

test("attention rejects mismatched q/k/v shapes", () => {
  const q = fromRows([[1, 0]]);
  const k = fromRows([[1], [0]]);
  const v = fromRows([[10], [20]]);
  assert.throws(() => attention(q, k, v, { temperature: 1, causal: false }));
  const k2 = fromRows([[1, 0], [0, 1], [1, 1]]);
  const v2 = fromRows([[10], [20]]);
  assert.throws(() => attention(q, k2, v2, { temperature: 1, causal: false }));
});

test("multiHead concatenates per-head outputs and keeps per-head weights", () => {
  const x = fromRows([[1, 0, 0, 0], [0, 1, 0, 0]]); // (2x4)
  const dH = 2, heads = 2;
  // projections are (d x dH) = (4 x 2): first two identity columns
  const w2 = fromRows([[1, 0], [0, 1], [0, 0], [0, 0]]);
  const wq = [w2, w2];
  const wk = [w2, w2];
  const wv = [w2, w2];
  const res = multiHead(x, wq, wk, wv, heads, { temperature: 1, causal: false });
  assert.strictEqual(res.out.r, 2);
  assert.strictEqual(res.out.c, heads * dH); // 4
  assert.strictEqual(res.perHead.length, heads);
  for (let h = 0; h < heads; h++) {
    // each head's weight matrix is N x N (N = number of tokens)
    assert.strictEqual(res.perHead[h].weights.r, 2);
    assert.strictEqual(res.perHead[h].weights.c, 2);
    // each head's output is N x dH
    assert.strictEqual(res.perHead[h].out.c, dH);
  }
});

test("multiHead rejects a wrong projection count", () => {
  const x = fromRows([[1, 0], [0, 1]]);
  const w2 = fromRows([[1, 0], [0, 1]]);
  assert.throws(() => multiHead(x, [w2], [w2], [w2], 2, { temperature: 1, causal: false }));
});

test("multiHead weight matrices are N x N and scaled identity peaks on self", () => {
  const x = fromRows([[1, 0, 0], [0, 1, 0], [0, 0, 1]]); // (3x3) one-hot
  // W = 6*identity: self score = 6/sqrt(3) = 3.464, cross score = 0
  const w6 = fromRows([[6, 0, 0], [0, 6, 0], [0, 0, 6]]);
  const res = multiHead(x, [w6], [w6], [w6], 1, { temperature: 1, causal: false });
  const w = res.perHead[0].weights;
  assert.strictEqual(w.r, 3);
  assert.strictEqual(w.c, 3);
  // softmax([3.464, 0, 0]) -> 31.94 / 33.94 = 0.941
  assert.ok(get(w, 0, 0) > 0.9, `w00=${get(w, 0, 0)}`);
  assert.ok(get(w, 1, 1) > 0.9);
  assert.ok(get(w, 2, 2) > 0.9);
  assert.ok(get(w, 0, 1) < 0.1);
});
