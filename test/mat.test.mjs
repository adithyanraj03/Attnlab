import { test } from "node:test";
import assert from "node:assert";
import {
  mat, fromRows, matmul, addm, scalem, transpose, row, get,
  softmax, causalMask, argmax, entropy,
} from "../dist/mat.js";

test("mat allocates zeroed storage with the right shape", () => {
  const m = mat(3, 4);
  assert.strictEqual(m.r, 3);
  assert.strictEqual(m.c, 4);
  assert.strictEqual(m.a.length, 12);
  assert.ok(m.a.every((x) => x === 0));
});

test("mat rejects non-integer or negative dims", () => {
  assert.throws(() => mat(1.5, 2));
  assert.throws(() => mat(2, -1));
});

test("matmul matches a hand-computed product", () => {
  // [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
  const a = fromRows([[1, 2], [3, 4]]);
  const b = fromRows([[5, 6], [7, 8]]);
  const c = matmul(a, b);
  assert.deepStrictEqual([c.r, c.c], [2, 2]);
  assert.strictEqual(c.a[0], 19);
  assert.strictEqual(c.a[1], 22);
  assert.strictEqual(c.a[2], 43);
  assert.strictEqual(c.a[3], 50); // 3*6 + 4*8
});

test("matmul rejects mismatched inner dims", () => {
  assert.throws(() => matmul(fromRows([[1, 2]]), fromRows([[1], [2], [3]])));
});

test("rectangular matmul: (2x3)*(3x2) -> (2x2)", () => {
  const a = fromRows([[1, 0, 1], [0, 1, 0]]);
  const b = fromRows([[1, 0], [0, 1], [1, 1]]);
  const c = matmul(a, b);
  assert.deepStrictEqual([c.r, c.c], [2, 2]);
  assert.strictEqual(c.a[0], 2); // 1*1 + 0*0 + 1*1
  assert.strictEqual(c.a[1], 1); // 1*0 + 0*1 + 1*1
  assert.strictEqual(c.a[2], 0);
  assert.strictEqual(c.a[3], 1);
});

test("addm and scalem are elementwise", () => {
  const a = fromRows([[1, 2], [3, 4]]);
  const b = fromRows([[10, 20], [30, 40]]);
  const s = addm(a, b);
  assert.strictEqual(s.a[1], 22);
  assert.strictEqual(s.a[3], 44);
  const m = scalem(a, 2);
  assert.strictEqual(m.a[3], 8);
  // source untouched
  assert.strictEqual(a.a[1], 2);
});

test("transpose is an involutive swap", () => {
  const a = fromRows([[1, 2, 3], [4, 5, 6]]);
  const t = transpose(a);
  assert.deepStrictEqual([t.r, t.c], [3, 2]);
  assert.strictEqual(t.a[0], 1);
  assert.strictEqual(t.a[1], 4);
  assert.strictEqual(t.a[3], 5);
  assert.deepStrictEqual(transpose(t).a, a.a);
});

test("row returns a copy that does not alias the source", () => {
  const a = fromRows([[1, 2], [3, 4]]);
  const r1 = row(a, 1);
  r1[0] = 999;
  assert.strictEqual(a.a[2], 3, "mutating the copy must not touch the source");
  assert.throws(() => row(a, 2));
});

test("get reads the right element", () => {
  const a = fromRows([[1, 2, 3], [4, 5, 6]]);
  assert.strictEqual(get(a, 0, 2), 3);
  assert.strictEqual(get(a, 1, 0), 4);
});

test("softmax sums to 1 and is stable under a large offset", () => {
  const base = new Float64Array([0, 1, 2]);
  const shifted = new Float64Array([1000, 1001, 1002]);
  const a = softmax(base, 1);
  const b = softmax(shifted, 1);
  assert.ok(Math.abs(a[0] + a[1] + a[2] - 1) < 1e-12);
  for (let i = 0; i < 3; i++) assert.ok(Math.abs(a[i] - b[i]) < 1e-9, `stability at ${i}`);
  // larger logits -> larger weight
  assert.ok(a[2] > a[1] && a[1] > a[0]);
});

test("softmax temperature: large T flattens toward uniform, small T sharpens", () => {
  const logits = new Float64Array([0, 0, 3]);
  const flat = softmax(logits, 4);
  const sharp = softmax(logits, 0.25);
  const mid = softmax(logits, 1);
  // monotone flattening: the smallest weight grows with T
  assert.ok(flat[0] > mid[0] && mid[0] > sharp[0], "min weight must grow with T");
  // monotone sharpening: the max weight shrinks with T
  assert.ok(sharp[2] > mid[2] && mid[2] > flat[2], "max weight must shrink with T");
  assert.ok(sharp[2] > 0.99, "small T should concentrate on the max");
});

test("softmax rejects non-positive temperature", () => {
  assert.throws(() => softmax(new Float64Array([1, 2]), 0));
});

test("softmax with all -Infinity has no finite mass and throws", () => {
  assert.throws(() => softmax(new Float64Array([-Infinity, -Infinity]), 1));
});

test("causalMask is lower-triangular including the diagonal", () => {
  const m = causalMask(3);
  assert.strictEqual(m[0][0], true);
  assert.strictEqual(m[0][1], false);
  assert.strictEqual(m[1][1], true);
  assert.strictEqual(m[1][2], false);
  assert.strictEqual(m[2][0], true);
  assert.strictEqual(m[2][2], true);
});

test("argmax returns the first maximum on ties", () => {
  assert.strictEqual(argmax(new Float64Array([1, 5, 5, 2])), 1);
  assert.strictEqual(argmax(new Float64Array([3, 1, 2])), 0);
});

test("entropy of a uniform distribution is ln(n)", () => {
  const n = 4;
  const u = new Float64Array(n).fill(1 / n);
  assert.ok(Math.abs(entropy(u) - Math.log(n)) < 1e-12);
  // 0*log(0) is treated as 0
  const onehot = new Float64Array([0, 0, 1, 0]);
  assert.strictEqual(entropy(onehot), 0);
});
